# -*- coding: utf-8 -*-
"""
compute_external_baselines.py

Compute "strong external baselines" for Table Y:
(A) Academic link prediction: heuristics + (optional) Node2Vec/DeepWalk + (optional) KGE (TransE/DistMult/ComplEx/RotatE)
(B) Diabetes prediction (CSV): LR/RF + XGBoost + LightGBM
(C) Breast cancer prediction (OWL -> tabular): LR/RF + XGBoost + LightGBM

Fixes:
- Windows path loading for OWL: uses Path(...).resolve().as_uri()
- No dp.is_data_property() (incorrect in Owlready2)
- BreastCancer label uses data property "diagnosis" by default; maps M/B to 1/0
- Prevents "only one class" crashes with clear diagnostics

Example (Windows CMD, ONE LINE) — do NOT type Python code like 'run_tabular_suite(...)' into CMD:
python baselines\\compute_external_baselines.py --academic_owl baselines\\AcademicLink.owl --diabetes_csv baselines\\diabetes.csv --breast_owl baselines\\BreastCancer.owl --results_dir results --seeds 42 43 44 45 46 --neg_ratio 1.0

Optional (if some data properties are not features):
... --breast_ignore_dp hasValue,hasMeasurementDate --breast_min_numeric_coverage 0.7

If you want KGE (slow, requires pykeen+torch):
... --run_kge --kge_epochs 50

If you want Node2Vec/DeepWalk (requires node2vec+gensim):
... --run_node2vec

Outputs:
results/external_baselines_academic.csv
results/external_baselines_academic_summary.csv
results/external_baselines_diabetes.csv
results/external_baselines_diabetes_summary.csv
results/external_baselines_breast.csv
results/external_baselines_breast_summary.csv
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional tabular boosters
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

# Optional graph packages
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

try:
    from node2vec import Node2Vec
    _HAS_NODE2VEC = True
except Exception:
    _HAS_NODE2VEC = False

# Optional KGE via PyKEEN
try:
    import torch
    from pykeen.pipeline import pipeline as pykeen_pipeline
    from pykeen.triples import TriplesFactory
    _HAS_PYKEEN = True
except Exception:
    _HAS_PYKEEN = False

# OWL parsing
try:
    from owlready2 import get_ontology
    import owlready2
    from owlready2.prop import DataPropertyClass, ObjectPropertyClass
    _HAS_OWL = True
except Exception:
    _HAS_OWL = False

# Optional RDFLib (fallback OWL parsing)
try:
    from rdflib import Graph, URIRef, Literal
    from rdflib.namespace import RDF as RDF_NS, RDFS as RDFS_NS
    _HAS_RDFLIB = True
except Exception:
    _HAS_RDFLIB = False



# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def safe_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (scores >= thr).astype(int)
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # AUC requires both classes
    if len(np.unique(y_true)) == 2:
        out["AUC"] = float(roc_auc_score(y_true, scores))
    else:
        out["AUC"] = float("nan")
    return out

def pick_threshold_on_val(y_val: np.ndarray, s_val: np.ndarray) -> float:
    # maximize F1 on validation
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (s_val >= thr).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr

def validate_binary_labels(y: np.ndarray, name: str) -> None:
    vals, cnt = np.unique(y, return_counts=True)
    dist = {int(v): int(c) for v, c in zip(vals, cnt)}
    if len(vals) < 2:
        raise ValueError(
            f"[ERROR] {name}: y has only one class: {dist}. "
            f"This usually means the label field was not extracted correctly. "
            f"Please check your label column / data property name."
        )

@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def stratified_split(X: np.ndarray, y: np.ndarray, seed: int,
                     train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2) -> Split:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )
    return Split(X_train, y_train, X_val, y_val, X_test, y_test)


# -------------------------
# Tabular baselines (LR/RF/XGB/LGB)
# -------------------------

def tune_and_train_lr(split: Split, C_grid: List[float]) -> Tuple[Pipeline, float]:
    best_C, best_auc = None, -1.0
    for C in C_grid:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
        ])
        model.fit(split.X_train, split.y_train)
        if len(np.unique(split.y_val)) == 2:
            s_val = model.predict_proba(split.X_val)[:, 1]
            auc = roc_auc_score(split.y_val, s_val)
        else:
            auc = -1.0
        if auc > best_auc:
            best_auc = auc
            best_C = C

    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=float(best_C), max_iter=5000, solver="lbfgs"))
    ])
    X_tr = np.vstack([split.X_train, split.X_val])
    y_tr = np.concatenate([split.y_train, split.y_val])
    final.fit(X_tr, y_tr)
    return final, float(best_C)

def tune_and_train_rf(split: Split, n_estimators_grid: List[int], max_depth_grid: List[Optional[int]]) -> Tuple[RandomForestClassifier, Dict]:
    best_cfg, best_auc = None, -1.0
    for n in n_estimators_grid:
        for d in max_depth_grid:
            m = RandomForestClassifier(
                n_estimators=n, max_depth=d, random_state=0,
                n_jobs=-1, class_weight="balanced"
            )
            m.fit(split.X_train, split.y_train)
            if len(np.unique(split.y_val)) == 2:
                s_val = m.predict_proba(split.X_val)[:, 1]
                auc = roc_auc_score(split.y_val, s_val)
            else:
                auc = -1.0
            if auc > best_auc:
                best_auc = auc
                best_cfg = {"n_estimators": n, "max_depth": d}
    final = RandomForestClassifier(
        n_estimators=best_cfg["n_estimators"], max_depth=best_cfg["max_depth"],
        random_state=0, n_jobs=-1, class_weight="balanced"
    )
    X_tr = np.vstack([split.X_train, split.X_val])
    y_tr = np.concatenate([split.y_train, split.y_val])
    final.fit(X_tr, y_tr)
    return final, best_cfg

def tune_and_train_xgb(split: Split) -> Tuple[object, Dict]:
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed. Please install xgboost.")
    params_grid = []
    for n_estimators in [200, 400]:
        for max_depth in [3, 5]:
            for lr in [0.05, 0.1]:
                for subsample in [0.8, 1.0]:
                    for colsample_bytree in [0.8, 1.0]:
                        params_grid.append(dict(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=lr,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            reg_lambda=1.0
                        ))

    best_params, best_auc = None, -1.0
    for p in params_grid:
        m = xgb.XGBClassifier(
            **p,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=0
        )
        m.fit(split.X_train, split.y_train)
        if len(np.unique(split.y_val)) == 2:
            s_val = m.predict_proba(split.X_val)[:, 1]
            auc = roc_auc_score(split.y_val, s_val)
        else:
            auc = -1.0
        if auc > best_auc:
            best_auc = auc
            best_params = p

    final = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=0
    )
    X_tr = np.vstack([split.X_train, split.X_val])
    y_tr = np.concatenate([split.y_train, split.y_val])
    final.fit(X_tr, y_tr)
    return final, best_params

def tune_and_train_lgb(split: Split) -> Tuple[object, Dict]:
    if not _HAS_LGB:
        raise RuntimeError("lightgbm is not installed. Please install lightgbm.")
    params_grid = []
    for n_estimators in [300, 600]:
        for max_depth in [-1, 5, 10]:
            for lr in [0.05, 0.1]:
                for subsample in [0.8, 1.0]:
                    for colsample_bytree in [0.8, 1.0]:
                        params_grid.append(dict(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=lr,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            num_leaves=31
                        ))

    best_params, best_auc = None, -1.0
    for p in params_grid:
        m = lgb.LGBMClassifier(
            **p,
            n_jobs=-1,
            random_state=0
        )
        m.fit(split.X_train, split.y_train)
        if len(np.unique(split.y_val)) == 2:
            s_val = m.predict_proba(split.X_val)[:, 1]
            auc = roc_auc_score(split.y_val, s_val)
        else:
            auc = -1.0
        if auc > best_auc:
            best_auc = auc
            best_params = p

    final = lgb.LGBMClassifier(
        **best_params,
        n_jobs=-1,
        random_state=0
    )
    X_tr = np.vstack([split.X_train, split.X_val])
    y_tr = np.concatenate([split.y_train, split.y_val])
    final.fit(X_tr, y_tr)
    return final, best_params


def run_tabular_suite(name: str, X: np.ndarray, y: np.ndarray, seeds: List[int],
                      results_path: Path) -> pd.DataFrame:
    validate_binary_labels(y, name)

    rows = []
    for seed in seeds:
        set_global_seed(seed)
        split = stratified_split(X, y, seed=seed, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

        # LR
        lr_model, bestC = tune_and_train_lr(split, C_grid=[0.01, 0.1, 1.0, 10.0, 100.0])
        s_val = lr_model.predict_proba(split.X_val)[:, 1]
        thr = pick_threshold_on_val(split.y_val, s_val)
        s_test = lr_model.predict_proba(split.X_test)[:, 1]
        m = metrics_from_scores(split.y_test, s_test, thr)
        rows.append({"Dataset": name, "Baseline": "LR", "Seed": seed, **m, "BestParams": json.dumps({"C": bestC})})

        # RF
        rf_model, cfg = tune_and_train_rf(split, n_estimators_grid=[200, 500, 800], max_depth_grid=[None, 5, 10, 20])
        s_val = rf_model.predict_proba(split.X_val)[:, 1]
        thr = pick_threshold_on_val(split.y_val, s_val)
        s_test = rf_model.predict_proba(split.X_test)[:, 1]
        m = metrics_from_scores(split.y_test, s_test, thr)
        rows.append({"Dataset": name, "Baseline": "RF", "Seed": seed, **m, "BestParams": json.dumps(cfg)})

        # XGBoost
        if _HAS_XGB:
            xgb_model, cfg = tune_and_train_xgb(split)
            s_val = xgb_model.predict_proba(split.X_val)[:, 1]
            thr = pick_threshold_on_val(split.y_val, s_val)
            s_test = xgb_model.predict_proba(split.X_test)[:, 1]
            m = metrics_from_scores(split.y_test, s_test, thr)
            rows.append({"Dataset": name, "Baseline": "XGBoost", "Seed": seed, **m, "BestParams": json.dumps(cfg)})
        else:
            print("[WARN] xgboost not installed -> skipping XGBoost baseline.")

        # LightGBM
        if _HAS_LGB:
            lgb_model, cfg = tune_and_train_lgb(split)
            s_val = lgb_model.predict_proba(split.X_val)[:, 1]
            thr = pick_threshold_on_val(split.y_val, s_val)
            s_test = lgb_model.predict_proba(split.X_test)[:, 1]
            m = metrics_from_scores(split.y_test, s_test, thr)
            rows.append({"Dataset": name, "Baseline": "LightGBM", "Seed": seed, **m, "BestParams": json.dumps(cfg)})
        else:
            print("[WARN] lightgbm not installed -> skipping LightGBM baseline.")

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[OK] Wrote: {results_path}")

    # Summary
    summary = (
        df.groupby(["Dataset", "Baseline"])[["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = results_path.with_name(results_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote: {summary_path}")
    return df


# -------------------------
# OWL -> tabular (BreastCancer.owl)
# -------------------------


def _localname(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def _owl_uri_windows_safe(path: str) -> str:
    """
    Owlready2 on Windows can mis-handle file:// URIs into '/D:/...' (invalid).
    We'll prefer absolute filesystem paths when possible, but keep a URI helper too.
    """
    p = Path(path).expanduser().resolve()
    try:
        return p.as_uri()  # file:///D:/...
    except Exception:
        # fallback
        return "file:///" + p.as_posix().lstrip("/")


def _extract_tabular_from_owl_owlready2(
    owl_path: str,
    label_dp_name: str,
    ignore_dp_names: Optional[set] = None,
    min_numeric_coverage: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    OWL -> tabular using owlready2.
    NOTE: If owlready2 cannot see individuals/values correctly, caller should fallback to RDFLib.
    """
    if not _HAS_OWL:
        raise RuntimeError("owlready2 is not installed.")
    ignore_dp_names = ignore_dp_names or set()

    # Load using absolute file path first (Windows-safe)
    p = Path(owl_path).expanduser().resolve()
    onto = None
    last_err = None
    for candidate in (str(p), _owl_uri_windows_safe(str(p))):
        try:
            onto = get_ontology(candidate).load()
            break
        except Exception as e:
            last_err = e
            onto = None
    if onto is None:
        raise OSError(f"Failed to load ontology '{owl_path}'. Last error: {last_err}")

    # Find label data property
    label_dp = onto.search_one(iri=f"*{label_dp_name}")
    if label_dp is None:
        try:
            label_dp = getattr(onto, label_dp_name)
        except Exception:
            label_dp = None
    if label_dp is None:
        raise ValueError(f"[ERROR] Could not find label data property '{label_dp_name}' in {owl_path}")

    # Candidate individuals: those having a label value.
    inds = []
    labels_raw = []
    for ind in onto.individuals():
        vals = None
        # Prefer dp[ind] access; more robust than getattr for some owlready2 builds
        try:
            vals = label_dp[ind]
        except Exception:
            vals = None
        if vals is None:
            try:
                vals = getattr(ind, label_dp.python_name, None)
            except Exception:
                vals = None

        if vals is None:
            continue
        if isinstance(vals, list):
            if len(vals) == 0:
                continue
            v0 = vals[0]
        else:
            v0 = vals
        inds.append(ind)
        labels_raw.append(v0)

    if len(inds) == 0:
        raise ValueError(f"[ERROR] No individuals with label '{label_dp_name}' found in {owl_path}")

    # Collect all data properties (exclude label + ignored)
    all_dps = []
    for dp in onto.data_properties():
        dp_name = dp.name
        if dp_name == label_dp.name:
            continue
        if dp_name in ignore_dp_names:
            continue
        all_dps.append(dp)

    # Build dataframe: rows=individuals, cols=data properties
    data = []
    for ind in inds:
        row = {}
        for dp in all_dps:
            try:
                vals = dp[ind]
            except Exception:
                try:
                    vals = getattr(ind, dp.python_name, None)
                except Exception:
                    vals = None
            if isinstance(vals, list):
                v = vals[0] if len(vals) > 0 else None
            else:
                v = vals
            row[dp.name] = v
        data.append(row)

    dfX = pd.DataFrame(data)

    # Convert to numeric where possible
    for c in dfX.columns:
        dfX[c] = pd.to_numeric(dfX[c].map(safe_float), errors="coerce")

    # Drop columns that are mostly missing
    keep_cols = []
    for c in dfX.columns:
        coverage = float(dfX[c].notna().mean())
        if coverage >= min_numeric_coverage:
            keep_cols.append(c)
    dfX = dfX[keep_cols].copy()

    # Fill NaNs with median
    for c in dfX.columns:
        dfX[c] = dfX[c].fillna(dfX[c].median())

    # Map labels to {0,1}
    y = []
    for v in labels_raw:
        if isinstance(v, (int, float, np.integer, np.floating)):
            y.append(int(v))
        else:
            s = str(v).strip().lower()
            if s in ["m", "malignant", "1", "true", "yes", "positive"]:
                y.append(1)
            elif s in ["b", "benign", "0", "false", "no", "negative"]:
                y.append(0)
            else:
                fv = safe_float(s)
                if np.isnan(fv):
                    raise ValueError(f"[ERROR] Unknown label value '{v}' in {owl_path}.")
                y.append(int(fv))

    X = dfX.values.astype(np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] OWL(owlready2) extracted from {Path(owl_path).name}: "
          f"{X.shape[0]} instances, {X.shape[1]} numeric features, label='{label_dp_name}', "
          f"class_dist={dict(zip(*np.unique(y, return_counts=True)))}")

    validate_binary_labels(y, f"OWL:{Path(owl_path).name}")
    return X, y, dfX.columns.tolist()


def _extract_tabular_from_owl_rdflib(
    owl_path: str,
    label_dp_name: str,
    ignore_dp_names: Optional[set] = None,
    min_numeric_coverage: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    OWL -> tabular using RDFLib (robust on Windows and does not depend on owlready2 APIs).
    This is the fallback when owlready2 cannot detect individuals/values.
    """
    if not _HAS_RDFLIB:
        raise RuntimeError(
            "RDFLib is not installed (pip install rdflib). "
            "Cannot fallback from owlready2."
        )
    ignore_dp_names = ignore_dp_names or set()

    g = Graph()
    g.parse(str(Path(owl_path).expanduser().resolve()))

    # Resolve label predicate by localname (or full IRI if provided)
    if label_dp_name.startswith("http://") or label_dp_name.startswith("https://"):
        label_pred = URIRef(label_dp_name)
    else:
        preds = set(p for _, p, _ in g.triples((None, None, None)))
        matches = [p for p in preds if _localname(str(p)).lower() == label_dp_name.lower()]
        if len(matches) == 0:
            raise ValueError(f"[ERROR] Could not find predicate localname '{label_dp_name}' in {owl_path}")
        # choose the one with max labeled subjects (literal)
        best_p, best_c = None, -1
        for p in matches:
            c = sum(1 for _s, o in g.subject_objects(p) if isinstance(o, Literal))
            if c > best_c:
                best_c, best_p = c, p
        label_pred = best_p

    # Subjects with label
    sub_to_label = {}
    for s, o in g.subject_objects(label_pred):
        if isinstance(o, Literal) and s not in sub_to_label:
            sub_to_label[s] = o

    subjects = list(sub_to_label.keys())
    labels_raw = [sub_to_label[s] for s in subjects]

    if len(subjects) == 0:
        raise ValueError(f"[ERROR] No individuals with label '{label_dp_name}' found in {owl_path}")

    # Feature predicates: all predicates with literal objects, excluding label and ignored
    literal_preds = set()
    for s, p, o in g:
        if isinstance(o, Literal):
            literal_preds.add(p)

    ignore_lc = {x.lower() for x in ignore_dp_names}
    feature_preds = []
    for p in literal_preds:
        if p == label_pred:
            continue
        if p == RDFS_NS.label:
            continue
        ln = _localname(str(p))
        if ln.lower() in ignore_lc or str(p).lower() in ignore_lc:
            continue
        feature_preds.append(p)

    # Build dataframe
    data = []
    for s in subjects:
        row = {}
        for p in feature_preds:
            vals = [o for o in g.objects(s, p) if isinstance(o, Literal)]
            if len(vals) == 0:
                row[_localname(str(p))] = np.nan
            else:
                row[_localname(str(p))] = safe_float(vals[0])
        data.append(row)

    dfX = pd.DataFrame(data)

    # Convert to numeric
    for c in dfX.columns:
        dfX[c] = pd.to_numeric(dfX[c], errors="coerce")

    # Keep columns by coverage
    keep_cols = []
    for c in dfX.columns:
        coverage = float(dfX[c].notna().mean())
        if coverage >= min_numeric_coverage:
            keep_cols.append(c)
    dfX = dfX[keep_cols].copy()

    # Fill NaNs
    for c in dfX.columns:
        dfX[c] = dfX[c].fillna(dfX[c].median())

    # Map labels
    y = []
    for v in labels_raw:
        s = str(v).strip().lower()
        if s in ["m", "malignant", "1", "true", "yes", "positive"]:
            y.append(1)
        elif s in ["b", "benign", "0", "false", "no", "negative"]:
            y.append(0)
        else:
            fv = safe_float(s)
            if np.isnan(fv):
                raise ValueError(f"[ERROR] Unknown label value '{v}' in {owl_path}.")
            y.append(int(fv))

    X = dfX.values.astype(np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] OWL(rdflib) extracted from {Path(owl_path).name}: "
          f"{X.shape[0]} instances, {X.shape[1]} numeric features, label='{label_dp_name}', "
          f"class_dist={dict(zip(*np.unique(y, return_counts=True)))}")

    validate_binary_labels(y, f"OWL:{Path(owl_path).name}")
    return X, y, dfX.columns.tolist()


def extract_tabular_from_owl(
    owl_path: str,
    label_dp_name: str,
    ignore_dp_names: Optional[set] = None,
    min_numeric_coverage: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Try owlready2 first; if it cannot detect any labeled individuals (your error),
    automatically fallback to RDFLib parsing.
    """
    ignore_dp_names = ignore_dp_names or set()

    # 1) Try owlready2
    if _HAS_OWL:
        try:
            return _extract_tabular_from_owl_owlready2(
                owl_path=owl_path,
                label_dp_name=label_dp_name,
                ignore_dp_names=ignore_dp_names,
                min_numeric_coverage=min_numeric_coverage,
            )
        except Exception as e:
            msg = str(e)
            # Fallback only on "no individuals with label" or other owlready2 oddities
            if ("No individuals with label" in msg) or ("Invalid argument" in msg) or ("Failed to load ontology" in msg):
                print(f"[WARN] owlready2 failed for BreastCancer OWL: {e}")
                if _HAS_RDFLIB:
                    print("[INFO] Falling back to RDFLib OWL parsing...")
                    return _extract_tabular_from_owl_rdflib(
                        owl_path=owl_path,
                        label_dp_name=label_dp_name,
                        ignore_dp_names=ignore_dp_names,
                        min_numeric_coverage=min_numeric_coverage,
                    )
                raise
            raise

    # 2) owlready2 unavailable -> use RDFLib
    return _extract_tabular_from_owl_rdflib(
        owl_path=owl_path,
        label_dp_name=label_dp_name,
        ignore_dp_names=ignore_dp_names,
        min_numeric_coverage=min_numeric_coverage,
    )



# -------------------------
# Academic baselines
# -------------------------

def extract_academic_graph_from_owl(
    owl_path: str,
    author_class_name: str = "Author",
    direct_coauthor_op: str = "hasCoAuthor",
    author1_op: str = "hasAuthor1",
    author2_op: str = "hasAuthor2",
) -> Tuple[List[str], List[Tuple[str, str]]]:
    if not _HAS_OWL:
        raise RuntimeError("owlready2 is not installed.")
    if not _HAS_NX:
        raise RuntimeError("networkx is not installed.")

    uri = _owl_uri(owl_path)
    onto = get_ontology(uri).load()

    # try to find Author class
    author_cls = onto.search_one(iri=f"*{author_class_name}")
    if author_cls is None:
        try:
            author_cls = getattr(onto, author_class_name)
        except Exception:
            author_cls = None
    if author_cls is None:
        raise ValueError(f"[ERROR] Could not find class '{author_class_name}' in {owl_path}")

    authors = [a for a in author_cls.instances()]
    author_ids = [str(a.name) for a in authors]

    # 1) Try direct co-author edges: Author --hasCoAuthor--> Author
    edges = set()
    op = onto.search_one(iri=f"*{direct_coauthor_op}")
    if op is None:
        try:
            op = getattr(onto, direct_coauthor_op)
        except Exception:
            op = None

    if op is not None:
        for a in authors:
            try:
                neigh = getattr(a, op.python_name, [])
            except Exception:
                neigh = []
            for b in neigh:
                u, v = str(a.name), str(b.name)
                if u == v:
                    continue
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))

    # 2) If empty, fallback to reified links: Link hasAuthor1/hasAuthor2
    if len(edges) == 0:
        op1 = onto.search_one(iri=f"*{author1_op}")
        op2 = onto.search_one(iri=f"*{author2_op}")
        if op1 is None:
            try:
                op1 = getattr(onto, author1_op)
            except Exception:
                op1 = None
        if op2 is None:
            try:
                op2 = getattr(onto, author2_op)
            except Exception:
                op2 = None
        if op1 is None or op2 is None:
            raise ValueError(
                f"[ERROR] Could not find direct edges '{direct_coauthor_op}' "
                f"and also could not find reified props '{author1_op}', '{author2_op}'."
            )

        for ind in onto.individuals():
            try:
                a1 = getattr(ind, op1.python_name, [])
                a2 = getattr(ind, op2.python_name, [])
            except Exception:
                continue
            if isinstance(a1, list) and len(a1) > 0 and isinstance(a2, list) and len(a2) > 0:
                u, v = str(a1[0].name), str(a2[0].name)
                if u == v:
                    continue
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))

    edges = list(edges)
    print(f"[INFO] Academic graph extracted: #Authors={len(author_ids)}, #PositiveEdges={len(edges)}")
    if len(edges) == 0:
        raise ValueError("[ERROR] No positive edges found in AcademicLink.owl. "
                         "Please confirm which object property encodes co-authorship.")
    return author_ids, edges


def sample_negative_edges(
    nodes: List[str],
    pos_edge_set: set,
    num_samples: int,
    rng: np.random.RandomState,
    max_attempts: int = 5_000_000
) -> List[Tuple[str, str]]:
    neg = set()
    n = len(nodes)
    attempts = 0
    while len(neg) < num_samples and attempts < max_attempts:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            attempts += 1
            continue
        u, v = nodes[i], nodes[j]
        if u < v:
            e = (u, v)
        else:
            e = (v, u)
        if e in pos_edge_set or e in neg:
            attempts += 1
            continue
        neg.add(e)
        attempts += 1

    if len(neg) < num_samples:
        print(f"[WARN] Negative sampling stopped early: got {len(neg)}/{num_samples}. "
              f"Graph may be dense or max_attempts too low.")
    return list(neg)


def edge_split(pos_edges: List[Tuple[str, str]], seed: int,
               train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pos_edges))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_pos = [pos_edges[i] for i in train_idx]
    val_pos = [pos_edges[i] for i in val_idx]
    test_pos = [pos_edges[i] for i in test_idx]
    return train_pos, val_pos, test_pos


def build_training_graph(train_pos: List[Tuple[str, str]]) -> "nx.Graph":
    G = nx.Graph()
    for u, v in train_pos:
        G.add_edge(u, v)
    return G


def heuristic_features(G: "nx.Graph", edges: List[Tuple[str, str]]) -> np.ndarray:
    # Using training graph only
    neigh = {n: set(G.neighbors(n)) for n in G.nodes()}
    deg = {n: len(neigh[n]) for n in neigh}

    X = np.zeros((len(edges), 4), dtype=np.float32)
    for i, (u, v) in enumerate(edges):
        Nu = neigh.get(u, set())
        Nv = neigh.get(v, set())
        inter = Nu & Nv
        union = Nu | Nv
        cn = len(inter)
        jacc = cn / len(union) if len(union) > 0 else 0.0
        aa = 0.0
        for w in inter:
            dw = deg.get(w, 0)
            if dw > 1:
                aa += 1.0 / math.log(dw)
        pa = float(deg.get(u, 0) * deg.get(v, 0))
        X[i, :] = [cn, jacc, aa, pa]
    return X


def fit_lr_on_features(X_tr, y_tr, X_val, y_val):
    # LR over heuristic features
    bestC, best_auc = None, -1.0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))])
        m.fit(X_tr, y_tr)
        s_val = m.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, s_val) if len(np.unique(y_val)) == 2 else -1.0
        if auc > best_auc:
            best_auc, bestC = auc, C
    final = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(C=float(bestC), max_iter=5000, solver="lbfgs"))])
    final.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    return final, float(bestC)


def run_academic_heuristics(nodes, pos_edges, seeds, neg_ratio, results_path: Path) -> pd.DataFrame:
    rows = []
    pos_set = set(pos_edges)

    for seed in seeds:
        set_global_seed(seed)
        rng = np.random.RandomState(seed)

        train_pos, val_pos, test_pos = edge_split(pos_edges, seed)
        # negatives per split
        train_neg = sample_negative_edges(nodes, pos_set, int(len(train_pos) * neg_ratio), rng)
        val_neg   = sample_negative_edges(nodes, pos_set, int(len(val_pos) * neg_ratio), rng)
        test_neg  = sample_negative_edges(nodes, pos_set, int(len(test_pos) * neg_ratio), rng)

        Gtr = build_training_graph(train_pos)

        X_tr = heuristic_features(Gtr, train_pos + train_neg)
        y_tr = np.array([1]*len(train_pos) + [0]*len(train_neg), dtype=np.int64)

        X_val = heuristic_features(Gtr, val_pos + val_neg)
        y_val = np.array([1]*len(val_pos) + [0]*len(val_neg), dtype=np.int64)

        X_te = heuristic_features(Gtr, test_pos + test_neg)
        y_te = np.array([1]*len(test_pos) + [0]*len(test_neg), dtype=np.int64)

        validate_binary_labels(y_tr, "Academic/train")
        validate_binary_labels(y_te, "Academic/test")

        model, bestC = fit_lr_on_features(X_tr, y_tr, X_val, y_val)
        s_val = model.predict_proba(X_val)[:, 1]
        thr = pick_threshold_on_val(y_val, s_val)
        s_te = model.predict_proba(X_te)[:, 1]
        m = metrics_from_scores(y_te, s_te, thr)

        rows.append({
            "Dataset": "Academic",
            "Baseline": "Heuristics(LR)",
            "Seed": seed,
            **m,
            "BestParams": json.dumps({"C": bestC, "neg_ratio": neg_ratio})
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[OK] Wrote: {results_path}")

    summary = (
        df.groupby(["Dataset", "Baseline"])[["Accuracy","Precision","Recall","F1-Score","AUC"]]
        .agg(["mean","std"]).reset_index()
    )
    summary_path = results_path.with_name(results_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote: {summary_path}")
    return df


def run_academic_node2vec(nodes, pos_edges, seeds, neg_ratio, results_path: Path,
                          mode: str = "node2vec") -> Optional[pd.DataFrame]:
    if not _HAS_NODE2VEC:
        print("[WARN] node2vec not installed -> skipping Node2Vec/DeepWalk baselines.")
        return None
    if not _HAS_NX:
        print("[WARN] networkx not installed -> skipping Node2Vec/DeepWalk baselines.")
        return None

    rows = []
    pos_set = set(pos_edges)

    # DeepWalk is approximated by node2vec with p=q=1
    if mode.lower() == "deepwalk":
        p, q = 1.0, 1.0
        tag = "DeepWalk+LR"
    else:
        # typical Node2Vec bias
        p, q = 0.25, 4.0
        tag = "Node2Vec+LR"

    for seed in seeds:
        set_global_seed(seed)
        rng = np.random.RandomState(seed)

        train_pos, val_pos, test_pos = edge_split(pos_edges, seed)
        train_neg = sample_negative_edges(nodes, pos_set, int(len(train_pos) * neg_ratio), rng)
        val_neg   = sample_negative_edges(nodes, pos_set, int(len(val_pos) * neg_ratio), rng)
        test_neg  = sample_negative_edges(nodes, pos_set, int(len(test_pos) * neg_ratio), rng)

        Gtr = build_training_graph(train_pos)
        if Gtr.number_of_nodes() == 0:
            print("[WARN] Training graph has 0 nodes -> skipping")
            continue

        # Fit embeddings on TRAIN graph only (no leakage)
        n2v = Node2Vec(
            Gtr, dimensions=128, walk_length=40, num_walks=10,
            p=p, q=q, workers=max(1, os.cpu_count() or 2)
        )
        w2v = n2v.fit(window=10, min_count=1, batch_words=128)
        emb = {str(n): w2v.wv[str(n)] for n in Gtr.nodes() if str(n) in w2v.wv}

        def edge_vec(u, v):
            eu = emb.get(u)
            ev = emb.get(v)
            if eu is None or ev is None:
                return None
            return eu * ev  # hadamard

        def make_edge_matrix(pairs):
            feats = []
            keep = []
            for (u, v) in pairs:
                vec = edge_vec(u, v)
                if vec is None:
                    # unseen nodes in training graph -> skip
                    continue
                feats.append(vec)
                keep.append((u, v))
            if len(feats) == 0:
                return np.zeros((0, 128), dtype=np.float32), keep
            return np.vstack(feats).astype(np.float32), keep

        X_tr_pos, _ = make_edge_matrix(train_pos)
        X_tr_neg, _ = make_edge_matrix(train_neg)
        X_val_pos, _ = make_edge_matrix(val_pos)
        X_val_neg, _ = make_edge_matrix(val_neg)
        X_te_pos, _ = make_edge_matrix(test_pos)
        X_te_neg, _ = make_edge_matrix(test_neg)

        # If too many skipped edges, warn
        if min(len(X_tr_pos), len(X_tr_neg), len(X_val_pos), len(X_val_neg), len(X_te_pos), len(X_te_neg)) == 0:
            print(f"[WARN] Too many edges missing embeddings (seed={seed}). "
                  f"Consider using a larger training graph.")
            continue

        X_tr = np.vstack([X_tr_pos, X_tr_neg])
        y_tr = np.array([1]*len(X_tr_pos) + [0]*len(X_tr_neg), dtype=np.int64)

        X_val = np.vstack([X_val_pos, X_val_neg])
        y_val = np.array([1]*len(X_val_pos) + [0]*len(X_val_neg), dtype=np.int64)

        X_te = np.vstack([X_te_pos, X_te_neg])
        y_te = np.array([1]*len(X_te_pos) + [0]*len(X_te_neg), dtype=np.int64)

        validate_binary_labels(y_tr, "Academic(train)/Node2Vec")
        validate_binary_labels(y_te, "Academic(test)/Node2Vec")

        model, bestC = fit_lr_on_features(X_tr, y_tr, X_val, y_val)
        s_val = model.predict_proba(X_val)[:, 1]
        thr = pick_threshold_on_val(y_val, s_val)
        s_te = model.predict_proba(X_te)[:, 1]
        m = metrics_from_scores(y_te, s_te, thr)

        rows.append({
            "Dataset": "Academic",
            "Baseline": tag,
            "Seed": seed,
            **m,
            "BestParams": json.dumps({"C": bestC, "p": p, "q": q, "neg_ratio": neg_ratio})
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[OK] Wrote: {results_path}")

    summary = (
        df.groupby(["Dataset","Baseline"])[["Accuracy","Precision","Recall","F1-Score","AUC"]]
        .agg(["mean","std"]).reset_index()
    )
    summary_path = results_path.with_name(results_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote: {summary_path}")
    return df


def run_academic_kge(nodes, pos_edges, seeds, neg_ratio, results_path: Path,
                     kge_epochs: int = 50) -> Optional[pd.DataFrame]:
    if not _HAS_PYKEEN:
        print("[WARN] pykeen/torch not installed -> skipping KGE baselines.")
        return None

    rows = []
    pos_set = set(pos_edges)

    # Use one relation label for KGE
    rel = "hasCoAuthor"

    for seed in seeds:
        set_global_seed(seed)
        rng = np.random.RandomState(seed)

        train_pos, val_pos, test_pos = edge_split(pos_edges, seed)
        train_neg = sample_negative_edges(nodes, pos_set, int(len(train_pos) * neg_ratio), rng)
        val_neg   = sample_negative_edges(nodes, pos_set, int(len(val_pos) * neg_ratio), rng)
        test_neg  = sample_negative_edges(nodes, pos_set, int(len(test_pos) * neg_ratio), rng)

        # Labeled triples for training: include both directions (undirected)
        train_triples = []
        for (u, v) in train_pos:
            train_triples.append([u, rel, v])
            train_triples.append([v, rel, u])

        train_triples = np.array(train_triples, dtype=str)
        tf = TriplesFactory.from_labeled_triples(train_triples)

        for model_name in ["TransE", "DistMult", "ComplEx", "RotatE"]:
            print(f"[INFO] Training KGE {model_name} (seed={seed}, epochs={kge_epochs}) ...")
            result = pykeen_pipeline(
                training=tf,
                model=model_name,
                random_seed=seed,
                training_kwargs=dict(num_epochs=kge_epochs),
                stopper="early",
                stopper_kwargs=dict(frequency=5, patience=3, metric="loss"),
            )
            model = result.model
            model = model.to("cpu")
            model.eval()

            # score edges in val/test
            def score_pairs(pairs):
                # PyKEEN needs mapped IDs; easiest is to reuse factory mapping
                h = []
                r = []
                t = []
                for (u, v) in pairs:
                    if u not in tf.entity_to_id or v not in tf.entity_to_id:
                        continue
                    h.append(tf.entity_to_id[u])
                    t.append(tf.entity_to_id[v])
                    r.append(tf.relation_to_id[rel])
                if len(h) == 0:
                    return np.array([], dtype=np.float32)
                hrt = torch.tensor(np.stack([h, r, t], axis=1), dtype=torch.long)
                with torch.no_grad():
                    s = model.score_hrt(hrt).cpu().numpy()
                # sigmoid to [0,1]
                s = 1.0 / (1.0 + np.exp(-s))
                return s.astype(np.float32)

            # Validation (calibrate threshold)
            val_pairs = val_pos + val_neg
            y_val = np.array([1]*len(val_pos) + [0]*len(val_neg), dtype=np.int64)
            s_val = score_pairs(val_pairs)
            if len(s_val) != len(y_val):
                # if some entities missing mapping, align by skipping; simplest: skip KGE in this case
                print("[WARN] Some entities not mapped in KGE factory; skipping this KGE run.")
                continue
            thr = pick_threshold_on_val(y_val, s_val)

            # Test
            test_pairs = test_pos + test_neg
            y_te = np.array([1]*len(test_pos) + [0]*len(test_neg), dtype=np.int64)
            s_te = score_pairs(test_pairs)
            if len(s_te) != len(y_te):
                print("[WARN] Some entities not mapped in KGE factory; skipping this KGE run.")
                continue

            m = metrics_from_scores(y_te, s_te, thr)
            rows.append({
                "Dataset": "Academic",
                "Baseline": f"{model_name}(KGE)",
                "Seed": seed,
                **m,
                "BestParams": json.dumps({"epochs": kge_epochs, "neg_ratio": neg_ratio})
            })

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[OK] Wrote: {results_path}")

    summary = (
        df.groupby(["Dataset","Baseline"])[["Accuracy","Precision","Recall","F1-Score","AUC"]]
        .agg(["mean","std"]).reset_index()
    )
    summary_path = results_path.with_name(results_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote: {summary_path}")
    return df


# -------------------------
# Diabetes CSV loader
# -------------------------

def load_diabetes_csv(path: str, label_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)

    # choose label column
    if label_col is None:
        candidates = ["Outcome", "outcome", "label", "Label", "target", "Target", "class", "Class"]
        for c in candidates:
            if c in df.columns:
                label_col = c
                break
    if label_col is None or label_col not in df.columns:
        raise ValueError(f"[ERROR] Could not infer label column in {path}. "
                         f"Please pass --diabetes_label_col <colname>")

    y = df[label_col].values
    # Map strings to int
    if df[label_col].dtype == object:
        y2 = []
        for v in y:
            s = str(v).strip().lower()
            if s in ["1", "true", "yes", "positive"]:
                y2.append(1)
            elif s in ["0", "false", "no", "negative"]:
                y2.append(0)
            else:
                y2.append(int(float(s)))
        y = np.array(y2, dtype=np.int64)
    else:
        y = y.astype(int)

    X = df.drop(columns=[label_col]).copy()

    # keep numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # fill
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())

    return X.values.astype(np.float32), y.astype(np.int64), list(X.columns)


# -------------------------
# Main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--academic_owl", type=str, default=None, help="Path to AcademicLink.owl")
    ap.add_argument("--diabetes_csv", type=str, default=None, help="Path to diabetes.csv")
    ap.add_argument("--breast_owl", type=str, default=None, help="Path to BreastCancer.owl")

    ap.add_argument("--results_dir", type=str, default="results", help="Output directory")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="Negative-to-positive ratio for academic link prediction")

    ap.add_argument("--diabetes_label_col", type=str, default=None, help="Label column name for diabetes.csv (optional)")

    ap.add_argument("--breast_label_dp", type=str, default="diagnosis", help="Breast cancer OWL label data property name (default: diagnosis)")
    ap.add_argument(
        "--breast_ignore_dp",
        type=str,
        default="",
        help=(
            "Comma-separated data property names to ignore when converting BreastCancer.owl to tabular. "
            "Example: --breast_ignore_dp hasValue,hasMeasurementDate"
        ),
    )
    ap.add_argument(
        "--breast_min_numeric_coverage",
        type=float,
        default=0.7,
        help=(
            "Keep a feature column only if at least this fraction of individuals have a numeric value "
            "(default: 0.7)."
        ),
    )
    ap.add_argument("--academic_author_class", type=str, default="Author", help="Author class name (default: Author)")
    ap.add_argument("--academic_coauthor_op", type=str, default="hasCoAuthor", help="Direct coauthor object property (default: hasCoAuthor)")
    ap.add_argument("--academic_author1_op", type=str, default="hasAuthor1", help="Reified edge author1 op (default: hasAuthor1)")
    ap.add_argument("--academic_author2_op", type=str, default="hasAuthor2", help="Reified edge author2 op (default: hasAuthor2)")

    ap.add_argument("--run_node2vec", action="store_true", help="Also run Node2Vec and DeepWalk (requires node2vec, gensim)")
    ap.add_argument("--run_kge", action="store_true", help="Also run KGE baselines (requires pykeen, torch)")
    ap.add_argument("--kge_epochs", type=int, default=50, help="KGE epochs (default: 50)")

    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.results_dir)

    # 1) Academic baselines
    if args.academic_owl:
        nodes, pos_edges = extract_academic_graph_from_owl(
            args.academic_owl,
            author_class_name=args.academic_author_class,
            direct_coauthor_op=args.academic_coauthor_op,
            author1_op=args.academic_author1_op,
            author2_op=args.academic_author2_op,
        )

        print("[1/3] Academic heuristics baseline ...")
        run_academic_heuristics(
            nodes, pos_edges, args.seeds, args.neg_ratio,
            outdir / "external_baselines_academic.csv"
        )

        if args.run_node2vec:
            print("[1/3] Academic DeepWalk baseline ...")
            run_academic_node2vec(
                nodes, pos_edges, args.seeds, args.neg_ratio,
                outdir / "external_baselines_academic_deepwalk.csv",
                mode="deepwalk"
            )
            print("[1/3] Academic Node2Vec baseline ...")
            run_academic_node2vec(
                nodes, pos_edges, args.seeds, args.neg_ratio,
                outdir / "external_baselines_academic_node2vec.csv",
                mode="node2vec"
            )

        if args.run_kge:
            print("[1/3] Academic KGE baselines ...")
            run_academic_kge(
                nodes, pos_edges, args.seeds, args.neg_ratio,
                outdir / "external_baselines_academic_kge.csv",
                kge_epochs=args.kge_epochs
            )

    # 2) Diabetes tabular baselines
    if args.diabetes_csv:
        print("[2/3] Diabetes tabular baselines...")
        Xd, yd, feat = load_diabetes_csv(args.diabetes_csv, label_col=args.diabetes_label_col)
        run_tabular_suite("Diabetes", Xd, yd, args.seeds, outdir / "external_baselines_diabetes.csv")

    # 3) Breast cancer tabular baselines from OWL
    if args.breast_owl:
        print("[3/3] Breast cancer tabular baselines...")
        # Ignore obviously non-feature columns if present
                # Ignore obviously non-feature columns if present (default list) + user-specified ignores
        ignore = {"hasValue", "hasMeasurementDate", "Unnamed:_32", "Unnamed:__32", "Unnamed__32"}
        if args.breast_ignore_dp:
            extra = {s.strip() for s in args.breast_ignore_dp.split(",") if s.strip()}
            ignore |= extra

        Xb, yb, feats = extract_tabular_from_owl(
            args.breast_owl,
            label_dp_name=args.breast_label_dp,
            ignore_dp_names=ignore,
            min_numeric_coverage=args.breast_min_numeric_coverage,
        )
        run_tabular_suite("Breast Cancer", Xb, yb, args.seeds, outdir / "external_baselines_breast.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
        sys.exit(1)
