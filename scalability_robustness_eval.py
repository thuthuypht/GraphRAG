#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scalability_robustness_eval.py

Experiments:
(1) KG-size slices: 25/50/75/100% of non-label edges retained
(2) Robustness: 5% / 10% triple-drop noise
(3) Optional injection: 1% random schema-compatible edges

Reports:
  - Runtime (seconds)
  - Peak memory (MB)
  - AUC and F1-score

Outputs:
  out_dir/scalability_runs.csv
  out_dir/scalability_summary.csv

Run (Windows, single line):
  python scalability_robustness_eval.py --owl data/AcademicLink_GNN_dense_colored.owl --out_dir results_scal --eval_pairs 300
"""

import os
import time
import threading
import argparse
from typing import Dict, Tuple, List
from dataclasses import dataclass

import psutil
import numpy as np
import pandas as pd

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# Your project module (graph building + embeddings)
from modules import graphrag_academiclink as ga


Pair = Tuple[URIRef, URIRef]


# ------------------- Time + Memory -------------------
@dataclass
class TimeMem:
    seconds: float
    peak_rss_mb: float


def _mem_sampler(stop_evt, interval, out_dict):
    proc = psutil.Process(os.getpid())
    peak = 0
    while not stop_evt.is_set():
        try:
            peak = max(peak, proc.memory_info().rss)
        except Exception:
            pass
        stop_evt.wait(interval)
    out_dict["peak"] = peak


def measure(fn, *args, **kwargs):
    stop_evt = threading.Event()
    out = {"peak": 0}
    t = threading.Thread(target=_mem_sampler, args=(stop_evt, 0.1, out), daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        res = fn(*args, **kwargs)
    finally:
        t1 = time.perf_counter()
        stop_evt.set()
        t.join(timeout=2)
    return res, TimeMem(t1 - t0, out["peak"] / (1024 * 1024))


# ------------------- Helpers -------------------
def canonical(u: URIRef, v: URIRef) -> Pair:
    return (u, v) if str(u) <= str(v) else (v, u)


def _local_name(x) -> str:
    s = str(x)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def build_pred_map(rdf_g: Graph) -> Dict[str, URIRef]:
    # Must match your ontology (as in your screenshot)
    need = ["hasCoAuthor", "hasLink", "affiliatedWith", "authored", "relatedTo"]
    found = ga.find_predicates_by_localname(rdf_g, need)
    missing = [k for k in need if k not in found]
    if missing:
        raise ValueError(f"Missing required predicates in OWL: {missing}. Found={list(found.keys())}")
    return found


# ------------------- KG Slicing -------------------
def slice_graph(
    rdf_full: Graph,
    pm: Dict[str, URIRef],
    keep_frac: float,
    drop_frac: float,
    rng: np.random.Generator,
    inject_frac: float = 0.0,
) -> Graph:
    """
    Keep all schema + label edges.
    Subsample scale edges (affiliatedWith/authored/relatedTo) by keep_frac.
    Optionally drop extra drop_frac from kept scale edges.
    Optionally inject random schema-compatible edges (inject_frac).
    """
    label_preds = {pm["hasCoAuthor"], pm["hasLink"]}
    scale_preds = {pm["affiliatedWith"], pm["authored"], pm["relatedTo"]}

    out = Graph()
    kept_scaled: List[Tuple[URIRef, URIRef, URIRef]] = []

    for s, p, o in rdf_full:
        if p in (RDF.type, RDFS.label, RDFS.comment):
            out.add((s, p, o))
            continue

        # always keep label edges
        if p in label_preds:
            out.add((s, p, o))
            continue

        # scale edges: keep_frac subsampling
        if p in scale_preds:
            if rng.random() < keep_frac:
                kept_scaled.append((s, p, o))
            continue

        # keep other predicates unchanged
        out.add((s, p, o))

    # Apply triple-drop noise on the already-kept scaled edges
    if drop_frac > 0 and len(kept_scaled) > 0:
        m = len(kept_scaled)
        drop_n = int(round(drop_frac * m))
        drop_idx = set(rng.choice(m, size=drop_n, replace=False).tolist()) if drop_n > 0 else set()
        for i, t in enumerate(kept_scaled):
            if i not in drop_idx:
                out.add(t)
    else:
        for t in kept_scaled:
            out.add(t)

    if inject_frac and inject_frac > 0:
        inject_random_edges(out, pm, rng, inject_frac)

    return out


def _pick_rdf_term(lst: List, rng: np.random.Generator):
    """
    SAFE sampler for rdflib terms: never returns numpy-string.
    """
    if not lst:
        return None
    idx = int(rng.integers(0, len(lst)))
    t = lst[idx]
    # rdflib terms are fine; if it became a string, convert back
    if isinstance(t, str):
        return URIRef(t)
    return t


def inject_random_edges(rdf_g: Graph, pm: Dict[str, URIRef], rng: np.random.Generator, inject_frac: float):
    """
    Inject a small number of random schema-compatible edges (best-effort).
    Uses safe sampling so subjects/objects remain rdflib.URIRef.
    """
    authors, pubs, interests, affs = [], [], [], []

    # Collect typed individuals
    for s, _, o in rdf_g.triples((None, RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        tname = _local_name(o)
        if tname == "Author":
            authors.append(s)
        elif tname in ("Publication", "Paper", "Article"):  # tolerate variants
            pubs.append(s)
        elif tname in ("ResearchInterest", "Topic", "Keyword"):
            interests.append(s)
        elif tname == "Affiliation":
            affs.append(s)

    # Count existing "scale" edges
    scale_preds = [pm["affiliatedWith"], pm["authored"], pm["relatedTo"]]
    cur_scaled = sum(1 for p in scale_preds for _ in rdf_g.triples((None, p, None)))
    if cur_scaled == 0:
        return

    inject_n = max(1, int(round(inject_frac * cur_scaled)))

    for _ in range(inject_n):
        r = rng.random()

        # authored: Author -> Publication
        if r < 1 / 3 and authors and pubs:
            s = _pick_rdf_term(authors, rng)
            o = _pick_rdf_term(pubs, rng)
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                rdf_g.add((s, pm["authored"], o))

        # relatedTo: Publication -> ResearchInterest
        elif r < 2 / 3 and pubs and interests:
            s = _pick_rdf_term(pubs, rng)
            o = _pick_rdf_term(interests, rng)
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                rdf_g.add((s, pm["relatedTo"], o))

        # affiliatedWith: Author -> Affiliation
        else:
            if authors and affs:
                s = _pick_rdf_term(authors, rng)
                o = _pick_rdf_term(affs, rng)
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    rdf_g.add((s, pm["affiliatedWith"], o))


# ------------------- Eval Pair Sampling -------------------
def sample_eval(pos_edges: List[Pair], nodes: List[URIRef], n_pos: int, n_neg: int, rng: np.random.Generator):
    """
    Sample eval positives by indices (so edges remain tuples, not lists).
    """
    pos_edges = [canonical(u, v) for (u, v) in pos_edges]
    pos_edges = list(dict.fromkeys(pos_edges))  # deduplicate

    if len(pos_edges) < n_pos:
        raise ValueError(f"Not enough positive edges for evaluation: need {n_pos}, but only {len(pos_edges)}.")

    idx = rng.choice(len(pos_edges), size=n_pos, replace=False).tolist()
    eval_pos = [pos_edges[i] for i in idx]  # list[tuple]
    pos_set = set(eval_pos)

    eval_neg: List[Pair] = []
    neg_set = set()
    while len(eval_neg) < n_neg:
        u = nodes[int(rng.integers(0, len(nodes)))]
        v = nodes[int(rng.integers(0, len(nodes)))]
        if u == v:
            continue
        pair = canonical(u, v)
        if pair in pos_set or pair in neg_set:
            continue
        eval_neg.append(pair)
        neg_set.add(pair)

    return eval_pos, eval_neg


# ------------------- Model Training/Eval -------------------
def build_Xy(pairs_pos: List[Pair], pairs_neg: List[Pair], emb: Dict[URIRef, np.ndarray]):
    X, y = [], []
    for u, v in pairs_pos:
        X.append(ga.pair_features(u, v, emb))
        y.append(1)
    for u, v in pairs_neg:
        X.append(ga.pair_features(u, v, emb))
        y.append(0)
    return np.vstack(X), np.array(y, dtype=int)


def train_lr_local(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """
    Local LR training (no dependency on ga.train_lr_rf).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    uniq = np.unique(y_train)
    if len(uniq) < 2:
        raise ValueError(f"Training set has only one class {uniq}. Check label/neg sampling.")

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_setting(
    rdf_g: Graph,
    pm: Dict[str, URIRef],
    seed: int,
    eval_pos: List[Pair],
    eval_neg: List[Pair],
    dim=64,
    alpha=0.5
):
    """
    Train LR on training edges (from this KG slice) using surrogate graph embeddings.
    Evaluate on fixed eval_pos/eval_neg pairs and return AUC/F1 plus graph size.
    """
    preds_for_graph = {pm["hasCoAuthor"], pm["hasLink"], pm["affiliatedWith"], pm["authored"], pm["relatedTo"]}
    Gnx = ga.rdf_to_nx(rdf_g, preds_for_graph)

    # positives present in this sliced KG
    pos_edges = ga.extract_positive_links(rdf_g, {"hasCoAuthor": pm["hasCoAuthor"], "hasLink": pm["hasLink"]})
    pos_edges = [canonical(u, v) for (u, v) in pos_edges]
    pos_edges = list(dict.fromkeys(pos_edges))

    # remove eval positives from training positives
    eval_pos_set = set(eval_pos)
    train_pos = [e for e in pos_edges if e not in eval_pos_set]

    if len(train_pos) == 0:
        raise ValueError("No training positives left after slicing. Try higher kg_frac or lower drop_frac.")

    nodes = list(Gnx.nodes())
    train_neg = ga.sample_negative_links(nodes, set(train_pos), n_samples=len(train_pos), seed=seed)

    # embeddings
    emb0 = ga.build_surrogate_embeddings(Gnx, dim=dim, seed=seed)
    emb1 = ga.gnn_one_layer_update(Gnx, emb0, alpha=alpha)

    # train LR
    X_train, y_train = build_Xy(train_pos, train_neg, emb1)
    clf = train_lr_local(X_train, y_train, seed)

    # eval
    X_test, y_test = build_Xy(eval_pos, eval_neg, emb1)
    y_prob = clf.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score, f1_score
    auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, (y_prob >= 0.5).astype(int)))

    return auc, f1, Gnx.number_of_nodes(), Gnx.number_of_edges()


# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", required=True, help="Path to OWL file")
    ap.add_argument("--out_dir", default="results_scalability")
    ap.add_argument("--eval_pairs", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--kg_fracs", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--drop_fracs", nargs="+", type=float, default=[0.0, 0.05, 0.10])
    ap.add_argument("--inject_fracs", nargs="+", type=float, default=[0.0, 0.01])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rdf_full, _ = measure(ga.load_owl, args.owl)
    pm = build_pred_map(rdf_full)

    pos_edges = ga.extract_positive_links(rdf_full, {"hasCoAuthor": pm["hasCoAuthor"], "hasLink": pm["hasLink"]})
    pos_edges = [canonical(u, v) for (u, v) in pos_edges]
    pos_edges = list(dict.fromkeys(pos_edges))

    if len(pos_edges) < max(10, args.eval_pairs // 2):
        raise ValueError(f"Too few positive edges in OWL ({len(pos_edges)}). Check hasCoAuthor/hasLink assertions.")

    # nodes from endpoints of positive edges
    nodes = sorted({u for (u, v) in pos_edges for u in (u, v)}, key=lambda x: str(x))

    n_pos = args.eval_pairs // 2
    n_neg = args.eval_pairs - n_pos

    rows = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        eval_pos, eval_neg = sample_eval(pos_edges, nodes, n_pos, n_neg, rng)

        for kg_frac in args.kg_fracs:
            for drop_frac in args.drop_fracs:
                for inj_frac in args.inject_fracs:
                    rdf_var, tm_slice = measure(slice_graph, rdf_full, pm, kg_frac, drop_frac, rng, inj_frac)
                    (auc, f1, n_nodes, n_edges), tm_run = measure(
                        evaluate_setting, rdf_var, pm, seed, eval_pos, eval_neg
                    )

                    rows.append({
                        "seed": seed,
                        "kg_frac": kg_frac,
                        "drop_frac": drop_frac,
                        "inject_frac": inj_frac,
                        "triples": len(rdf_var),
                        "nodes": n_nodes,
                        "edges": n_edges,
                        "auc": auc,
                        "f1": f1,
                        "slice_time_s": tm_slice.seconds,
                        "run_time_s": tm_run.seconds,
                        "peak_rss_mb": max(tm_slice.peak_rss_mb, tm_run.peak_rss_mb),
                    })

                    print(f"[OK] seed={seed} kg={kg_frac} drop={drop_frac} inj={inj_frac} "
                          f"AUC={auc:.3f} F1={f1:.3f} time={tm_run.seconds:.2f}s "
                          f"peak={max(tm_slice.peak_rss_mb, tm_run.peak_rss_mb):.1f}MB")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "scalability_runs.csv"), index=False)

    summary = df.groupby(["kg_frac", "drop_frac", "inject_frac"], as_index=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        time_mean=("run_time_s", "mean"),
        time_std=("run_time_s", "std"),
        mem_mean=("peak_rss_mb", "mean"),
        mem_std=("peak_rss_mb", "std"),
    )
    summary.to_csv(os.path.join(args.out_dir, "scalability_summary.csv"), index=False)

    print("\nSaved:")
    print(f"  {os.path.join(args.out_dir, 'scalability_runs.csv')}")
    print(f"  {os.path.join(args.out_dir, 'scalability_summary.csv')}")


if __name__ == "__main__":
    main()
