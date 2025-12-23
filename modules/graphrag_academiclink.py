#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphrag_academiclink.py (UPDATED with export logs)

Exports per-dataset logs for Section 4.5:
  - proposed_triples.csv
  - validated_triples.csv
  - explanations.jsonl

Works with AcademicLink.owl and your predicate local names:
  affiliatedWith, authored, hasAuthor1, hasAuthor2, hasCoAuthor, hasLink, relatedTo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from rdflib.namespace import XSD

# ---------- Configuration defaults ----------
DEFAULT_K = 30
DEFAULT_TAU = 0.70
DEFAULT_SIM_TAU = 0.65
DEFAULT_MAX_HOPS = 4
DEFAULT_NEG_RATIO = 2

SURROGATE_DIM = 64
RE_AUTHOR_ID = re.compile(r"^\d{6,}$")


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def uri_local(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def find_predicates_by_localname(g: Graph, local_names: List[str]) -> Dict[str, URIRef]:
    found: Dict[str, URIRef] = {}
    all_preds: Set[URIRef] = set(p for p in g.predicates() if isinstance(p, URIRef))

    def local_of(uri: URIRef) -> str:
        s = str(uri)
        if "#" in s:
            return s.rsplit("#", 1)[-1]
        return s.rsplit("/", 1)[-1]

    local_to_uri: Dict[str, URIRef] = {}
    for p in all_preds:
        ln = local_of(p)
        local_to_uri.setdefault(ln, p)

    missing = []
    for name in local_names:
        if name in local_to_uri:
            found[name] = local_to_uri[name]
        else:
            cand = None
            for ln, uri in local_to_uri.items():
                if ln.lower() == name.lower():
                    cand = uri
                    break
            if cand is not None:
                found[name] = cand
            else:
                missing.append(name)

    if missing:
        raise ValueError(
            f"Could not find these predicate local-names in OWL: {missing}\n"
            f"Tip: open the OWL in Protégé and confirm exact spellings."
        )
    return found


def guess_author_uri(g: Graph, author_id: str) -> Optional[URIRef]:
    if not RE_AUTHOR_ID.match(author_id):
        raise ValueError(f"author_id must be digits (>=6). Got: {author_id}")

    # 1) localname exact match
    for s in g.subjects():
        if isinstance(s, URIRef) and uri_local(s) == author_id:
            return s

    # 2) rdfs:label exact match
    for s, _, lbl in g.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef) and isinstance(lbl, Literal):
            if str(lbl).strip() == author_id:
                return s

    # 3) token containment
    needle = f"/{author_id}"
    needle2 = f"#{author_id}"
    for s in g.subjects():
        if isinstance(s, URIRef):
            ss = str(s)
            if needle in ss or needle2 in ss or ss.endswith(author_id):
                return s

    return None


# ---------- Evidence retrieval ----------
def collect_text_annotations(g: Graph, node: URIRef, max_items: int = 20) -> List[str]:
    texts: List[str] = []

    for _, _, o in g.triples((node, RDFS.label, None)):
        if isinstance(o, Literal):
            texts.append(str(o))
    for _, _, o in g.triples((node, RDFS.comment, None)):
        if isinstance(o, Literal):
            texts.append(str(o))

    for p in set(g.predicates(subject=node)):
        if not isinstance(p, URIRef):
            continue
        ln = uri_local(p).lower()
        if any(k in ln for k in ["abstract", "title", "keyword", "name", "description", "summary"]):
            for _, _, o in g.triples((node, p, None)):
                if isinstance(o, Literal):
                    texts.append(str(o))

    seen = set()
    out = []
    for t in texts:
        t2 = re.sub(r"\s+", " ", t).strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            out.append(t2)
        if len(out) >= max_items:
            break
    return out


def retrieve_evidence_passages(
    g: Graph,
    author_u: URIRef,
    author_v: URIRef,
    preds: Dict[str, URIRef],
    k: int = DEFAULT_K,
) -> List[str]:
    authored = preds["authored"]
    relatedTo = preds["relatedTo"]
    affiliatedWith = preds["affiliatedWith"]

    passages: List[str] = []

    def add_texts(node: URIRef):
        for t in collect_text_annotations(g, node):
            passages.append(t)

    add_texts(author_u)
    add_texts(author_v)

    papers_u = list(set(o for _, _, o in g.triples((author_u, authored, None)) if isinstance(o, URIRef)))
    papers_v = list(set(o for _, _, o in g.triples((author_v, authored, None)) if isinstance(o, URIRef)))

    aff_u = list(set(o for _, _, o in g.triples((author_u, affiliatedWith, None)) if isinstance(o, URIRef)))
    aff_v = list(set(o for _, _, o in g.triples((author_v, affiliatedWith, None)) if isinstance(o, URIRef)))

    for n in papers_u[: max(1, k // 6)] + papers_v[: max(1, k // 6)]:
        add_texts(n)

    for n in aff_u + aff_v:
        add_texts(n)

    inter_u: Set[URIRef] = set()
    inter_v: Set[URIRef] = set()
    for p in papers_u:
        for _, _, o in g.triples((p, relatedTo, None)):
            if isinstance(o, URIRef):
                inter_u.add(o)
    for p in papers_v:
        for _, _, o in g.triples((p, relatedTo, None)):
            if isinstance(o, URIRef):
                inter_v.add(o)

    for n in list(inter_u)[: max(1, k // 8)] + list(inter_v)[: max(1, k // 8)]:
        add_texts(n)

    return passages[:k]


# ---------- Triples ----------
@dataclass(frozen=True)
class ProposedTriple:
    s: URIRef
    p: URIRef
    o: URIRef
    score: float
    reason: str
    evidence: str  # store a short evidence string for auditing


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def rdf_to_nx(g: Graph, allowed_predicates: Set[URIRef]) -> nx.Graph:
    Gnx = nx.Graph()
    for s, p, o in g:
        if not (isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef)):
            continue
        if p not in allowed_predicates:
            continue
        Gnx.add_node(s)
        Gnx.add_node(o)
        Gnx.add_edge(s, o, predicate=p)
    return Gnx


def build_surrogate_embeddings(Gnx: nx.Graph, dim: int = SURROGATE_DIM, seed: int = 42) -> Dict[URIRef, np.ndarray]:
    _seed_everything(seed)
    nodes = list(Gnx.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    R = np.random.normal(0, 1, size=(n, dim)).astype(np.float32)

    emb: Dict[URIRef, np.ndarray] = {}
    for node in nodes:
        neigh = list(Gnx.neighbors(node))
        vec = R[idx[node]].copy()
        for m in neigh[:200]:
            vec += 0.5 * R[idx[m]]
        emb[node] = vec / (np.linalg.norm(vec) + 1e-12)
    return emb


def heuristic_extract_triples(
    g: Graph,
    author_u: URIRef,
    author_v: URIRef,
    preds: Dict[str, URIRef],
    emb: Dict[URIRef, np.ndarray],
    passages: List[str],
    sim_tau: float = DEFAULT_SIM_TAU,
) -> List[ProposedTriple]:
    authored = preds["authored"]
    relatedTo = preds["relatedTo"]
    affiliatedWith = preds["affiliatedWith"]
    hasCoAuthor = preds["hasCoAuthor"]
    hasLink = preds["hasLink"]

    papers_u = set(o for _, _, o in g.triples((author_u, authored, None)) if isinstance(o, URIRef))
    papers_v = set(o for _, _, o in g.triples((author_v, authored, None)) if isinstance(o, URIRef))

    aff_u = set(o for _, _, o in g.triples((author_u, affiliatedWith, None)) if isinstance(o, URIRef))
    aff_v = set(o for _, _, o in g.triples((author_v, affiliatedWith, None)) if isinstance(o, URIRef))
    shared_aff = aff_u.intersection(aff_v)

    inter_u: Set[URIRef] = set()
    inter_v: Set[URIRef] = set()
    for p in papers_u:
        for _, _, o in g.triples((p, relatedTo, None)):
            if isinstance(o, URIRef):
                inter_u.add(o)
    for p in papers_v:
        for _, _, o in g.triples((p, relatedTo, None)):
            if isinstance(o, URIRef):
                inter_v.add(o)
    shared_inter = inter_u.intersection(inter_v)

    sim_uv = cosine(emb[author_u], emb[author_v]) if (author_u in emb and author_v in emb) else 0.0

    evidence = " || ".join([re.sub(r"\s+", " ", t).strip() for t in passages[:3]])
    evidence = evidence[:500]  # keep short for CSV

    props: List[ProposedTriple] = []

    if shared_aff:
        score = 0.75 + 0.05 * min(1.0, sim_uv)
        props.append(ProposedTriple(author_u, relatedTo, author_v, score, "shared_affiliation", evidence))

    if shared_inter:
        score = 0.78 + 0.07 * min(1.0, sim_uv)
        props.append(ProposedTriple(author_u, relatedTo, author_v, score, "shared_research_interest", evidence))

    if sim_uv >= sim_tau:
        props.append(ProposedTriple(author_u, hasLink, author_v, sim_uv, "high_embedding_similarity", evidence))
        props.append(ProposedTriple(author_v, hasLink, author_u, sim_uv, "high_embedding_similarity", evidence))

    joined = " ".join(passages).lower()
    cue = any(w in joined for w in ["coauthor", "co-author", "collaborat", "joint work", "together"])
    if cue and sim_uv >= (sim_tau - 0.05):
        props.append(ProposedTriple(author_u, hasCoAuthor, author_v, 0.80 + 0.10 * sim_uv, "textual_collaboration_cue", evidence))
        props.append(ProposedTriple(author_v, hasCoAuthor, author_u, 0.80 + 0.10 * sim_uv, "textual_collaboration_cue", evidence))

    # de-dup by (s,p,o) keep max score
    best: Dict[Tuple[URIRef, URIRef, URIRef], ProposedTriple] = {}
    for t in props:
        key = (t.s, t.p, t.o)
        if key not in best or t.score > best[key].score:
            best[key] = t
    return list(best.values())


def validate_triple_with_reason(g: Graph, triple: ProposedTriple, tau: float) -> Tuple[bool, str]:
    if triple.score < tau:
        return False, "below_tau"
    if triple.s is None or triple.p is None or triple.o is None:
        return False, "null_component"
    if (triple.s, RDF.type, OWL.Class) in g:
        return False, "subject_is_owl_class"
    if (triple.o, RDF.type, OWL.Class) in g:
        return False, "object_is_owl_class"
    return True, "accepted"


def gnn_one_layer_update(Gnx: nx.Graph, emb: Dict[URIRef, np.ndarray], alpha: float = 0.6) -> Dict[URIRef, np.ndarray]:
    new_emb: Dict[URIRef, np.ndarray] = {}
    for v in Gnx.nodes():
        hv = emb.get(v)
        if hv is None:
            hv = np.zeros((SURROGATE_DIM,), dtype=np.float32)
        neigh = list(Gnx.neighbors(v))
        if neigh:
            m = np.mean(np.stack([emb.get(u, hv) for u in neigh], axis=0), axis=0)
        else:
            m = hv
        h2 = alpha * hv + (1 - alpha) * m
        h2 = h2 / (np.linalg.norm(h2) + 1e-12)
        new_emb[v] = h2.astype(np.float32)
    return new_emb


def explain_path(Gnx: nx.Graph, u: URIRef, v: URIRef, max_hops: int = DEFAULT_MAX_HOPS) -> Optional[List[Tuple[URIRef, URIRef, URIRef]]]:
    if u not in Gnx or v not in Gnx:
        return None
    try:
        path_nodes = nx.shortest_path(Gnx, u, v)
    except Exception:
        return None

    if len(path_nodes) - 1 > max_hops:
        return None

    triples = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        pred = Gnx.edges[a, b].get("predicate", URIRef("urn:edge"))
        triples.append((a, pred, b))
    return triples


def format_path(triples: List[Tuple[URIRef, URIRef, URIRef]]) -> str:
    chunks = []
    for s, p, o in triples:
        chunks.append(f"{uri_local(s)} --{uri_local(p)}--> {uri_local(o)}")
    return "  |  ".join(chunks)


# ---------- Pos/neg edges ----------
def extract_positive_links(g: Graph, preds: Dict[str, URIRef]) -> Set[Tuple[URIRef, URIRef]]:
    pos: Set[Tuple[URIRef, URIRef]] = set()
    for p_name in ["hasCoAuthor", "hasLink"]:
        p = preds[p_name]
        for s, _, o in g.triples((None, p, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                a, b = (s, o) if str(s) < str(o) else (o, s)
                pos.add((a, b))
    return pos


def sample_negative_links(nodes: List[URIRef], pos: Set[Tuple[URIRef, URIRef]], n_samples: int, seed: int = 42) -> List[Tuple[URIRef, URIRef]]:
    _seed_everything(seed)
    neg = []
    seen = set(pos)
    tries = 0
    while len(neg) < n_samples and tries < n_samples * 80:
        a = random.choice(nodes)
        b = random.choice(nodes)
        if a == b:
            tries += 1
            continue
        u, v = (a, b) if str(a) < str(b) else (b, a)
        if (u, v) in seen:
            tries += 1
            continue
        seen.add((u, v))
        neg.append((u, v))
    return neg


def pair_features(u: URIRef, v: URIRef, emb: Dict[URIRef, np.ndarray]) -> np.ndarray:
    eu = emb.get(u, np.zeros((SURROGATE_DIM,), dtype=np.float32))
    ev = emb.get(v, np.zeros((SURROGATE_DIM,), dtype=np.float32))
    return np.concatenate([eu, ev, np.abs(eu - ev), eu * ev], axis=0)


def train_lr(X: np.ndarray, y: np.ndarray, seed: int = 42):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return clf


@dataclass
class GraphRAGResult:
    prob: float
    decision: int
    explanation: Optional[str]
    path_len: int
    proposed: List[ProposedTriple]
    validated_rows: List[Dict]
    accepted: List[ProposedTriple]


@dataclass
class GraphRAGContext:
    g: Graph
    preds: Dict[str, URIRef]
    allowed: Set[URIRef]
    Gnx: nx.Graph
    emb0: Dict[URIRef, np.ndarray]
    nodes: List[URIRef]
    pos_edges: Set[Tuple[URIRef, URIRef]]


def build_context(g: Graph, preds: Dict[str, URIRef], seed: int = 42) -> GraphRAGContext:
    allowed = {
        preds["affiliatedWith"],
        preds["authored"],
        preds["hasAuthor1"],
        preds["hasAuthor2"],
        preds["hasCoAuthor"],
        preds["hasLink"],
        preds["relatedTo"],
    }
    Gnx = rdf_to_nx(g, allowed)
    emb0 = build_surrogate_embeddings(Gnx, dim=SURROGATE_DIM, seed=seed)
    nodes = list(Gnx.nodes())
    pos_edges = extract_positive_links(g, preds)
    return GraphRAGContext(g=g, preds=preds, allowed=allowed, Gnx=Gnx, emb0=emb0, nodes=nodes, pos_edges=pos_edges)


def graphrag_predict_link(ctx: GraphRAGContext, author_u: URIRef, author_v: URIRef, k: int, tau: float, sim_tau: float, max_hops: int, neg_ratio: int, seed: int) -> GraphRAGResult:
    _seed_everything(seed)

    passages = retrieve_evidence_passages(ctx.g, author_u, author_v, ctx.preds, k=k)

    proposed = heuristic_extract_triples(ctx.g, author_u, author_v, ctx.preds, ctx.emb0, passages, sim_tau=sim_tau)

    validated_rows = []
    accepted: List[ProposedTriple] = []
    for t in proposed:
        ok, reason = validate_triple_with_reason(ctx.g, t, tau=tau)
        validated_rows.append({
            "s": str(t.s), "p": str(t.p), "o": str(t.o),
            "s_local": uri_local(t.s), "p_local": uri_local(t.p), "o_local": uri_local(t.o),
            "score": float(t.score), "proposal_reason": t.reason,
            "evidence": t.evidence,
            "accepted": int(ok),
            "reject_reason": "" if ok else reason
        })
        if ok:
            accepted.append(t)

    # Build KG+ just for this instance (do NOT mutate RDF graph)
    Gnx_plus = ctx.Gnx.copy()
    for t in accepted:
        Gnx_plus.add_node(t.s)
        Gnx_plus.add_node(t.o)
        Gnx_plus.add_edge(t.s, t.o, predicate=t.p)

    # One-layer update on KG+
    emb1 = gnn_one_layer_update(Gnx_plus, ctx.emb0, alpha=0.6)

    # Base ML predictor (LR) trained from existing positives + sampled negatives
    if len(ctx.pos_edges) < 10:
        sim = cosine(emb1.get(author_u, np.zeros((SURROGATE_DIM,), dtype=np.float32)),
                     emb1.get(author_v, np.zeros((SURROGATE_DIM,), dtype=np.float32)))
        prob = max(0.05, min(0.95, 0.5 * (sim + 1.0)))
    else:
        neg = sample_negative_links(ctx.nodes, ctx.pos_edges, n_samples=len(ctx.pos_edges) * neg_ratio, seed=seed)
        pairs = list(ctx.pos_edges) + neg
        y = np.array([1] * len(ctx.pos_edges) + [0] * len(neg), dtype=np.int64)
        X = np.stack([pair_features(u, v, emb1) for (u, v) in pairs], axis=0)

        clf = train_lr(X, y, seed=seed)
        x_uv = pair_features(author_u, author_v, emb1).reshape(1, -1)
        prob = float(clf.predict_proba(x_uv)[0, 1])

    decision = int(prob >= 0.5)

    path_triples = explain_path(Gnx_plus, author_u, author_v, max_hops=max_hops)
    explanation = format_path(path_triples) if path_triples else None
    path_len = (len(path_triples) if path_triples else 0)

    return GraphRAGResult(
        prob=prob,
        decision=decision,
        explanation=explanation,
        path_len=path_len,
        proposed=proposed,
        validated_rows=validated_rows,
        accepted=accepted
    )


# ---------- Export helpers ----------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_proposed_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        # still create header-only file
        rows = [{"dataset": "", "instance_id": "", "s_local": "", "p_local": "", "o_local": "", "score": "", "proposal_reason": "", "evidence": ""}]
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_validated_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        rows = [{"dataset": "", "instance_id": "", "s_local": "", "p_local": "", "o_local": "", "score": "", "proposal_reason": "", "accepted": "", "reject_reason": "", "evidence": ""}]
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_explanations_jsonl(rows: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_owl(path: str) -> Graph:
    g = Graph()
    g.parse(path)
    return g


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="GraphRAG AcademicLink with export logs for Section 4.5")
    parser.add_argument("--owl", type=str, required=True, help="Path to AcademicLink OWL file")
    parser.add_argument("--dataset_name", type=str, default="Academic", help="Name written into CSV logs (e.g., Academic)")
    parser.add_argument("--author1", type=str, default=None, help="Author ID (digits)")
    parser.add_argument("--author2", type=str, default=None, help="Author ID (digits)")

    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--sim_tau", type=float, default=DEFAULT_SIM_TAU)
    parser.add_argument("--max_hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--export_dir", type=str, default=None, help="Folder to export proposed_triples.csv, validated_triples.csv, explanations.jsonl")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional tag appended to filenames")

    # Batch mode for Table Z statistics
    parser.add_argument("--batch_eval", action="store_true", help="Run many instances and export logs (recommended)")
    parser.add_argument("--n_eval", type=int, default=200, help="Number of eval instances in batch mode")
    parser.add_argument("--pos_frac", type=float, default=0.5, help="Fraction of positives in batch mode")
    args = parser.parse_args()

    g = load_owl(args.owl)

    pred_names = ["affiliatedWith", "authored", "hasAuthor1", "hasAuthor2", "hasCoAuthor", "hasLink", "relatedTo"]
    preds = find_predicates_by_localname(g, pred_names)

    ctx = build_context(g, preds, seed=args.seed)

    # File naming
    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.dataset_name}_{tag}"
    if args.export_dir:
        _ensure_dir(args.export_dir)

    proposed_rows_all: List[Dict] = []
    validated_rows_all: List[Dict] = []
    expl_rows_all: List[Dict] = []

    def run_one(instance_id: str, u: URIRef, v: URIRef, y_true: Optional[int] = None):
        res = graphrag_predict_link(
            ctx=ctx, author_u=u, author_v=v,
            k=args.k, tau=args.tau, sim_tau=args.sim_tau,
            max_hops=args.max_hops, neg_ratio=DEFAULT_NEG_RATIO, seed=args.seed
        )

        # proposed_triples.csv rows (only proposed)
        for t in res.proposed:
            proposed_rows_all.append({
                "dataset": args.dataset_name,
                "instance_id": instance_id,
                "s": str(t.s), "p": str(t.p), "o": str(t.o),
                "s_local": uri_local(t.s), "p_local": uri_local(t.p), "o_local": uri_local(t.o),
                "score": float(t.score),
                "proposal_reason": t.reason,
                "evidence": t.evidence
            })

        # validated_triples.csv rows (proposed + accepted/reject reason)
        for r in res.validated_rows:
            r2 = dict(r)
            r2["dataset"] = args.dataset_name
            r2["instance_id"] = instance_id
            validated_rows_all.append(r2)

        expl_rows_all.append({
            "dataset": args.dataset_name,
            "instance_id": instance_id,
            "y_true": y_true,
            "prob": float(res.prob),
            "decision": int(res.decision),
            "explanation": res.explanation,
            "path_len": int(res.path_len),
            "n_proposed": int(len(res.proposed)),
            "n_accepted": int(len(res.accepted)),
            "accepted_triples": [
                {"s_local": uri_local(t.s), "p_local": uri_local(t.p), "o_local": uri_local(t.o), "score": float(t.score), "reason": t.reason}
                for t in res.accepted
            ]
        })

        return res

    if args.batch_eval:
        # Build balanced pairs for eval
        pos_list = list(ctx.pos_edges)
        if not pos_list:
            raise ValueError("No positive edges found (hasCoAuthor/hasLink). Cannot run batch_eval.")

        n_pos = int(args.n_eval * args.pos_frac)
        n_neg = args.n_eval - n_pos

        _seed_everything(args.seed)
        chosen_pos = random.sample(pos_list, k=min(n_pos, len(pos_list)))
        chosen_neg = sample_negative_links(ctx.nodes, ctx.pos_edges, n_samples=n_neg, seed=args.seed)

        # run positives
        for i, (u, v) in enumerate(chosen_pos):
            run_one(instance_id=f"pos_{i}", u=u, v=v, y_true=1)

        # run negatives
        for i, (u, v) in enumerate(chosen_neg):
            run_one(instance_id=f"neg_{i}", u=u, v=v, y_true=0)

        print(f"[OK] Batch completed: {len(chosen_pos)} positives + {len(chosen_neg)} negatives = {len(chosen_pos)+len(chosen_neg)} instances")

    else:
        if not (args.author1 and args.author2):
            raise ValueError("Provide --author1 and --author2 unless using --batch_eval.")

        u = guess_author_uri(g, args.author1)
        v = guess_author_uri(g, args.author2)
        if u is None or v is None:
            raise ValueError(
                f"Could not find one/both author URIs in OWL for IDs: {args.author1}, {args.author2}"
            )
        res = run_one(instance_id=f"{args.author1}_{args.author2}", u=u, v=v, y_true=None)

        print("=== GraphRAG Academic Link Prediction ===")
        print(f"Author1: {args.author1} ({u})")
        print(f"Author2: {args.author2} ({v})")
        print(f"Predicted collaboration probability: {res.prob:.4f}")
        print(f"Decision (1=collaborate, 0=no): {res.decision}")
        print("--- Explanation path (KG+) ---")
        print(res.explanation or "No short explanation path found. Try --max_hops 6.")

    # export files
    if args.export_dir:
        proposed_path = os.path.join(args.export_dir, f"{base}_proposed_triples.csv")
        validated_path = os.path.join(args.export_dir, f"{base}_validated_triples.csv")
        expl_path = os.path.join(args.export_dir, f"{base}_explanations.jsonl")

        export_proposed_csv(proposed_rows_all, proposed_path)
        export_validated_csv(validated_rows_all, validated_path)
        export_explanations_jsonl(expl_rows_all, expl_path)

        print(f"[SAVED] {proposed_path}")
        print(f"[SAVED] {validated_path}")
        print(f"[SAVED] {expl_path}")


if __name__ == "__main__":
    main()
