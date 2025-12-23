#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphrag_academiclink.py

GraphRAG for Academic Link Prediction with Ontology-Filtered Augmentation (Algorithm 1)

Goal
----
Given two author IDs (digit strings), predict whether they will collaborate (co-author link)
and return an explanation path in the augmented knowledge graph.

This script is designed to match your ontology's *object property* names exactly:
    - affiliatedWith
    - authored
    - hasAuthor1
    - hasAuthor2
    - hasCoAuthor
    - hasLink
    - relatedTo

Input
-----
- An OWL file (RDF/XML, OWL/XML, Turtle, etc.) e.g.: AcademicLink_GNN_dense_colored.owl
- Two author IDs (digits), e.g.: 57202379117 57205675686

Output
------
- Predicted probability/confidence of collaboration link
- Explanation path (a short KG path connecting the two authors)
- Optionally: an augmented OWL file with accepted new triples and confidence annotations

Dependencies (recommended)
-------------------------
pip install rdflib networkx numpy scikit-learn

Optional (for Louvain community detection):
pip install python-louvain

Optional (for nicer progress logging):
pip install tqdm

Notes on "GraphRAG" in offline mode
----------------------------------
If you don't call an external LLM, this script still performs a GraphRAG-like loop:
- Retrieve evidence from graph-attached texts (labels/comments/abstract-like annotations)
- Propose candidate triples using heuristic "generative" patterns + similarity signals
- Validate candidates against ontology-typed constraints + thresholds
- Augment KG and produce explanation paths

If you later want to plug in an LLM, see the `llm_extract_triples()` stub.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from rdflib.namespace import XSD

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- Configuration defaults ----------
DEFAULT_K = 30                # evidence retrieval budget
DEFAULT_TAU = 0.70            # validation threshold for accepting a proposed triple
DEFAULT_SIM_TAU = 0.65        # semantic similarity threshold (pre-filter)
DEFAULT_MAX_HOPS = 4          # explanation path max length
DEFAULT_NEG_RATIO = 2         # negative sampling ratio for link prediction

# A lightweight embedding dimension for "surrogate" embeddings (if ontology has no embeddings)
SURROGATE_DIM = 64

# Regex for digit author IDs (your authors are digit sequences)
RE_AUTHOR_ID = re.compile(r"^\d{6,}$")


# ---------- Utility: stable random ----------
def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------- Ontology predicate discovery ----------
def find_predicates_by_localname(g: Graph, local_names: List[str]) -> Dict[str, URIRef]:
    """
    Find predicate URIs in the graph that match local names (suffix after # or /).
    Returns dict: local_name -> URIRef
    Raises if any local name is not found.
    """
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
        # keep first if duplicates
        local_to_uri.setdefault(ln, p)

    missing = []
    for name in local_names:
        if name in local_to_uri:
            found[name] = local_to_uri[name]
        else:
            # try case-insensitive match
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


def uri_local(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


# ---------- Node identification ----------
def guess_author_uri(g: Graph, author_id: str) -> Optional[URIRef]:
    """
    Try to locate an Author individual URI using:
    1) exact localname match (…#5720…)
    2) rdfs:label exact match
    3) any URI containing the digits as a token
    """
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
    """
    Collect text attached to a node via common predicates:
    rdfs:label, rdfs:comment, and any annotation property ending with
    'abstract', 'title', 'keywords', 'name', 'description' (heuristic).
    """
    texts: List[str] = []

    # standard
    for _, _, o in g.triples((node, RDFS.label, None)):
        if isinstance(o, Literal):
            texts.append(str(o))
    for _, _, o in g.triples((node, RDFS.comment, None)):
        if isinstance(o, Literal):
            texts.append(str(o))

    # heuristic annotations
    for p in set(g.predicates(subject=node)):
        if not isinstance(p, URIRef):
            continue
        ln = uri_local(p).lower()
        if any(k in ln for k in ["abstract", "title", "keyword", "name", "description", "summary"]):
            for _, _, o in g.triples((node, p, None)):
                if isinstance(o, Literal):
                    texts.append(str(o))

    # de-dup + clip
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
    """
    Retrieve evidence passages for a candidate pair (u, v) by walking:
    Author -> authored -> Paper,
    Paper -> relatedTo -> ResearchInterest,
    Author -> affiliatedWith -> Affiliation
    and collecting available annotations on these nodes.

    Returns a list of short "passages" (strings).
    """
    authored = preds["authored"]
    relatedTo = preds["relatedTo"]
    affiliatedWith = preds["affiliatedWith"]

    passages: List[str] = []

    def add_texts(node: URIRef):
        for t in collect_text_annotations(g, node):
            passages.append(t)

    # Include author node texts
    add_texts(author_u)
    add_texts(author_v)

    # Collect papers (authored)
    papers_u = list(set(o for _, _, o in g.triples((author_u, authored, None)) if isinstance(o, URIRef)))
    papers_v = list(set(o for _, _, o in g.triples((author_v, authored, None)) if isinstance(o, URIRef)))

    # Collect affiliations
    aff_u = list(set(o for _, _, o in g.triples((author_u, affiliatedWith, None)) if isinstance(o, URIRef)))
    aff_v = list(set(o for _, _, o in g.triples((author_v, affiliatedWith, None)) if isinstance(o, URIRef)))

    for n in papers_u[: max(1, k // 6)] + papers_v[: max(1, k // 6)]:
        add_texts(n)

    for n in aff_u + aff_v:
        add_texts(n)

    # Interests via Paper relatedTo Interest
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

    # Clip to k
    return passages[:k]


# ---------- "Generative" triple proposal ----------
@dataclass(frozen=True)
class ProposedTriple:
    s: URIRef
    p: URIRef
    o: URIRef
    score: float
    reason: str


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def build_surrogate_embeddings(
    Gnx: nx.Graph,
    dim: int = SURROGATE_DIM,
    seed: int = 42,
) -> Dict[URIRef, np.ndarray]:
    """
    If the ontology file does NOT contain embeddings, we create a dense surrogate
    embedding per node from random projections of adjacency features.

    This is *not* OWL2Vec; it is a lightweight stand-in to support:
    - semantic similarity pre-filter
    - GNN-style update demonstration
    """
    _seed_everything(seed)
    nodes = list(Gnx.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Random projection matrix
    R = np.random.normal(0, 1, size=(n, dim)).astype(np.float32)

    # adjacency-based "bag-of-neighbors" in projected space
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
    """
    Offline "LLMExtract" surrogate:
    Propose plausible new triples based on overlap signals:
      - shared affiliations -> relatedTo
      - shared interests via their papers -> relatedTo
      - high embedding similarity -> hasLink or relatedTo
      - inferred co-author edge (hasCoAuthor) if strong evidence found

    Returns list of proposed triples with scores and reasons.
    """
    authored = preds["authored"]
    relatedTo = preds["relatedTo"]
    affiliatedWith = preds["affiliatedWith"]
    hasCoAuthor = preds["hasCoAuthor"]
    hasLink = preds["hasLink"]

    # gather neighborhoods
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

    # embedding similarity
    sim_uv = cosine(emb[author_u], emb[author_v]) if (author_u in emb and author_v in emb) else 0.0

    props: List[ProposedTriple] = []

    # propose "relatedTo" via shared affiliation
    for aff in list(shared_aff)[:5]:
        score = 0.75 + 0.05 * min(1.0, sim_uv)
        props.append(ProposedTriple(author_u, relatedTo, author_v, score, "shared_affiliation"))
        break

    # propose "relatedTo" via shared interest
    for it in list(shared_inter)[:5]:
        score = 0.78 + 0.07 * min(1.0, sim_uv)
        props.append(ProposedTriple(author_u, relatedTo, author_v, score, "shared_research_interest"))
        break

    # propose "hasLink" if similarity high
    if sim_uv >= sim_tau:
        props.append(ProposedTriple(author_u, hasLink, author_v, sim_uv, "high_embedding_similarity"))
        props.append(ProposedTriple(author_v, hasLink, author_u, sim_uv, "high_embedding_similarity"))

    # propose inferred co-author if strong textual cue
    joined = " ".join(passages).lower()
    cue = any(w in joined for w in ["coauthor", "co-author", "collaborat", "joint work", "together"])
    if cue and sim_uv >= (sim_tau - 0.05):
        props.append(ProposedTriple(author_u, hasCoAuthor, author_v, 0.80 + 0.10 * sim_uv, "textual_collaboration_cue"))
        props.append(ProposedTriple(author_v, hasCoAuthor, author_u, 0.80 + 0.10 * sim_uv, "textual_collaboration_cue"))

    # de-dup by (s,p,o) keep max score
    best: Dict[Tuple[URIRef, URIRef, URIRef], ProposedTriple] = {}
    for t in props:
        key = (t.s, t.p, t.o)
        if key not in best or t.score > best[key].score:
            best[key] = t
    return list(best.values())


# ---------- Validation: ontology-filtered augmentation ----------
def get_rdf_types(g: Graph, node: URIRef) -> Set[URIRef]:
    return set(o for _, _, o in g.triples((node, RDF.type, None)) if isinstance(o, URIRef))


def validate_triple(
    g: Graph,
    triple: ProposedTriple,
    tau: float = DEFAULT_TAU,
) -> bool:
    """
    Validate proposed triple by:
    - score threshold
    - node existence
    - lightweight type sanity: avoid OWL classes as instances
    """
    if triple.score < tau:
        return False
    if triple.s is None or triple.p is None or triple.o is None:
        return False

    # avoid asserting between OWL classes
    if (triple.s, RDF.type, OWL.Class) in g or (triple.o, RDF.type, OWL.Class) in g:
        return False

    return True


def augment_graph(
    g: Graph,
    accepted: List[ProposedTriple],
    conf_pred: Optional[URIRef] = None,
) -> None:
    """
    Add accepted triples to the RDF graph.
    If conf_pred is provided, add a confidence literal as annotation:
        (s, conf_pred, "0.85"^^xsd:float) for each accepted triple.
    """
    for t in accepted:
        g.add((t.s, t.p, t.o))
        if conf_pred is not None:
            g.add((t.s, conf_pred, Literal(float(t.score), datatype=XSD.float)))


# ---------- Build a working NetworkX graph ----------
def rdf_to_nx(
    g: Graph,
    allowed_predicates: Set[URIRef],
) -> nx.Graph:
    """
    Convert RDF triples (s,p,o) to an undirected NetworkX graph for connectivity.
    Only includes edges with predicates in allowed_predicates.
    """
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


# ---------- Community detection (for completeness, optional in prediction loop) ----------
def detect_communities(Gnx: nx.Graph) -> List[Set[URIRef]]:
    """
    Try Louvain communities; fallback to greedy modularity communities.
    """
    try:
        import community as community_louvain  # python-louvain
        part = community_louvain.best_partition(Gnx)
        comm_map: Dict[int, Set[URIRef]] = {}
        for node, cid in part.items():
            comm_map.setdefault(cid, set()).add(node)
        return list(comm_map.values())
    except Exception:
        # fallback
        try:
            comms = nx.algorithms.community.greedy_modularity_communities(Gnx)
            return [set(c) for c in comms]
        except Exception:
            return [set(Gnx.nodes())]


def community_embedding(
    nodes: Set[URIRef],
    emb: Dict[URIRef, np.ndarray],
) -> np.ndarray:
    vecs = [emb[n] for n in nodes if n in emb]
    if not vecs:
        return np.zeros((SURROGATE_DIM,), dtype=np.float32)
    v = np.mean(np.stack(vecs, axis=0), axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


# ---------- Link prediction dataset construction ----------
def extract_positive_links(
    g: Graph,
    preds: Dict[str, URIRef],
) -> Set[Tuple[URIRef, URIRef]]:
    """
    Positive collaboration edges:
    - hasCoAuthor
    - OR hasLink (if you use it for co-authorship)
    """
    pos: Set[Tuple[URIRef, URIRef]] = set()
    for p_name in ["hasCoAuthor", "hasLink"]:
        p = preds[p_name]
        for s, _, o in g.triples((None, p, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                # store undirected canonical
                a, b = (s, o) if str(s) < str(o) else (o, s)
                pos.add((a, b))
    return pos


def sample_negative_links(
    nodes: List[URIRef],
    pos: Set[Tuple[URIRef, URIRef]],
    n_samples: int,
    seed: int = 42,
) -> List[Tuple[URIRef, URIRef]]:
    _seed_everything(seed)
    neg = []
    seen = set(pos)
    tries = 0
    while len(neg) < n_samples and tries < n_samples * 50:
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


def pair_features(
    u: URIRef,
    v: URIRef,
    emb: Dict[URIRef, np.ndarray],
) -> np.ndarray:
    """
    Simple pair feature: [u, v, |u-v|, u*v]
    """
    eu = emb.get(u)
    ev = emb.get(v)
    if eu is None or ev is None:
        # zero fallback
        eu = np.zeros((SURROGATE_DIM,), dtype=np.float32)
        ev = np.zeros((SURROGATE_DIM,), dtype=np.float32)
    return np.concatenate([eu, ev, np.abs(eu - ev), eu * ev], axis=0)


# ---------- Models: LR / RF / lightweight GNN-like message passing ----------
def train_lr_rf(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "lr",
    seed: int = 42,
):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    if model_name.lower() == "lr":
        clf = LogisticRegression(max_iter=2000, n_jobs=None)
    elif model_name.lower() == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            max_depth=None,
        )
    else:
        raise ValueError("model_name must be 'lr' or 'rf'")

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else float("nan")
    return clf, auc


def gnn_one_layer_update(
    Gnx: nx.Graph,
    emb: Dict[URIRef, np.ndarray],
    alpha: float = 0.6,
) -> Dict[URIRef, np.ndarray]:
    """
    A simple one-layer message passing update (GraphSAGE/GCN-like):
        h'_v = normalize(alpha*h_v + (1-alpha)*mean_{u in N(v)} h_u)
    """
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


# ---------- Explanation path ----------
def explain_path(
    Gnx: nx.Graph,
    u: URIRef,
    v: URIRef,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> Optional[List[Tuple[URIRef, URIRef, URIRef]]]:
    """
    Return a short path as triples (node_i, predicate, node_{i+1}).
    """
    if u not in Gnx or v not in Gnx:
        return None

    try:
        path_nodes = nx.shortest_path(Gnx, u, v)
    except Exception:
        return None

    if len(path_nodes) - 1 > max_hops:
        # try bounded BFS for short paths
        try:
            path_nodes = nx.shortest_path(Gnx, u, v, method="dijkstra")
        except Exception:
            return None
        if len(path_nodes) - 1 > max_hops:
            return None

    triples = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        pred = Gnx.edges[a, b].get("predicate", None)
        if pred is None:
            pred = URIRef("urn:edge")
        triples.append((a, pred, b))
    return triples


def format_path(triples: List[Tuple[URIRef, URIRef, URIRef]]) -> str:
    """
    Pretty-print path as:
        Author(5720..) --affiliatedWith--> Affiliation(...) --...--> Author(...)
    """
    def short(n: URIRef) -> str:
        ln = uri_local(n)
        return ln if ln else str(n)

    chunks = []
    for s, p, o in triples:
        chunks.append(f"{short(s)} --{uri_local(p)}--> {short(o)}")
    return "  |  ".join(chunks)


# ---------- Main GraphRAG pipeline (Algorithm 1) ----------
@dataclass
class GraphRAGResult:
    prob: float
    decision: int
    explanation: Optional[str]
    accepted_triples: List[ProposedTriple]


def graphrag_predict_link(
    g: Graph,
    preds: Dict[str, URIRef],
    author_u: URIRef,
    author_v: URIRef,
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    sim_tau: float = DEFAULT_SIM_TAU,
    max_hops: int = DEFAULT_MAX_HOPS,
    neg_ratio: int = DEFAULT_NEG_RATIO,
    seed: int = 42,
) -> GraphRAGResult:
    """
    Implements Algorithm 1 (academic link prediction) in a reproducible, paper-friendly way.
    """
    _seed_everything(seed)

    # Step 0: Build a working KG graph (restricted to key predicates)
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

    # Step 1: Retrieve evidence passages Tx
    passages = retrieve_evidence_passages(g, author_u, author_v, preds, k=k)

    # Step 2: Initialize embeddings (surrogate if needed)
    emb0 = build_surrogate_embeddings(Gnx, dim=SURROGATE_DIM, seed=seed)

    # Step 3: Propose triples (LLMExtract surrogate)
    proposed = heuristic_extract_triples(
        g, author_u, author_v, preds, emb0, passages, sim_tau=sim_tau
    )

    # Step 4: Validate triples using ontology-filtered augmentation
    accepted = [t for t in proposed if validate_triple(g, t, tau=tau)]

    # Step 5: Build KG+ (augment)
    # (Optionally, define a confidence predicate if you already have one. Here we omit.)
    augment_graph(g, accepted, conf_pred=None)

    # Update NX with augmented edges for explanation/search
    Gnx_plus = rdf_to_nx(g, allowed)

    # Step 6: Communities + community embeddings (optional; here for completeness)
    communities = detect_communities(Gnx_plus)
    _ = [community_embedding(c, emb0) for c in communities]  # ci vectors (not directly used in this minimal loop)

    # Step 7: GNN update (one layer) to simulate using KG structure
    emb1 = gnn_one_layer_update(Gnx_plus, emb0, alpha=0.6)

    # Step 8: Train a base ML model to obtain P_ML(y|x)
    # Build a small training set from current KG+ positives and random negatives
    pos = extract_positive_links(g, preds)
    nodes = list(Gnx_plus.nodes())
    if len(pos) < 10:
        # If KG has too few explicit positives, still allow prediction via similarity
        sim = cosine(emb1[author_u], emb1[author_v])
        prob = max(0.05, min(0.95, 0.5 * (sim + 1.0)))  # map [-1,1] -> [0,1]
        decision = int(prob >= 0.5)
    else:
        neg = sample_negative_links(nodes, pos, n_samples=len(pos) * neg_ratio, seed=seed)

        pairs = list(pos) + neg
        y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)

        X = np.stack([pair_features(u, v, emb1) for (u, v) in pairs], axis=0)

        # Use Logistic Regression as "base" predictor in the algorithm loop (paper can report LR/RF/GNN separately)
        clf, _auc = train_lr_rf(X, y, model_name="lr", seed=seed)
        x_uv = pair_features(author_u, author_v, emb1).reshape(1, -1)
        prob = float(clf.predict_proba(x_uv)[0, 1])
        decision = int(prob >= 0.5)

    # Step 9: Explanation path from KG+ (use graph shortest path)
    path_triples = explain_path(Gnx_plus, author_u, author_v, max_hops=max_hops)
    explanation = format_path(path_triples) if path_triples else None

    return GraphRAGResult(prob=prob, decision=decision, explanation=explanation, accepted_triples=accepted)


# ---------- Evaluation: reproduce Figure 9/10-like metrics (minimal) ----------
def evaluate_three_settings(
    g_raw: Graph,
    g_emb: Graph,
    g_graphrag: Graph,
    preds: Dict[str, URIRef],
    seed: int = 42,
) -> None:
    """
    Minimal evaluation scaffold (for your GitHub reproducibility):
    - Baseline (CSV) is not in OWL, so here we demonstrate the *ontology-side* settings:
        (2) Ontology + Embedding
        (3) Ontology + GraphRAG

    For your paper's *full* figures, you will connect this scaffold to your real CSV pipeline.
    """
    print("NOTE: This is a scaffold. Connect it to your CSV baseline pipeline to reproduce Figure 9/10 exactly.")
    print("      Here we show how to compute metrics on ontology-derived link prediction sets.\n")


# ---------- CLI ----------
def load_owl(path: str) -> Graph:
    g = Graph()
    # rdflib auto-detects many formats; if needed, you can pass format="xml" or "turtle"
    g.parse(path)
    return g


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG for Academic Link Prediction with Ontology-Filtered Augmentation"
    )
    parser.add_argument("--owl", type=str, required=True, help="Path to AcademicLink OWL file")
    parser.add_argument("--author1", type=str, required=True, help="Author ID (digits)")
    parser.add_argument("--author2", type=str, required=True, help="Author ID (digits)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Evidence retrieval budget k")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Validation threshold tau")
    parser.add_argument("--sim_tau", type=float, default=DEFAULT_SIM_TAU, help="Similarity threshold for proposing edges")
    parser.add_argument("--max_hops", type=int, default=DEFAULT_MAX_HOPS, help="Max hops for explanation path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export_augmented_owl", type=str, default=None, help="Optional output OWL path")
    args = parser.parse_args()

    g = load_owl(args.owl)

    # Match your ontology's predicate names exactly (from your Protégé screenshot)
    pred_names = [
        "affiliatedWith",
        "authored",
        "hasAuthor1",
        "hasAuthor2",
        "hasCoAuthor",
        "hasLink",
        "relatedTo",
    ]
    preds = find_predicates_by_localname(g, pred_names)

    u = guess_author_uri(g, args.author1)
    v = guess_author_uri(g, args.author2)
    if u is None or v is None:
        raise ValueError(
            f"Could not find one/both author URIs in OWL for IDs: {args.author1}, {args.author2}\n"
            f"Tip: ensure the individual local names or rdfs:label match the digit IDs."
        )

    res = graphrag_predict_link(
        g=g,
        preds=preds,
        author_u=u,
        author_v=v,
        k=args.k,
        tau=args.tau,
        sim_tau=args.sim_tau,
        max_hops=args.max_hops,
        seed=args.seed,
    )

    print("=== GraphRAG Academic Link Prediction ===")
    print(f"Author1: {args.author1}  ({u})")
    print(f"Author2: {args.author2}  ({v})")
    print(f"Predicted collaboration probability: {res.prob:.4f}")
    print(f"Decision (1=collaborate, 0=no): {res.decision}")

    print("\n--- Explanation path (KG+) ---")
    if res.explanation:
        print(res.explanation)
    else:
        print("No short explanation path found (try increasing --max_hops).")

    print("\n--- Accepted augmented triples ---")
    if not res.accepted_triples:
        print("None (try lowering --tau or --sim_tau).")
    else:
        for t in res.accepted_triples:
            print(f"[score={t.score:.3f}] {uri_local(t.s)} --{uri_local(t.p)}--> {uri_local(t.o)}   ({t.reason})")

    # Optional export
    if args.export_augmented_owl:
        out_path = args.export_augmented_owl
        # choose an output format; "xml" is usually safe for OWL
        g.serialize(destination=out_path, format="xml")
        print(f"\nSaved augmented OWL to: {out_path}")


if __name__ == "__main__":
    main()
