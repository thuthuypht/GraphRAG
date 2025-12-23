# graphrag_multidataset.py
# Exports: proposed_triples.csv, validated_triples.csv, explanations.jsonl
# Supports: AcademicLink.owl (RDF/XML), BreastCancer.owl (RDF/XML), Diabetes.owl (OWL/XML)

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import networkx as nx
import difflib

try:
    from rdflib import Graph as RDFGraph
    from rdflib.namespace import RDF, OWL
    from rdflib.term import URIRef, Literal
except Exception:
    RDFGraph = None


# -----------------------------
# Utilities
# -----------------------------
def _read_head(path: str, nbytes: int = 2048) -> str:
    with open(path, "rb") as f:
        b = f.read(nbytes)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return str(b)


def detect_owl_flavor(path: str) -> str:
    """
    Returns:
      - 'rdfxml' for <rdf:RDF ...>
      - 'owlxml' for <Ontology ...> (OWL/XML)
      - 'unknown' otherwise
    """
    head = _read_head(path).lower()
    if "<rdf:rdf" in head:
        return "rdfxml"
    if "<ontology" in head and "http://www.w3.org/2002/07/owl" in head:
        return "owlxml"
    return "unknown"


def local_name(uri: str) -> str:
    if uri is None:
        return ""
    if uri.startswith("#"):
        return uri[1:]
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rsplit("/", 1)[-1]


def str_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# -----------------------------
# Ontology container
# -----------------------------
@dataclass
class OntologyData:
    dataset_name: str
    owl_path: str
    nodes: set
    obj_props: set
    data_props: set
    obj_triples: List[Tuple[str, str, str]]
    data_triples: List[Tuple[str, str, str]]  # literal stored as string
    # for quick validation
    data_prop_is_numeric: Dict[str, bool]


def load_rdfxml_ontology(path: str, dataset_name: str) -> OntologyData:
    if RDFGraph is None:
        raise RuntimeError("rdflib is not available. Please install rdflib (pip install rdflib).")

    g = RDFGraph()
    g.parse(path, format="xml")

    # declared properties (if present)
    obj_props = set(local_name(str(s)) for s in g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(local_name(str(s)) for s in g.subjects(RDF.type, OWL.DatatypeProperty))

    nodes = set()
    obj_triples = []
    data_triples = []

    # If declarations are missing, infer by usage: literal objects -> data props
    inferred_data_props = set()
    inferred_obj_props = set()

    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef):
            sL = local_name(str(s))
            pL = local_name(str(p))
            nodes.add(sL)
            if isinstance(o, URIRef):
                oL = local_name(str(o))
                nodes.add(oL)
                obj_triples.append((sL, pL, oL))
                inferred_obj_props.add(pL)
            elif isinstance(o, Literal):
                lit = str(o)
                data_triples.append((sL, pL, lit))
                inferred_data_props.add(pL)

    if not obj_props and inferred_obj_props:
        obj_props = inferred_obj_props
    if not data_props and inferred_data_props:
        data_props = inferred_data_props

    # detect numeric data properties
    data_prop_is_numeric = {}
    by_prop = {}
    for s, p, v in data_triples:
        by_prop.setdefault(p, []).append(v)

    for p, vals in by_prop.items():
        # numeric if most values can be parsed as float
        ok = 0
        for v in vals[:3000]:
            if safe_float(v) is not None:
                ok += 1
        data_prop_is_numeric[p] = (ok / max(1, min(len(vals), 3000))) >= 0.7

    return OntologyData(
        dataset_name=dataset_name,
        owl_path=path,
        nodes=nodes,
        obj_props=set(obj_props),
        data_props=set(data_props),
        obj_triples=obj_triples,
        data_triples=data_triples,
        data_prop_is_numeric=data_prop_is_numeric,
    )


def load_owlxml_ontology(path: str, dataset_name: str) -> OntologyData:
    # OWL/XML parser (Diabetes.owl in your case)
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    def tag_name(el):
        return el.tag.split("}")[-1]

    nodes = set()
    obj_props = set()
    data_props = set()
    obj_triples = []
    data_triples = []

    # Read declarations / assertions
    for el in root.iter():
        t = tag_name(el)
        if t == "NamedIndividual" and "IRI" in el.attrib:
            nodes.add(local_name(el.attrib["IRI"]))
        if t == "ObjectProperty" and "IRI" in el.attrib:
            obj_props.add(local_name(el.attrib["IRI"]))
        if t == "DataProperty" and "IRI" in el.attrib:
            data_props.add(local_name(el.attrib["IRI"]))

        if t == "ObjectPropertyAssertion":
            prop = None
            src = None
            tgt = None
            for child in el:
                ct = tag_name(child)
                if ct == "ObjectProperty":
                    prop = local_name(child.attrib.get("IRI"))
                elif ct == "NamedIndividual" and src is None:
                    src = local_name(child.attrib.get("IRI"))
                elif ct == "NamedIndividual":
                    tgt = local_name(child.attrib.get("IRI"))
            if prop and src and tgt:
                nodes.add(src)
                nodes.add(tgt)
                obj_triples.append((src, prop, tgt))

        if t == "DataPropertyAssertion":
            prop = None
            src = None
            lit = None
            for child in el:
                ct = tag_name(child)
                if ct == "DataProperty":
                    prop = local_name(child.attrib.get("IRI"))
                elif ct == "NamedIndividual":
                    src = local_name(child.attrib.get("IRI"))
                elif ct == "Literal":
                    lit = child.text
            if prop and src and lit is not None:
                nodes.add(src)
                data_triples.append((src, prop, str(lit)))

    # numeric detection for OWL/XML data props (if any)
    data_prop_is_numeric = {}
    if data_triples:
        by_prop = {}
        for s, p, v in data_triples:
            by_prop.setdefault(p, []).append(v)
        for p, vals in by_prop.items():
            ok = 0
            for v in vals[:3000]:
                if safe_float(v) is not None:
                    ok += 1
            data_prop_is_numeric[p] = (ok / max(1, min(len(vals), 3000))) >= 0.7

    return OntologyData(
        dataset_name=dataset_name,
        owl_path=path,
        nodes=nodes,
        obj_props=set(obj_props),
        data_props=set(data_props),
        obj_triples=obj_triples,
        data_triples=data_triples,
        data_prop_is_numeric=data_prop_is_numeric,
    )


def load_ontology(path: str, dataset_name: str) -> OntologyData:
    flavor = detect_owl_flavor(path)
    if flavor == "rdfxml":
        return load_rdfxml_ontology(path, dataset_name)
    if flavor == "owlxml":
        return load_owlxml_ontology(path, dataset_name)

    # try rdfxml as fallback
    try:
        return load_rdfxml_ontology(path, dataset_name)
    except Exception:
        return load_owlxml_ontology(path, dataset_name)


# -----------------------------
# Graph building
# -----------------------------
def build_multidigraph(onto: OntologyData, include_literals: bool = True) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n in onto.nodes:
        G.add_node(n)

    # original object edges
    for s, p, o in onto.obj_triples:
        G.add_edge(s, o, key=f"KG::{p}", predicate=p, source="KG", weight=1.0)

    # original data edges (optional as literal nodes)
    if include_literals:
        for s, p, v in onto.data_triples:
            lit_node = f"LIT::{p}={v}"
            G.add_node(lit_node)
            G.add_edge(s, lit_node, key=f"KG::{p}", predicate=p, source="KG", weight=1.0)

    return G


# -----------------------------
# Evidence retrieval
# -----------------------------
def bfs_collect_edges(G: nx.MultiDiGraph, start: str, max_hops: int, limit: int = 200) -> List[Tuple[str, str, str, str]]:
    """
    Collect a set of nearby edges around `start`.
    Returns list of edges as (s, p, o, source)
    """
    if start not in G:
        return []

    visited = {start: 0}
    q = [start]
    edges = []
    while q:
        u = q.pop(0)
        d = visited[u]
        if d >= max_hops:
            continue

        # outgoing
        for v, edict in G[u].items():
            for k, attr in edict.items():
                p = attr.get("predicate", "")
                src = attr.get("source", "KG")
                edges.append((u, p, v, src))
                if len(edges) >= limit:
                    return edges
            if v not in visited:
                visited[v] = d + 1
                q.append(v)

        # incoming (helps when graph is directed)
        for v in G.predecessors(u):
            for k, attr in G[v][u].items():
                p = attr.get("predicate", "")
                src = attr.get("source", "KG")
                edges.append((v, p, u, src))
                if len(edges) >= limit:
                    return edges
            if v not in visited:
                visited[v] = d + 1
                q.append(v)

    return edges[:limit]


def collect_edges_between(G: nx.MultiDiGraph, a: str, b: str, max_hops: int, limit: int = 300) -> List[Tuple[str, str, str, str]]:
    """
    Collect edges along BFS layers from a, prioritizing nodes that can reach b quickly.
    """
    if a not in G or b not in G:
        return []

    # use undirected for reachability
    UG = nx.Graph()
    for u, v, attr in G.edges(data=True):
        UG.add_edge(u, v)

    if not nx.has_path(UG, a, b):
        return bfs_collect_edges(G, a, max_hops=max_hops, limit=limit)

    # get one shortest path, then collect neighborhoods along it
    path = nx.shortest_path(UG, a, b)
    edges = []
    for node in path:
        edges.extend(bfs_collect_edges(G, node, max_hops=1, limit=limit))
        if len(edges) >= limit:
            break

    # de-dup
    seen = set()
    out = []
    for s, p, o, src in edges:
        key = (s, p, o, src)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:limit]


# -----------------------------
# Triple proposal + validation
# -----------------------------
@dataclass
class TripleCand:
    subject: str
    predicate_raw: str
    obj_raw: str
    confidence: float


def align_predicate(pred: str, onto: OntologyData, sim_tau: float) -> Tuple[Optional[str], float]:
    if pred in onto.obj_props or pred in onto.data_props:
        return pred, 1.0

    best_p = None
    best_s = 0.0
    for p in list(onto.obj_props)[:5000]:
        s = str_sim(pred, p)
        if s > best_s:
            best_s, best_p = s, p
    for p in list(onto.data_props)[:5000]:
        s = str_sim(pred, p)
        if s > best_s:
            best_s, best_p = s, p

    if best_s >= sim_tau:
        return best_p, best_s
    return None, best_s


def align_object(obj: str, onto: OntologyData, sim_tau: float) -> Tuple[Optional[str], float]:
    if obj in onto.nodes:
        return obj, 1.0
    # allow literal nodes as-is
    if obj.startswith("LIT::"):
        return obj, 1.0

    # try nearest node name (limited scan for speed)
    best = None
    best_s = 0.0

    # heuristic: if looks like Patient_###, filter candidates
    if "patient" in obj.lower():
        candidates = [n for n in onto.nodes if "Patient_" in n]
        candidates = candidates[:4000]
    else:
        candidates = list(onto.nodes)[:8000]

    for n in candidates:
        s = str_sim(obj, n)
        if s > best_s:
            best_s, best = s, n

    if best_s >= sim_tau:
        return best, best_s
    return None, best_s


def propose_triples(
    evidence_edges: List[Tuple[str, str, str, str]],
    onto: OntologyData,
    k: int,
    noise_rate: float,
    seed: int,
) -> List[TripleCand]:
    """
    Evidence edges are already structured triples. We simulate an LLM extractor by:
      - taking some correct edges (high confidence)
      - injecting noisy edges with misspelled predicates / random objects (lower confidence)
    """
    rnd = random.Random(seed)

    # sample evidence edges
    if not evidence_edges:
        return []

    sample = evidence_edges[:]
    rnd.shuffle(sample)
    sample = sample[: min(k, len(sample))]

    cands: List[TripleCand] = []

    for (s, p, o, src) in sample:
        # base correct triple
        base_conf = 0.90 if src == "KG" else 0.85
        cands.append(TripleCand(s, p, o, base_conf))

        # noise / hallucination
        if rnd.random() < noise_rate:
            mode = rnd.choice(["typo_pred", "unknown_pred", "random_obj"])

            if mode == "typo_pred":
                # small typo to allow alignment sometimes
                if len(p) >= 4:
                    i = rnd.randint(0, len(p) - 1)
                    p2 = p[:i] + rnd.choice("abcdefghijklmnopqrstuvwxyz") + p[i + 1 :]
                else:
                    p2 = p + "x"
                cands.append(TripleCand(s, p2, o, 0.55))

            elif mode == "unknown_pred":
                p2 = "has_" + p + "_extra"
                cands.append(TripleCand(s, p2, o, 0.40))

            else:  # random_obj
                o2 = rnd.choice(list(onto.nodes)) if onto.nodes else (o + "_x")
                cands.append(TripleCand(s, p, o2, 0.45))

    # also add a few completely random triples
    extra = max(1, k // 10)
    props_pool = list(onto.obj_props | onto.data_props)
    for _ in range(extra):
        if not props_pool or not onto.nodes:
            break
        s = rnd.choice(list(onto.nodes))
        p = rnd.choice(props_pool)
        o = rnd.choice(list(onto.nodes))
        cands.append(TripleCand(s, p, o, 0.35))

    return cands


def validate_triple(
    cand: TripleCand,
    onto: OntologyData,
    tau: float,
    sim_tau: float,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate + align:
      - predicate must be known or alignable (>= sim_tau)
      - subject must exist
      - object must exist for object properties, or be numeric if numeric data property
    """
    subj_ok = cand.subject in onto.nodes
    pred_aligned, pred_sim = align_predicate(cand.predicate_raw, onto, sim_tau)

    if not subj_ok:
        return False, {
            "subject_aligned": None,
            "predicate_aligned": pred_aligned,
            "object_aligned": None,
            "predicate_sim": pred_sim,
            "reason": "subject_not_in_ontology",
        }

    if pred_aligned is None:
        return False, {
            "subject_aligned": cand.subject,
            "predicate_aligned": None,
            "object_aligned": None,
            "predicate_sim": pred_sim,
            "reason": "predicate_not_in_ontology",
        }

    is_obj = pred_aligned in onto.obj_props
    is_data = pred_aligned in onto.data_props

    if not (is_obj or is_data):
        return False, {
            "subject_aligned": cand.subject,
            "predicate_aligned": pred_aligned,
            "object_aligned": None,
            "predicate_sim": pred_sim,
            "reason": "predicate_type_unknown",
        }

    # object alignment / check
    if is_obj:
        obj_aligned, obj_sim = align_object(cand.obj_raw, onto, sim_tau)
        if obj_aligned is None:
            return False, {
                "subject_aligned": cand.subject,
                "predicate_aligned": pred_aligned,
                "object_aligned": None,
                "object_sim": obj_sim,
                "predicate_sim": pred_sim,
                "reason": "object_not_in_ontology",
            }

        conf = cand.confidence * max(0.5, pred_sim) * max(0.5, obj_sim)
        if conf < tau:
            return False, {
                "subject_aligned": cand.subject,
                "predicate_aligned": pred_aligned,
                "object_aligned": obj_aligned,
                "predicate_sim": pred_sim,
                "object_sim": obj_sim,
                "final_confidence": conf,
                "reason": "below_tau",
            }

        return True, {
            "subject_aligned": cand.subject,
            "predicate_aligned": pred_aligned,
            "object_aligned": obj_aligned,
            "predicate_sim": pred_sim,
            "object_sim": obj_sim,
            "final_confidence": conf,
            "reason": "ok",
        }

    # data property
    # For RDF/XML datasets we stored data edges as literals; for Graph building we used LIT nodes.
    # Here, accept any string if property is non-numeric; for numeric require float.
    val = cand.obj_raw
    is_numeric = onto.data_prop_is_numeric.get(pred_aligned, False)
    if is_numeric and safe_float(val) is None:
        return False, {
            "subject_aligned": cand.subject,
            "predicate_aligned": pred_aligned,
            "object_aligned": None,
            "predicate_sim": pred_sim,
            "reason": "non_numeric_literal_for_numeric_property",
        }

    conf = cand.confidence * max(0.5, pred_sim)
    if conf < tau:
        return False, {
            "subject_aligned": cand.subject,
            "predicate_aligned": pred_aligned,
            "object_aligned": val,
            "predicate_sim": pred_sim,
            "final_confidence": conf,
            "reason": "below_tau",
        }

    return True, {
        "subject_aligned": cand.subject,
        "predicate_aligned": pred_aligned,
        "object_aligned": val,
        "predicate_sim": pred_sim,
        "final_confidence": conf,
        "reason": "ok",
    }


def add_validated_to_graph(G: nx.MultiDiGraph, validated_rows: List[Dict[str, Any]]):
    for r in validated_rows:
        s = r["subject_aligned"]
        p = r["predicate_aligned"]
        o = r["object_aligned"]
        w = float(r.get("final_confidence", 0.8))

        if p is None or s is None or o is None:
            continue

        # data props become literal nodes
        if r.get("object_kind") == "data":
            lit_node = f"LIT::{p}={o}"
            if lit_node not in G:
                G.add_node(lit_node)
            G.add_edge(s, lit_node, key=f"GraphRAG::{p}", predicate=p, source="GraphRAG", weight=w)
        else:
            if o not in G:
                G.add_node(o)
            G.add_edge(s, o, key=f"GraphRAG::{p}", predicate=p, source="GraphRAG", weight=w)


# -----------------------------
# Explainability output (paths)
# -----------------------------
def community_representative(UG: nx.Graph, node: str) -> Optional[str]:
    if node not in UG or UG.number_of_nodes() == 0:
        return None
    # greedy modularity (works without extra packages)
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(UG))
    except Exception:
        comms = [set(UG.nodes())]

    for c in comms:
        if node in c:
            # representative = highest degree inside community
            best = None
            best_deg = -1
            for n in c:
                d = UG.degree(n)
                if d > best_deg:
                    best_deg = d
                    best = n
            return best
    return None


def pick_edge_label(G: nx.MultiDiGraph, u: str, v: str) -> str:
    # prefer GraphRAG edges if exist
    if G.has_edge(u, v):
        candidates = list(G[u][v].values())
        # sort by source preference + weight
        candidates.sort(key=lambda a: (a.get("source") != "GraphRAG", -a.get("weight", 1.0)))
        return candidates[0].get("predicate", "relatedTo")
    if G.has_edge(v, u):
        candidates = list(G[v][u].values())
        candidates.sort(key=lambda a: (a.get("source") != "GraphRAG", -a.get("weight", 1.0)))
        return candidates[0].get("predicate", "relatedTo")
    return "relatedTo"


def build_explanation_path(G: nx.MultiDiGraph, start: str, end: str, max_hops: int) -> List[Tuple[str, str, str]]:
    if start not in G or end not in G:
        return []
    UG = nx.Graph()
    for u, v in G.edges():
        UG.add_edge(u, v)
    if not nx.has_path(UG, start, end):
        return []

    path = nx.shortest_path(UG, start, end)
    if len(path) - 1 > max_hops:
        path = path[: max_hops + 1]

    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        p = pick_edge_label(G, u, v)
        edges.append((u, p, v))
    return edges


# -----------------------------
# Dataset-specific labels / sampling
# -----------------------------
def academic_is_positive(onto: OntologyData, a1: str, a2: str) -> int:
    # positive if hasCoAuthor edge in either direction
    for s, p, o in onto.obj_triples:
        if p == "hasCoAuthor" and ((s == a1 and o == a2) or (s == a2 and o == a1)):
            return 1
    return 0


def diabetes_label(onto: OntologyData, patient_id: str, pos_relation: str = "hasOutcome") -> int:
    # in your Diabetes.owl: positive patients have a self-loop (patient hasOutcome patient)
    for s, p, o in onto.obj_triples:
        if p == pos_relation and s == patient_id and o == patient_id:
            return 1
    return 0


def breast_label(onto: OntologyData, patient_id: str, label_dp: str = "diagnosis", pos_value: str = "M") -> int:
    # BreastCancer.owl uses data property diagnosis: 'M' or 'B'
    for s, p, v in onto.data_triples:
        if s == patient_id and p == label_dp:
            return 1 if str(v).strip() == pos_value else 0
    return 0


def list_instances_for_dataset(onto: OntologyData, dataset_name: str) -> List[str]:
    # heuristic instance listing
    if dataset_name.lower() == "breastcancer":
        # Breast individuals are Patient_###
        inst = [n for n in onto.nodes if n.startswith("Patient_")]
        inst.sort()
        return inst
    if dataset_name.lower() == "diabetes":
        # Diabetes individuals are numeric strings "1".."768"
        # keep only those that look like integers and are in range
        inst = []
        for n in onto.nodes:
            if re.fullmatch(r"\d+", n):
                inst.append(n)
        inst = sorted(inst, key=lambda x: int(x))
        # likely first 768 are patients; keep up to 1000 to be safe
        return inst[:1000]
    # Academic: authors are numeric IDs (long)
    inst = []
    for n in onto.nodes:
        if re.fullmatch(r"\d{6,}", n):
            inst.append(n)
    inst.sort()
    return inst


def stratified_sample(instances: List[str], labels: Dict[str, int], n: int, pos_frac: float, seed: int) -> List[str]:
    rnd = random.Random(seed)
    pos = [i for i in instances if labels.get(i, 0) == 1]
    neg = [i for i in instances if labels.get(i, 0) == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    n_pos = int(round(n * pos_frac))
    n_neg = max(0, n - n_pos)
    return pos[:n_pos] + neg[:n_neg]


# -----------------------------
# Main run
# -----------------------------
def run_one_instance(
    onto: OntologyData,
    dataset_name: str,
    G0: nx.MultiDiGraph,
    instance_id: str,
    run_tag: str,
    export_dir: Path,
    k: int,
    tau: float,
    sim_tau: float,
    max_hops: int,
    noise_rate: float,
    seed: int,
    label_info: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # evidence = neighborhood edges
    evidence = bfs_collect_edges(G0, instance_id, max_hops=max_hops, limit=max(200, k * 4))

    cands = propose_triples(evidence, onto, k=k, noise_rate=noise_rate, seed=seed)

    proposed_rows = []
    validated_rows = []

    for c in cands:
        ok, meta = validate_triple(c, onto, tau=tau, sim_tau=sim_tau)

        pred_aligned = meta.get("predicate_aligned")
        obj_aligned = meta.get("object_aligned")

        object_kind = "object"
        if pred_aligned in onto.data_props:
            object_kind = "data"

        row = {
            "dataset": dataset_name,
            "run_tag": run_tag,
            "instance_id": instance_id,
            "subject": c.subject,
            "predicate_raw": c.predicate_raw,
            "object_raw": c.obj_raw,
            "confidence_raw": c.confidence,
            "predicate_aligned": pred_aligned,
            "object_aligned": obj_aligned,
            "object_kind": object_kind,
            "predicate_sim": meta.get("predicate_sim"),
            "object_sim": meta.get("object_sim"),
            "final_confidence": meta.get("final_confidence"),
            "validated": int(ok),
            "reason": meta.get("reason"),
        }
        proposed_rows.append(row)

        if ok:
            meta2 = dict(meta)
            meta2["dataset"] = dataset_name
            meta2["run_tag"] = run_tag
            meta2["instance_id"] = instance_id
            meta2["object_kind"] = object_kind
            validated_rows.append(meta2)

    df_prop = pd.DataFrame(proposed_rows)
    df_val = pd.DataFrame([{
        "dataset": r["dataset"],
        "run_tag": r["run_tag"],
        "instance_id": r["instance_id"],
        "subject_aligned": r.get("subject_aligned"),
        "predicate_aligned": r.get("predicate_aligned"),
        "object_aligned": r.get("object_aligned"),
        "object_kind": r.get("object_kind"),
        "predicate_sim": r.get("predicate_sim"),
        "object_sim": r.get("object_sim"),
        "final_confidence": r.get("final_confidence"),
        "reason": r.get("reason"),
    } for r in validated_rows])

    # build KG+ and explanation path
    G = G0.copy()
    add_validated_to_graph(G, validated_rows)

    UG = nx.Graph()
    for u, v in G.edges():
        UG.add_edge(u, v)

    rep = community_representative(UG, instance_id)
    if rep is None:
        rep = instance_id

    exp_path = build_explanation_path(G, instance_id, rep, max_hops=max_hops)

    # hallucination / acceptance stats
    n_prop = len(df_prop)
    n_val = int(df_prop["validated"].sum()) if n_prop > 0 else 0
    halluc_rate = 1.0 - (n_val / max(1, n_prop))

    exp = {
        "dataset": dataset_name,
        "run_tag": run_tag,
        "instance_id": instance_id,
        "label": label_info.get("label"),
        "label_definition": label_info.get("label_definition"),
        "n_proposed": n_prop,
        "n_validated": n_val,
        "hallucination_rate": round(halluc_rate, 4),
        "community_representative": rep,
        "explanation_path": [{"s": s, "p": p, "o": o} for (s, p, o) in exp_path],
    }

    return df_prop, df_val, exp


def run_academic_pair(
    onto: OntologyData,
    G0: nx.MultiDiGraph,
    author1: str,
    author2: str,
    run_tag: str,
    export_dir: Path,
    k: int,
    tau: float,
    sim_tau: float,
    max_hops: int,
    noise_rate: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    evidence = collect_edges_between(G0, author1, author2, max_hops=max_hops, limit=max(300, k * 6))
    cands = propose_triples(evidence, onto, k=k, noise_rate=noise_rate, seed=seed)

    proposed_rows = []
    validated_rows = []

    for c in cands:
        ok, meta = validate_triple(c, onto, tau=tau, sim_tau=sim_tau)

        pred_aligned = meta.get("predicate_aligned")
        obj_aligned = meta.get("object_aligned")

        object_kind = "object"
        if pred_aligned in onto.data_props:
            object_kind = "data"

        row = {
            "dataset": "Academic",
            "run_tag": run_tag,
            "pair": f"{author1}__{author2}",
            "subject": c.subject,
            "predicate_raw": c.predicate_raw,
            "object_raw": c.obj_raw,
            "confidence_raw": c.confidence,
            "predicate_aligned": pred_aligned,
            "object_aligned": obj_aligned,
            "object_kind": object_kind,
            "predicate_sim": meta.get("predicate_sim"),
            "object_sim": meta.get("object_sim"),
            "final_confidence": meta.get("final_confidence"),
            "validated": int(ok),
            "reason": meta.get("reason"),
        }
        proposed_rows.append(row)

        if ok:
            meta2 = dict(meta)
            meta2["dataset"] = "Academic"
            meta2["run_tag"] = run_tag
            meta2["pair"] = f"{author1}__{author2}"
            meta2["object_kind"] = object_kind
            validated_rows.append(meta2)

    df_prop = pd.DataFrame(proposed_rows)
    df_val = pd.DataFrame([{
        "dataset": r["dataset"],
        "run_tag": r["run_tag"],
        "pair": r["pair"],
        "subject_aligned": r.get("subject_aligned"),
        "predicate_aligned": r.get("predicate_aligned"),
        "object_aligned": r.get("object_aligned"),
        "object_kind": r.get("object_kind"),
        "predicate_sim": r.get("predicate_sim"),
        "object_sim": r.get("object_sim"),
        "final_confidence": r.get("final_confidence"),
        "reason": r.get("reason"),
    } for r in validated_rows])

    G = G0.copy()
    add_validated_to_graph(G, validated_rows)

    # explanation path between authors in KG+
    exp_path = build_explanation_path(G, author1, author2, max_hops=max_hops)

    y = academic_is_positive(onto, author1, author2)
    label_def = "y=1 if (author1 hasCoAuthor author2) in the ontology; else y=0."

    n_prop = len(df_prop)
    n_val = int(df_prop["validated"].sum()) if n_prop > 0 else 0
    halluc_rate = 1.0 - (n_val / max(1, n_prop))

    exp = {
        "dataset": "Academic",
        "run_tag": run_tag,
        "author1": author1,
        "author2": author2,
        "label": y,
        "label_definition": label_def,
        "n_proposed": n_prop,
        "n_validated": n_val,
        "hallucination_rate": round(halluc_rate, 4),
        "explanation_path": [{"s": s, "p": p, "o": o} for (s, p, o) in exp_path],
    }

    return df_prop, df_val, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", required=True, help="Path to .owl file")
    ap.add_argument("--dataset_name", required=True, choices=["Academic", "Diabetes", "BreastCancer"])
    ap.add_argument("--export_dir", default="exports")
    ap.add_argument("--run_tag", default="run")

    # Academic pair
    ap.add_argument("--author1", default=None)
    ap.add_argument("--author2", default=None)

    # Medical single-instance
    ap.add_argument("--instance_id", default=None)

    # Batch evaluation
    ap.add_argument("--batch_eval", action="store_true")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--pos_frac", type=float, default=0.5)

    # GraphRAG knobs
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--sim_tau", type=float, default=0.65)
    ap.add_argument("--max_hops", type=int, default=4)
    ap.add_argument("--noise_rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)

    # Breast label controls
    ap.add_argument("--breast_label_dp", default="diagnosis")
    ap.add_argument("--breast_pos_value", default="M")

    # Diabetes label controls
    ap.add_argument("--diabetes_pos_relation", default="hasOutcome")

    args = ap.parse_args()

    owl_path = str(Path(args.owl).resolve())
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    onto = load_ontology(owl_path, args.dataset_name)
    G0 = build_multidigraph(onto, include_literals=True)

    # -------------------------
    # Academic
    # -------------------------
    if args.dataset_name == "Academic":
        if args.batch_eval:
            # sample random author pairs
            all_authors = list_instances_for_dataset(onto, "Academic")
            rnd = random.Random(args.seed)
            rnd.shuffle(all_authors)

            pairs = []
            # build pairs with half positives if possible
            # (simple heuristic: random pairs then check label)
            need = args.n_eval
            trials = 0
            while len(pairs) < need and trials < need * 50:
                a1 = rnd.choice(all_authors)
                a2 = rnd.choice(all_authors)
                if a1 == a2:
                    trials += 1
                    continue
                y = academic_is_positive(onto, a1, a2)
                pairs.append((a1, a2, y))
                trials += 1

            # stratify by label
            pos = [(a1, a2) for a1, a2, y in pairs if y == 1]
            neg = [(a1, a2) for a1, a2, y in pairs if y == 0]
            rnd.shuffle(pos)
            rnd.shuffle(neg)
            n_pos = int(round(args.n_eval * args.pos_frac))
            n_neg = max(0, args.n_eval - n_pos)
            final_pairs = pos[:n_pos] + neg[:n_neg]
            if len(final_pairs) < args.n_eval:
                final_pairs = (pos + neg)[: args.n_eval]

            all_prop = []
            all_val = []
            exp_path = export_dir / f"Academic_{args.run_tag}_explanations.jsonl"

            with open(exp_path, "w", encoding="utf-8") as fexp:
                for i, (a1, a2) in enumerate(final_pairs, 1):
                    print(f"[Academic] {i}/{len(final_pairs)} pair={a1},{a2}")
                    dfp, dfv, exp = run_academic_pair(
                        onto, G0, a1, a2, args.run_tag, export_dir,
                        k=args.k, tau=args.tau, sim_tau=args.sim_tau,
                        max_hops=args.max_hops, noise_rate=args.noise_rate,
                        seed=args.seed + i,
                    )
                    all_prop.append(dfp)
                    all_val.append(dfv)
                    fexp.write(json.dumps(exp, ensure_ascii=False) + "\n")

            df_prop = pd.concat(all_prop, ignore_index=True) if all_prop else pd.DataFrame()
            df_val = pd.concat(all_val, ignore_index=True) if all_val else pd.DataFrame()

            p1 = export_dir / f"Academic_{args.run_tag}_proposed_triples.csv"
            p2 = export_dir / f"Academic_{args.run_tag}_validated_triples.csv"
            df_prop.to_csv(p1, index=False)
            df_val.to_csv(p2, index=False)
            print(f"[SAVED] {p1}")
            print(f"[SAVED] {p2}")
            print(f"[SAVED] {exp_path}")
            return

        # demo pair
        if not args.author1 or not args.author2:
            raise SystemExit("Academic requires --author1 and --author2 (or use --batch_eval).")

        dfp, dfv, exp = run_academic_pair(
            onto, G0, args.author1, args.author2, args.run_tag, export_dir,
            k=args.k, tau=args.tau, sim_tau=args.sim_tau,
            max_hops=args.max_hops, noise_rate=args.noise_rate,
            seed=args.seed,
        )

        p1 = export_dir / f"Academic_{args.run_tag}_proposed_triples.csv"
        p2 = export_dir / f"Academic_{args.run_tag}_validated_triples.csv"
        p3 = export_dir / f"Academic_{args.run_tag}_explanations.jsonl"
        dfp.to_csv(p1, index=False)
        dfv.to_csv(p2, index=False)
        with open(p3, "w", encoding="utf-8") as f:
            f.write(json.dumps(exp, ensure_ascii=False) + "\n")
        print(f"[SAVED] {p1}")
        print(f"[SAVED] {p2}")
        print(f"[SAVED] {p3}")
        return

    # -------------------------
    # Diabetes / BreastCancer
    # -------------------------
    dataset = args.dataset_name

    if args.batch_eval:
        instances = list_instances_for_dataset(onto, dataset)
        if not instances:
            raise SystemExit(f"No instances detected for dataset={dataset}. Please check the OWL file.")

        # labels for stratified sample
        labels = {}
        if dataset == "Diabetes":
            for i in instances:
                labels[i] = diabetes_label(onto, i, pos_relation=args.diabetes_pos_relation)
            label_def = f"y=1 if ({args.diabetes_pos_relation}(patient, patient)) exists in Diabetes.owl; else y=0."
        else:
            for i in instances:
                labels[i] = breast_label(onto, i, label_dp=args.breast_label_dp, pos_value=args.breast_pos_value)
            label_def = f"y=1 if {args.breast_label_dp} == '{args.breast_pos_value}', else y=0."

        chosen = stratified_sample(instances, labels, n=args.n_eval, pos_frac=args.pos_frac, seed=args.seed)

        all_prop = []
        all_val = []
        exp_path = export_dir / f"{dataset}_{args.run_tag}_explanations.jsonl"
        with open(exp_path, "w", encoding="utf-8") as fexp:
            for idx, inst in enumerate(chosen, 1):
                y = labels.get(inst, 0)
                print(f"[{dataset}] {idx}/{len(chosen)} instance={inst} label={y}")

                label_info = {"label": int(y), "label_definition": label_def}
                dfp, dfv, exp = run_one_instance(
                    onto, dataset, G0, inst, args.run_tag, export_dir,
                    k=args.k, tau=args.tau, sim_tau=args.sim_tau,
                    max_hops=args.max_hops, noise_rate=args.noise_rate,
                    seed=args.seed + idx,
                    label_info=label_info,
                )
                all_prop.append(dfp)
                all_val.append(dfv)
                fexp.write(json.dumps(exp, ensure_ascii=False) + "\n")

        df_prop = pd.concat(all_prop, ignore_index=True) if all_prop else pd.DataFrame()
        df_val = pd.concat(all_val, ignore_index=True) if all_val else pd.DataFrame()

        p1 = export_dir / f"{dataset}_{args.run_tag}_proposed_triples.csv"
        p2 = export_dir / f"{dataset}_{args.run_tag}_validated_triples.csv"
        df_prop.to_csv(p1, index=False)
        df_val.to_csv(p2, index=False)
        print(f"[SAVED] {p1}")
        print(f"[SAVED] {p2}")
        print(f"[SAVED] {exp_path}")
        return

    # demo single instance
    if not args.instance_id:
        raise SystemExit(f"{dataset} requires --instance_id (or use --batch_eval).")

    if dataset == "Diabetes":
        y = diabetes_label(onto, args.instance_id, pos_relation=args.diabetes_pos_relation)
        label_def = f"y=1 if ({args.diabetes_pos_relation}(patient, patient)) exists; else y=0."
    else:
        y = breast_label(onto, args.instance_id, label_dp=args.breast_label_dp, pos_value=args.breast_pos_value)
        label_def = f"y=1 if {args.breast_label_dp} == '{args.breast_pos_value}', else y=0."

    label_info = {"label": int(y), "label_definition": label_def}

    dfp, dfv, exp = run_one_instance(
        onto, dataset, G0, args.instance_id, args.run_tag, export_dir,
        k=args.k, tau=args.tau, sim_tau=args.sim_tau,
        max_hops=args.max_hops, noise_rate=args.noise_rate,
        seed=args.seed,
        label_info=label_info,
    )

    p1 = export_dir / f"{dataset}_{args.run_tag}_proposed_triples.csv"
    p2 = export_dir / f"{dataset}_{args.run_tag}_validated_triples.csv"
    p3 = export_dir / f"{dataset}_{args.run_tag}_explanations.jsonl"
    dfp.to_csv(p1, index=False)
    dfv.to_csv(p2, index=False)
    with open(p3, "w", encoding="utf-8") as f:
        f.write(json.dumps(exp, ensure_ascii=False) + "\n")

    print(f"[SAVED] {p1}")
    print(f"[SAVED] {p2}")
    print(f"[SAVED] {p3}")


if __name__ == "__main__":
    main()
