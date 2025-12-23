# GraphRAG: Ontology-Filtered Augmentation for Academic Link Prediction

This repository provides a complete research prototype for **academic co-authorship link prediction** with **explanation paths**, combining:
1) **Baseline CSV-based machine learning** (Logistic Regression, Random Forest),
2) **Ontology-based semantic embedding** (knowledge-graph/ontology feature extraction), and
3) **GraphRAG-style generative augmentation** with **ontology-filtered validation** to enrich the knowledge graph before prediction.

The codebase also includes a **Streamlit application** to predict whether two authors will collaborate (i.e., form a co-author link) and to display an **interpretable explanation path** derived from the ontology graph.

---

## 1. Key Idea (What “GraphRAG” means here)

Given an academic ontology (OWL) containing **Authors**, **Publications**, **Affiliations**, and relations such as:

- `authored` (Author → Publication)
- `hasCoAuthor` (Author ↔ Author)
- `affiliatedWith` (Author → Affiliation)
- `relatedTo` (Publication/Author ↔ ResearchInterest)
- `hasAuthor1`, `hasAuthor2`, `hasLink` (link modeling for supervision)

we build a graph, retrieve evidence paths between two authors, propose additional “candidate” relations (augmentation), and **keep only those consistent with the ontology vocabulary and constraints** (ontology-filtered validation). The augmented graph is then used to generate **richer features/embeddings** for LR/RF/GNN models and produce a prediction plus an explanation path.

> Note: The current GraphRAG augmentation module is implemented in a **lightweight, reproducible, LLM-free “GraphRAG-like”** mode 

---

## 2. Repository Structure
GraphRAG/
app.py # Streamlit UI entrypoint
appWithShap.py # Optional interpretability UI (if enabled)
test_app.py # Minimal Streamlit test
modules/
data_handler.py # Loading, preprocessing
feature_extraction.py # Feature building / embeddings
model_handler.py # LR / RF / GNN training & inference
graphrag_academiclink.py # GraphRAG augmentation + explanation path for author pairs
models/ # Saved model files (example artifacts)
data/ # Place datasets / ontology here (user-provided)
README.md
.gitignore


---

## 3. Requirements

- Python **3.9+** (recommended 3.10)
- Core libraries:
  - `streamlit`, `pandas`, `numpy`, `scikit-learn`
  - `rdflib`, `owlready2`, `networkx`
  - `matplotlib`
- For GNN (optional but supported):
  - `torch`, `torch-geometric`

### Install (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install streamlit pandas numpy scikit-learn rdflib owlready2 networkx matplotlib tqdm umap-learn


