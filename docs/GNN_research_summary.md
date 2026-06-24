# GNN Incident Discovery — Research Summary

**Project:** `efficient-graph-attack-agent`  
**Spec:** `specs/GNN_incident_discovery_v1_a.md`  
**Status (June 2026):** Pipeline implemented on `main`; literature baselines runnable via `uv`; no `.jsonl` files committed to git.

---

## Executive summary

Security operations centers (SOCs) receive hundreds of alerts per shift, but analysts must **group related alerts into incidents** — a task that rarely has labeled training data at the incident level. This project implements a **weakly-supervised GNN pipeline** that learns from **alert-level malicious/benign labels only**, then discovers incident clusters by embedding and DBSCAN grouping of predicted-malicious alerts.

**Our method (HGAT)** builds a **heterogeneous alert–entity graph** (alerts linked to hosts, users, processes, IPs) and trains a 2-layer **Heterogeneous Graph Transformer** (Hu et al., WWW 2020) to produce triage scores and 64-dimensional embeddings. Unlike classic GNN-IDS work on NetFlow classification, the target output is **incident discovery**, not single-alert detection alone.

**Baselines** — simplified, runnable reimplementations of three published methods on the **same graph, split, and evaluator**:

| Method | Style | Key reference |
|--------|-------|---------------|
| GNN-IDS | Supervised GraphSAGE | Sun et al., ARES 2024 |
| GraphIDS | Self-supervised GraphSAGE + Transformer MAE | Guerra et al., NeurIPS 2025 |
| Anomal-E | Self-supervised GraphSAGE + DGI contrastive | Caville et al., KBS 2022 |

Fair side-by-side comparison: `uv run gnn-baseline-compare`. Published benchmark numbers in `literature_baselines.json` are **reference-only** (NetFlow datasets, different splits) — not reproduced here.

**Datasets:** Primary evaluation uses a ~500-alert tenant JSONL (local, not in git). Loaders also exist for DARPA TC, ExCyTIn/SecRL, LANL cyber1, and CIC-IDS 2017 — the latter four support **incident-level ground-truth metrics** where labels exist.

**Evaluation:** Node-level AUC/F1 on validation alerts; cluster proxy metrics (tactic coherence, time span) on primary data; cluster P/R/F1 where incident GT is available.

**Status:** End-to-end pipeline, baselines, tests, and uv entry points are on `main`. Full multi-dataset benchmark runs and hyperparameter tuning remain open. The GNN approach complements the repo’s rule-based **E-ACS** system (`src/eacs/`) with learned embeddings instead of hand-crafted entity/time clustering.

---

## 1. Research goal

Build a **weakly-supervised** pipeline that:

1. Takes **alert-level** labels (malicious / benign) only — **no incident-level ground truth at training time**.
2. Trains a graph neural network on a **heterogeneous SOC alert graph** (alerts linked to host, user, process, IP entities).
3. Produces **alert embeddings** and binary triage scores.
4. Clusters **predicted-malicious** alerts with DBSCAN into **discovered incident groups**.

This differs from classic GNN-IDS papers, which focus on **flow-level malicious/benign classification** on NetFlow graphs, not SOC alert incident grouping.

---

## 2. Our method (HGAT)

| Item | Detail |
|------|--------|
| **CLI name** | `hgat` |
| **Architecture** | 2-layer **Heterogeneous Graph Transformer** (PyG `HGTConv`) with **separate attention per edge type** (`alert→host`, `alert→user`, etc.) |
| **Alert features** | One-hot MITRE tactic/technique, normalized severity, time-delta from first alert |
| **Entity features** | Learned embeddings (random init, updated during training) |
| **Training** | BCEWithLogitsLoss, Adam (lr=1e-3), early stopping on **val AUC** (patience=20) |
| **Incident discovery** | DBSCAN on 64-d embeddings of malicious-predicted alerts (cosine, eps=0.3, min_samples=2) |
| **Schema** | Runtime inference from first 5 records — no hardcoded field names |

### Reference (backbone, not our novelty)

Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). **Heterogeneous Graph Transformer.** *WWW 2020*.  
- arXiv: https://arxiv.org/abs/2003.02332  
- DOI: https://doi.org/10.1145/3366423.3380027  

**Our novelty:** heterogeneous SOC alert–entity graph + weakly-supervised two-stage pipeline (GNN triage → embedding clustering) with schema-agnostic ingestion; contrast with E-ACS rule-based entity/time clustering in the same repo.

---

## 3. Literature baselines implemented

All baselines share the **same graph, train/val split, DBSCAN, and evaluator** as HGAT for fair comparison (`benchmarks/run_baseline_comparison.py`).

### 3.1 GNN-IDS (`gnn_ids`)

| Item | Detail |
|------|--------|
| **Implementation** | Supervised 2-layer **GraphSAGE** on homogeneous alert–alert graph (alerts connected via shared entities) |
| **Training** | BCE on train mask |

**Reference:**  
Sun, Z., Teixeira, A. M. H., & Toor, S. (2024). **GNN-IDS: Graph Neural Network based Intrusion Detection System.** *ARES 2024*.  
- DOI: https://doi.org/10.1145/3664476.3664515  
- Code: https://github.com/zhenlus/GNN-IDS  

**Original eval data:** Synthetic + public IDS–derived attack graphs (not our tenant JSONL).

---

### 3.2 GraphIDS (`graph_ids`)

| Item | Detail |
|------|--------|
| **Implementation** | **GraphSAGE** encoder + **Transformer masked autoencoder**; self-supervised reconstruction on **benign train** alerts, then BCE fine-tune |
| **Inference signal** | Reconstruction error + learned score head |

**Reference:**  
Guerra, L., Chapuis, T., Duc, G., Mozharovskyi, P., & Nguyen, V.-T. (2025). **Self-Supervised Learning of Graph Representations for Network Intrusion Detection.** *NeurIPS 2025*.  
- arXiv: https://arxiv.org/abs/2509.16625  
- OpenReview: https://openreview.net/forum?id=5bu1IOOvf0  
- Code: https://github.com/lorenzo9uerra/GraphIDS  

**Reported results (their data):** up to ~99.61% macro-F1, ~99.98% PR-AUC on NF-CSE-CIC-IDS2018-v3 (NetFlow). Stored as read-only rows in `benchmarks/literature_baselines.json`.

---

### 3.3 Anomal-E (`anomal_e`)

| Item | Detail |
|------|--------|
| **Implementation** | **GraphSAGE** + **DGI-style** contrastive loss (positive vs feature-shuffled graph), then BCE fine-tune |
| **Original paper** | Also uses downstream anomaly detectors (Isolation Forest, etc.) — not replicated here |

**Reference:**  
Caville, E., Lo, W. W., Layeghy, S., & Portmann, M. (2022). **Anomal-E: A Self-Supervised Network Intrusion Detection System based on Graph Neural Networks.** *Knowledge-Based Systems*, 258, 110030.  
- DOI: https://doi.org/10.1016/j.knosys.2022.110030  
- arXiv: https://arxiv.org/abs/2207.06819  

**Original eval data:** NF-UNSW-NB15, NF-BoT-IoT (NetFlow).

---

### 3.4 Supporting component papers

| Component | Reference |
|-----------|-----------|
| GraphSAGE | Hamilton, W., Ying, Z., & Leskovec, J. (2017). *NeurIPS 2017*. https://arxiv.org/abs/1706.02216 |
| Deep Graph Infomax | Veličković, P., et al. (2019). *ICLR 2019*. https://arxiv.org/abs/1809.10341 |

---

## 4. Datasets

### 4.1 Primary (tenant SOC alerts)

| Item | Detail |
|------|--------|
| **File** | `datasets/0b1972fe_backup/training_data_rich_examples.jsonl` (~500 alerts, local only) |
| **Labels** | Alert-level: True Positive – Malicious / Benign, False Positive (~15% malicious) |
| **Incident GT** | **None** — cluster quality judged via proxy metrics only |
| **Git policy** | `*.jsonl` in `.gitignore`; never committed |

Source: Edwards Lifesciences tenant backup (CrowdStrike-style alerts with MITRE, host, user, process, IP in nested `traces`).

---

### 4.2 Benchmark datasets (loaders implemented)

| Dataset | Loader | Node mapping | Labels | Incident GT for eval |
|---------|--------|--------------|--------|----------------------|
| **DARPA TC** (E3/E5) | `data/loaders/darpa_tc_loader.py` | process→process, file→host, socket→ip | Attack node labels | Yes (campaign/incident IDs) |
| **ExCyTIn-Bench / SecRL** | `data/loaders/excytin_loader.py` | alerts, devices, users, IOCs | Alert malicious/benign; incident IDs | Yes |
| **LANL cyber1** | `data/loaders/lanl_loader.py` | user, host, process, ip | Red-team = malicious | Yes (red-team sessions) |
| **CIC-IDS 2017** | `data/loaders/cicids_loader.py` | flows→alert, IPs | Attack type → binary | Yes (attack-type/day groups) |

#### Dataset references

**DARPA Transparent Computing (TC)**  
- DARPA TC program, E3/E5 sub-datasets (CADETS, THEIA, TRACE).  
- CDM provenance schema. Public releases via DARPA TC engagement.  

**ExCyTIn-Bench / SecRL**  
- Microsoft SecRL: https://github.com/microsoft/SecRL  
- ExCyTIn-Bench paper: Mudgerikar, A., et al. (2025). *ExCyTIn-Bench: A Benchmark for LLM Agents in Cyber Threat Investigation.* arXiv:2507.14201 — https://arxiv.org/abs/2507.14201  

**LANL Comprehensive Multi-Source Cyber-Security Events**  
- DOI: https://doi.org/10.17021/1179829  
- Portal: https://csr.lanl.gov/data/cyber1/  

**CIC-IDS 2017**  
- Canadian Institute for Cybersecurity / UNB.  
- https://www.unb.ca/cic/datasets/ids-2017.html  
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSp 2018*.

---

## 5. Evaluation methodology

### 5.1 Fair comparison protocol

`uv run gnn-baseline-compare` runs all methods on the **same**:

- Dataset and graph construction
- 80/20 stratified train/val split
- DBSCAN hyperparameters (`eps`, `min_samples`)
- Clustering and evaluation code

### 5.2 Metrics

**Node classification (validation set) — all methods**

| Metric | Meaning |
|--------|---------|
| AUC | Rank malicious above benign |
| F1, Precision, Recall | Hard classification at threshold 0.5 |

**Incident clustering — proxy (primary dataset, no incident GT)**

| Metric | Meaning |
|--------|---------|
| cluster_count | Non-noise DBSCAN groups |
| mean_tactic_coherence | Fraction of alerts sharing dominant MITRE tactic per cluster |
| mean_time_span_hours | Temporal span of each cluster |
| mean_distinct_entity_types | Entity-type diversity per cluster |

**Incident clustering — ground truth (benchmark datasets with incident labels)**

| Metric | Meaning |
|--------|---------|
| cluster_precision / recall / f1 | Best-match overlap vs true incident groups |

**Literature reference rows** (`benchmarks/literature_baselines.json`, `"source": "literature"`) — published numbers on **different datasets/splits**; not reproduced by this repo.

---

## 6. Repository layout (GNN pipeline)

```
main.py                          # End-to-end HGAT pipeline CLI
data/
  schema.py                      # Runtime schema inference
  graph_builder.py               # JSONL → PyG HeteroData
  loaders/                       # primary, darpa_tc, excytin, lanl, cicids
models/
  hgat.py                        # Our heterogeneous GAT
  baselines/                     # gnn_ids, graph_ids, anomal_e
training/
  trainer.py                     # HGAT training
  baseline_trainer.py            # Unified baseline training
clustering/incident_clusterer.py # DBSCAN
evaluation/evaluator.py          # Metrics
benchmarks/
  run_benchmarks.py              # Multi-dataset HGAT runner
  run_baseline_comparison.py     # Side-by-side method comparison
  comparison_table.py            # LaTeX table renderer
  literature_baselines.json      # Published reference numbers
src/gnn_entry.py                 # uv entry points
specs/GNN_incident_discovery_v1_a.md
requirements-gnn.txt
```

---

## 7. How to run (uv)

```bash
cd efficient-graph-attack-agent
uv sync --group gnn

# Our method only
uv run gnn-incident \
  --data datasets/0b1972fe_backup/training_data_rich_examples.jsonl \
  --output-dir outputs/gnn_incident

# All methods side-by-side
uv run gnn-baseline-compare \
  --dataset primary \
  --methods hgat gnn_ids graph_ids anomal_e \
  --epochs 50 \
  --output-dir outputs/baseline_comparison

# LaTeX table
uv run gnn-benchmark-table \
  --summary outputs/baseline_comparison/primary_baseline_comparison.json
```

---

## 8. Relationship to E-ACS (same repo)

The **E-ACS** system (`src/eacs/`) uses rule-based sketch filtering, entity/time clustering, and hand-crafted cluster scores on SecRL data. Reported in `RESULTS.md`.

| | E-ACS | GNN pipeline (HGAT) |
|---|-------|---------------------|
| Alert selection | Count-min sketch filter | GNN binary classifier |
| Grouping | Shared entity + time windows | DBSCAN on learned embeddings |
| Graph | Alert–entity (heuristic) | Heterogeneous GAT |
| Training signal | Rules + optional GT for eval | Alert labels only (weak supervision) |

E-ACS SecRL baselines (high_severity_only, entity_time_cluster, graph_oracle, etc.) are **separate** from the GNN literature baselines above.

---

## 9. Current status and open items

### Done

- [x] Full HGAT pipeline (graph → train → cluster → evaluate)
- [x] Schema-agnostic graph builder
- [x] Benchmark loaders (primary, DARPA, ExCyTIn, LANL, CIC-IDS)
- [x] Runnable baselines: GNN-IDS, GraphIDS, Anomal-E
- [x] Side-by-side comparison runner + LaTeX/markdown tables
- [x] uv entry points (`gnn-incident`, `gnn-baseline-compare`, etc.)
- [x] Unit tests (`tests/test_gnn_pipeline.py`, `tests/test_gnn_baselines.py`)
- [x] Pushed to `main` (commit `4528534`); `*.jsonl` gitignored

### Not yet done

- [ ] Full benchmark runs on downloaded DARPA / ExCyTIn / LANL / CIC-IDS data
- [ ] Published comparison numbers reproduced on identical splits (literature rows are reference-only)
- [ ] Incident GT evaluation on primary tenant (no labels available)
- [ ] Hyperparameter search (DBSCAN eps, HGT heads, etc.)
- [ ] Direct comparison against E-ACS on SecRL incident graphs

---

## 10. Key references (bibtex)

```bibtex
@inproceedings{hu2020hgt,
  author    = {Hu, Ziniu and Dong, Yuxiao and Wang, Kuansan and Sun, Yizhou},
  title     = {Heterogeneous Graph Transformer},
  booktitle = {WWW},
  year      = {2020},
  doi       = {10.1145/3366423.3380027}
}

@inproceedings{sun2024gnnids,
  author    = {Sun, Zhenlu and Teixeira, Andr{\'e} M. H. and Toor, Salman},
  title     = {{GNN-IDS}: Graph Neural Network based Intrusion Detection System},
  booktitle = {ARES},
  year      = {2024},
  doi       = {10.1145/3664476.3664515}
}

@inproceedings{guerra2025graphids,
  author    = {Guerra, Lorenzo and Chapuis, Thomas and Duc, Guillaume and Mozharovskyi, Pavlo and Nguyen, Van-Tam},
  title     = {Self-Supervised Learning of Graph Representations for Network Intrusion Detection},
  booktitle = {NeurIPS},
  year      = {2025},
  eprint    = {2509.16625},
  archivePrefix = {arXiv}
}

@article{caville2022anomale,
  author    = {Caville, Evan and Lo, Wai Weng and Layeghy, Siamak and Portmann, Marius},
  title     = {{Anomal-E}: A Self-Supervised Network Intrusion Detection System based on Graph Neural Networks},
  journal   = {Knowledge-Based Systems},
  volume    = {258},
  pages     = {110030},
  year      = {2022},
  doi       = {10.1016/j.knosys.2022.110030}
}

@inproceedings{sharafaldin2018cicids,
  author    = {Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {ICISSp},
  year      = {2018}
}
```

---

*Last updated: June 2026*
