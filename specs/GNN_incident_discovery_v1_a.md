Implement a modular PyTorch + PyG pipeline for weakly-supervised GNN incident discovery on a SOC alert dataset. The goal is: given alert-level malicious/benign labels, train a heterogeneous GNN to produce alert embeddings, then cluster malicious-predicted alerts into incident groups — without ever having incident-level ground truth labels.

Dataset


Format: .jsonl, one alert per line
Each alert has: alert fields (e.g. alert_id, tactic, technique, severity, timestamp, label: malicious|benign) and entity fields (e.g. host, user, process, ip) — extract actual field names from the first 5 records before writing any schema assumptions into code
~500 records; fits entirely in memory

primary 500-alerts JSONL is in datasets/0b1972fe_backup/training_data_rich_examples.jsonl

Benchmark Datasets
In addition to the primary 500-alert JSONL, the pipeline must support the following public datasets for benchmarking and comparison. Each requires a dedicated loader in data/loaders/ that outputs the same HeteroData format as the primary dataset loader, so all downstream modules (trainer, clusterer, evaluator) work without modification.

1. DARPA Transparent Computing (TC) — E3 & E5


Download: publicly available via DARPA (E3 CADETS, THEIA, TRACE sub-datasets)
Format: raw provenance logs (JSON); use the parsed CDM-schema format where available
Node mapping: process nodes → process, file nodes → host, network socket nodes → ip
Edge mapping: system call events (read/write/connect/exec) → alert edges to entity nodes
Labels: DARPA provides ground-truth attack node/edge labels — use as both alert-level labels (for training) AND incident-level ground truth (for cluster evaluation)
This is the only dataset where evaluator.py should compute cluster-level precision/recall/F1 against ground-truth incident groupings, in addition to proxy metrics
Loader file: data/loaders/darpa_tc_loader.py


2. ExCyTIn-Bench — Microsoft (2025)


Download: https://github.com/microsoft/SecRL
Format: MySQL database / CSV exports from Microsoft Sentinel (57 log tables); 44-day log stream spanning multiple incidents
Node mapping: alerts → alert nodes; devices → host; users → user; IOCs (IPs) → ip
Labels: incident-level groupings are provided — use alert-level malicious/benign as training signal, incident IDs as evaluation ground truth
Note: the dataset already constructs bipartite incident graphs linking alerts and entities — use this graph structure directly as input to graph_builder.py rather than reconstructing from raw logs
Loader file: data/loaders/excytin_loader.py


3. LANL Comprehensive Multi-Source Cyber-Security Events


Download: https://csr.lanl.gov/data/cyber1/ (doi:10.17021/1179829)
Format: CSV files for auth events, process events, DNS, network flows, and red team events; 58-day log, ~12GB compressed
Node mapping: users → user; computers → host; processes → process; DNS/IPs → ip; red-team events → alert nodes labeled malicious; sample benign events as benign alert nodes
Labels: red team events are explicitly labeled — use as malicious alert labels; treat each continuous red-team session as a ground-truth incident cluster for evaluation
Important: dataset is very large — implement a configurable --lanl_sample_days arg (default: 5) to load a time slice rather than all 58 days
Loader file: data/loaders/lanl_loader.py


4. CIC-IDS 2017


Download: https://www.unb.ca/cic/datasets/ids-2017.html
Format: CSV with 78 flow-level features and attack labels (DDoS, Brute Force, XSS, SQL Injection, Infiltration, Port Scan, Botnet, Benign)
Node mapping: source IP → ip; destination IP → ip; flows → alert nodes; no user/process nodes available
Labels: attack type column → binary malicious/benign for training; treat each attack-type day (e.g. Wednesday = DoS day) as a ground-truth "incident" for cluster evaluation
Use this dataset primarily for node-classification benchmarking (AUC/F1) to compare the GAT backbone against published GNN-IDS numbers (GraphIDS, Anomal-E, GNN-IDS baselines)
Loader file: data/loaders/cicids_loader.py


Benchmark runner: benchmarks/run_benchmarks.py


Accepts --datasets as a list (e.g. --datasets darpa_tc excytin lanl cicids primary)
For each dataset: runs the full pipeline (graph_builder → trainer → clusterer → evaluator) and saves results to {output_dir}/{dataset_name}/
Produces a unified benchmark_summary.json comparing all datasets on: node-classification AUC, F1; cluster count; mean tactic coherence; mean time span; and (where ground truth exists) cluster-level precision, recall, F1
Add a benchmarks/comparison_table.py that reads benchmark_summary.json and prints a LaTeX-formatted results table ready for paper inclusion


Architecture

Build the following modules, each in its own file:

data/graph_builder.py


Parse the JSONL into a PyG HeteroData object
Node types: alert, host, user, process, ip
Edge types: alert → host (via connects_to), alert → user, alert → process, alert → ip — and their reverses
Alert node features: one-hot encode tactic, technique, numeric severity, time-delta from first alert (normalized). Stack into a float tensor.
Entity node features: learned embeddings (initialize randomly; they will be updated during training)
Alert node labels: binary tensor (1=malicious, 0=benign)
Add a train_mask (80%) and val_mask (20%), stratified by label


models/hgat.py


Implement a 2-layer Heterogeneous Graph Attention Network using torch_geometric.nn.HGTConv or HANConv
Separate attention heads per edge type (do not share weights across alert→host and alert→user)
Output: 64-dim embedding per alert node
Add a linear classification head on top of the alert embeddings for the binary node classification task (malicious/benign)
Use dropout=0.3 between layers


training/trainer.py


Train with BCEWithLogitsLoss on labeled alert nodes using train_mask
Adam optimizer, lr=1e-3, weight_decay=1e-4
Early stopping on val AUC (patience=20)
After training, save: (a) model checkpoint, (b) alert embedding matrix as .npy, (c) per-alert predictions


clustering/incident_clusterer.py


Load the saved embedding matrix
Filter to alerts predicted malicious (threshold=0.5)
Run DBSCAN on the embeddings (metric=cosine, eps=0.3, min_samples=2) — expose eps and min_samples as CLI args
Output: a .jsonl where each line is {"incident_id": int, "alert_ids": [...]} for each discovered cluster
Noise points (DBSCAN label=-1) go into {"incident_id": -1, "alert_ids": [...]} as unclustered


evaluation/evaluator.py


Since no incident ground truth exists, compute proxy metrics:

Per cluster: fraction of alerts sharing at least one MITRE tactic → "tactic coherence"
Per cluster: time span (max_timestamp - min_timestamp) in hours
Per cluster: count of distinct entity types touched
Print a summary table and save as evaluation_report.json



Also compute node-classification metrics: AUC, F1, precision, recall on val_mask


main.py


CLI entry point using argparse:

--data: path to JSONL file
--epochs: default 100
--eps: DBSCAN eps, default 0.3
--min_samples: default 2
--output_dir: where to save all artifacts



Calls graph_builder → trainer → clusterer → evaluator in sequence
Log each stage with Python logging, not print statements


Constraints


PyTorch + PyG only (no DGL, no NetworkX for core logic)
No hardcoded field names — infer schema from data at runtime
Every module must be importable independently (no circular imports)
Include a requirements.txt
Add docstrings to every class and public method