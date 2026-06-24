# efficient-graph-attack-agent

Lightweight implementation of the E-ACS specification: async alert ingestion, graph-sketch filtering, two-hop graph correlation, and a small agentic investigation flow.

See [RESULTS.md](RESULTS.md) for benchmark methodology, SecRL raw-log evaluation results, and interpretation.

## What is implemented

- Pydantic alert/entity schemas.
- Abstract ports for `LLMProvider`, `GraphStore`, `AlertStream`, and `LogStore`.
- Count-min sketch based `GraphSketchingFilter` with the required `> 10,000` baseline suppression rule.
- Async `StreamProcessor` that stores only interesting alerts.
- In-memory graph repository plus optional Neo4j repository adapter.
- OpenAI Responses API adapter using `httpx`.
- Discovery, validation, summarization, synthesis, and lazy hydration agents.
- Post-filter incident discovery that clusters stored alerts by shared entities and time windows, scores them against known attack graph topology patterns, then separates known-overlap clusters from candidate-new incidents.
- OpenTelemetry span hooks with a no-op fallback when telemetry is not installed.
- Weakly-supervised GNN incident discovery pipeline (`main.py`) with heterogeneous GAT training, DBSCAN clustering, and benchmark runners under `benchmarks/`.

## GNN incident discovery

Install and run with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group gnn
uv run gnn-incident \
  --data datasets/0b1972fe_backup/training_data_rich_examples.jsonl \
  --output-dir outputs/gnn_incident
```

Benchmark all configured datasets:

```bash
uv run gnn-benchmark --datasets primary --output-dir outputs/benchmarks
uv run gnn-benchmark-table --summary outputs/benchmarks/benchmark_summary.json
```

Compare HGAT against baselines on the same split. Methods are tiered (see `docs/GNN_research_summary.md`):

- **GraphWeaver** — rule-based entity-overlap lower bound (no ML)
- **HGAT** — our weakly-supervised method
- **GNN-IDS / GraphIDS / Anomal-E** — flow-level GNN methods reimplemented on our alert graph

```bash
uv run gnn-baseline-compare \
  --dataset primary \
  --methods graphweaver hgat gnn_ids graph_ids anomal_e \
  --epochs 50 \
  --output-dir outputs/baseline_comparison

# Alert-domain benchmark (after downloading AIT-ADS to datasets/ait_ads/)
uv run gnn-baseline-compare \
  --dataset ait_ads \
  --methods graphweaver hgat \
  --output-dir outputs/baseline_comparison

uv run gnn-benchmark-table \
  --summary outputs/baseline_comparison/primary_baseline_comparison.json
```

This writes both `primary_baseline_comparison.json` and a markdown table with a `comparison_tier` column. Rows marked `literature` are published reference numbers (GRAIN, Eckhoff GMN, flow-level GNN papers) — not reproduced here and not head-to-head claims.

One-shot without syncing first (installs GNN deps into the project env):

```bash
uv run --group gnn gnn-incident --epochs 3
```

Alternative without uv:

```bash
pip install -r requirements-gnn.txt
PYTHONPATH=. python3 main.py \
  --data datasets/0b1972fe_backup/training_data_rich_examples.jsonl \
  --output-dir outputs/gnn_incident
```

This pipeline is isolated from the existing `src/eacs/` correlation agents and does not modify them.

## Run locally

```bash
PYTHONPATH=src python3 -m eacs.demo
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

The demo processes 10,000 mock alerts and stores the intentionally interesting 10 percent.

## ExCyTIn-Bench evaluation

The project includes a lightweight evaluator for the Hugging Face `anandmudgerikar/excytin-bench` QA metadata:

```bash
python3 scripts/evaluate_excytin_bench.py --split test
```

By default, the script writes reports under `reports/excytin_bench/`, runs metadata/path evaluation from Hugging Face, and runs the deterministic local SecRL QA baseline when `~/Code/Datasets/SecRL/secgym/questions/o1/<split>` exists. To run only a quick metadata smoke test:

```bash
python3 scripts/evaluate_excytin_bench.py --split test --limit 25 --skip-qa
```

The lower-level module command is also available:

```bash
PYTHONPATH=src python3 -m eacs.excytin --split test --output reports/excytin_eval_test.json
```

For a quick smoke run:

```bash
PYTHONPATH=src python3 -m eacs.excytin --split test --limit 25
```

This evaluator does not download the raw-log archive. It measures the current graph-correlation behavior on benchmark question metadata, including shortest-path alert recall and end-alert recall.

Run a deterministic QA baseline over local SecRL question files:

```bash
PYTHONPATH=src python3 -m eacs.excytin qa \
  --secrl-root ~/Code/Datasets/SecRL \
  --split test \
  --question-set o1 \
  --context-source eacs_retrieved \
  --answer-mode extractive \
  --output reports/excytin_qa_eacs_retrieved_test.json
```

Evaluate GIDS variants on raw ExCyTIn/SecRL `SecurityAlert` rows without question answering:

```bash
python3 scripts/evaluate_excytin_raw_gids.py --scope full
python3 scripts/evaluate_excytin_raw_gids.py --scope incidents --hide-severity
```

## SecRL raw-alert evaluation

Clone SecRL outside this repo, then download the raw anonymized logs into SecRL's expected database path:

```bash
git clone --depth 1 https://github.com/microsoft/SecRL.git ~/Code/Datasets/SecRL
PYTHONPATH=src python3 -m eacs.secrl download --secrl-root ~/Code/Datasets/SecRL
```

Evaluate whether E-ACS identifies incident alerts from SecRL `SecurityAlert` rows:

```bash
PYTHONPATH=src python3 -m eacs.secrl evaluate-alerts \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope incidents \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --output reports/secrl_alert_detection_incidents.json
```

The evaluator reads SecRL CSVs directly, using SecRL `qagen/graph_files/incident_*.graphml` as benchmark incident ground truth and `SecurityAlert.SystemAlertId` as candidate alert IDs. It reports alert precision/recall/F1 plus incident-level recall. To evaluate against raw `SecurityIncident.AlertIds` instead, pass `--ground-truth security-incidents`.

The downloaded archive is about 1.5 GiB and extracts to about 20 GiB in `~/Code/Datasets/SecRL/secgym/database/data_anonymized`.

Explain false positives and missed incident alerts:

```bash
PYTHONPATH=src python3 -m eacs.secrl analyze-errors \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope full \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --output-json reports/secrl_error_analysis_full.json \
  --output-md reports/secrl_error_analysis_full.md
```

Cluster stored alerts into known-overlap and candidate-new incidents, then verify discovered clusters against ground-truth incident alert IDs:

```bash
PYTHONPATH=src python3 -m eacs.secrl discover-incidents \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope full \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --output-json reports/secrl_incident_discovery_full.json \
  --output-md reports/secrl_incident_discovery_full.md
```

Run the stricter discovery profile:

```bash
PYTHONPATH=src python3 -m eacs.secrl discover-incidents \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope full \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --profile refined \
  --output-json reports/secrl_incident_discovery_full_refined.json \
  --output-md reports/secrl_incident_discovery_full_refined.md
```

Run discovery ablations:

```bash
PYTHONPATH=src python3 -m eacs.secrl ablate-discovery \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope full \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --output-json reports/secrl_discovery_ablation_full.json \
  --output-md reports/secrl_discovery_ablation_full.md
```

Compare E-ACS against simple raw-log discovery baselines:

```bash
PYTHONPATH=src python3 -m eacs.secrl compare-baselines \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope full \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --output-json reports/secrl_discovery_baselines_full.json \
  --output-md reports/secrl_discovery_baselines_full.md
```

Audit whether ground-truth and raw incident labels affect discovery generation:

```bash
PYTHONPATH=src python3 -m eacs.secrl audit-leakage \
  --data-root ~/Code/Datasets/SecRL/secgym/database/data_anonymized \
  --scope full \
  --ground-truth incident-graphs \
  --secrl-root ~/Code/Datasets/SecRL \
  --profile refined \
  --output-json reports/secrl_leakage_audit_full_refined.json \
  --output-md reports/secrl_leakage_audit_full_refined.md
```
