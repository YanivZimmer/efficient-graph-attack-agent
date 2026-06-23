"""Run HGAT and literature GNN baselines side-by-side."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from clustering.incident_clusterer import cluster_incidents
from data.graph_builder import AlertGraphArtifacts
from data.loaders.cicids_loader import load_cicids_graph
from data.loaders.primary_loader import load_primary_graph
from evaluation.evaluator import evaluate_run
from models.baselines.registry import BASELINE_MODELS
from training.baseline_trainer import train_baseline


logger = logging.getLogger(__name__)

DEFAULT_METHODS = ["hgat", "gnn_ids", "graph_ids", "anomal_e"]
DATASET_LOADERS = {
    "primary": load_primary_graph,
    "cicids": load_cicids_graph,
}


def parse_args() -> argparse.Namespace:
    """Parse baseline comparison CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare HGAT against literature GNN-IDS baselines")
    parser.add_argument("--dataset", choices=sorted(DATASET_LOADERS), default="primary")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(BASELINE_MODELS),
        help="Models to compare",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_comparison"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--literature",
        type=Path,
        default=Path("benchmarks/literature_baselines.json"),
        help="Published reference numbers to append as read-only rows",
    )
    return parser.parse_args()


def _resolve_dataset_path(dataset: str, data_root: Path) -> Path:
    if dataset == "primary":
        return data_root / "0b1972fe_backup" / "training_data_rich_examples.jsonl"
    return data_root / dataset


def _load_dataset(dataset: str, data_root: Path) -> AlertGraphArtifacts:
    path = _resolve_dataset_path(dataset, data_root)
    return DATASET_LOADERS[dataset](path)


def _run_method(
    method: str,
    artifacts: AlertGraphArtifacts,
    *,
    dataset_name: str,
    output_dir: Path,
    epochs: int,
    pretrain_epochs: int,
    eps: float,
    min_samples: int,
) -> dict:
    method_output = output_dir / method
    training_result = train_baseline(
        method,
        artifacts,
        output_dir=method_output,
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
    )
    clusters = cluster_incidents(
        embeddings=training_result.embeddings,
        alert_ids=artifacts.alert_ids,
        predictions=training_result.predictions,
        probabilities=training_result.probabilities,
        output_path=method_output / "discovered_incidents.jsonl",
        eps=eps,
        min_samples=min_samples,
    )
    report = evaluate_run(
        artifacts,
        predictions=training_result.predictions,
        probabilities=training_result.probabilities,
        clusters=clusters,
        output_path=method_output / "evaluation_report.json",
    )
    return {
        "dataset": dataset_name,
        "method": method,
        "description": BASELINE_MODELS[method],
        "source": "local_run",
        "node_auc": report.node_classification["auc"],
        "node_f1": report.node_classification["f1"],
        "node_precision": report.node_classification["precision"],
        "node_recall": report.node_classification["recall"],
        "best_val_auc": training_result.best_val_auc,
        "cluster_count": report.cluster_summary["cluster_count"],
        "mean_tactic_coherence": report.cluster_summary["mean_tactic_coherence"],
        "mean_time_span_hours": report.cluster_summary["mean_time_span_hours"],
        "cluster_precision": (report.ground_truth_clustering or {}).get("cluster_precision"),
        "cluster_recall": (report.ground_truth_clustering or {}).get("cluster_recall"),
        "cluster_f1": (report.ground_truth_clustering or {}).get("cluster_f1"),
    }


def _load_literature_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for reference in payload.get("references", []):
        rows.append(
            {
                "dataset": payload.get("dataset", "literature"),
                "method": reference["method"],
                "description": reference.get("citation"),
                "source": "literature",
                "node_auc": reference.get("node_auc"),
                "node_f1": reference.get("node_f1"),
                "metric_source": reference.get("metric_source"),
            }
        )
    return rows


def render_markdown_table(rows: list[dict]) -> str:
    """Render a side-by-side markdown comparison table."""
    headers = [
        "Method",
        "Source",
        "AUC",
        "F1",
        "Precision",
        "Recall",
        "Clusters",
        "Tactic coherence",
        "Cluster F1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "-")),
                    str(row.get("source", "-")),
                    _fmt(row.get("node_auc")),
                    _fmt(row.get("node_f1")),
                    _fmt(row.get("node_precision")),
                    _fmt(row.get("node_recall")),
                    _fmt(row.get("cluster_count"), 0),
                    _fmt(row.get("mean_tactic_coherence")),
                    _fmt(row.get("cluster_f1")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "-"
    if digits == 0:
        return str(int(round(numeric)))
    return f"{numeric:.{digits}f}"


def main() -> None:
    """Run all requested methods and write JSON + markdown comparison tables."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_dataset(args.dataset, args.data_root)
    comparison_rows = []
    for method in args.methods:
        logger.info("Running baseline comparison for method=%s dataset=%s", method, args.dataset)
        comparison_rows.append(
            _run_method(
                method,
                artifacts,
                dataset_name=args.dataset,
                output_dir=args.output_dir / args.dataset,
                epochs=args.epochs,
                pretrain_epochs=args.pretrain_epochs,
                eps=args.eps,
                min_samples=args.min_samples,
            )
        )

    literature_rows = _load_literature_rows(args.literature)
    all_rows = comparison_rows + literature_rows

    json_path = args.output_dir / f"{args.dataset}_baseline_comparison.json"
    md_path = args.output_dir / f"{args.dataset}_baseline_comparison.md"
    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_table(all_rows), encoding="utf-8")
    logger.info("Wrote comparison JSON to %s", json_path)
    logger.info("Wrote comparison markdown to %s", md_path)


if __name__ == "__main__":
    main()
