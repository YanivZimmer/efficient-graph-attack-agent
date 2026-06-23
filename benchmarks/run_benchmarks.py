"""Run the GNN incident discovery pipeline across benchmark datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

from clustering.incident_clusterer import cluster_incidents
from data.graph_builder import AlertGraphArtifacts
from data.loaders.cicids_loader import load_cicids_graph
from data.loaders.darpa_tc_loader import load_darpa_tc_graph
from data.loaders.excytin_loader import load_excytin_graph
from data.loaders.lanl_loader import load_lanl_graph
from data.loaders.primary_loader import load_primary_graph
from evaluation.evaluator import evaluate_run
from training.trainer import train_model


logger = logging.getLogger(__name__)

DATASET_LOADERS: dict[str, Callable[..., AlertGraphArtifacts]] = {
    "primary": load_primary_graph,
    "darpa_tc": load_darpa_tc_graph,
    "excytin": load_excytin_graph,
    "lanl": load_lanl_graph,
    "cicids": load_cicids_graph,
}


def parse_args() -> argparse.Namespace:
    """Parse benchmark runner arguments."""
    parser = argparse.ArgumentParser(description="Benchmark GNN incident discovery across datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["primary"],
        choices=sorted(DATASET_LOADERS),
        help="Datasets to benchmark",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets"), help="Root directory for dataset files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"), help="Benchmark output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--lanl-sample-days", type=int, default=5)
    return parser.parse_args()


def _resolve_dataset_path(dataset: str, data_root: Path) -> Path:
    if dataset == "primary":
        return data_root / "0b1972fe_backup" / "training_data_rich_examples.jsonl"
    return data_root / dataset


def _load_dataset(dataset: str, data_root: Path, lanl_sample_days: int) -> AlertGraphArtifacts:
    path = _resolve_dataset_path(dataset, data_root)
    if dataset == "lanl":
        return load_lanl_graph(path, sample_days=lanl_sample_days)
    return DATASET_LOADERS[dataset](path)


def run_benchmark(
    dataset: str,
    *,
    data_root: Path,
    output_dir: Path,
    epochs: int,
    eps: float,
    min_samples: int,
    lanl_sample_days: int,
) -> dict:
    """Run the full pipeline for one dataset."""
    logger.info("Running benchmark for dataset=%s", dataset)
    artifacts = _load_dataset(dataset, data_root, lanl_sample_days)
    dataset_output = output_dir / dataset
    training_result = train_model(artifacts, output_dir=dataset_output, epochs=epochs)
    clusters = cluster_incidents(
        embeddings=training_result.embeddings,
        alert_ids=artifacts.alert_ids,
        predictions=training_result.predictions,
        probabilities=training_result.probabilities,
        output_path=dataset_output / "discovered_incidents.jsonl",
        eps=eps,
        min_samples=min_samples,
    )
    report = evaluate_run(
        artifacts,
        predictions=training_result.predictions,
        probabilities=training_result.probabilities,
        clusters=clusters,
        output_path=dataset_output / "evaluation_report.json",
    )
    summary = {
        "dataset": dataset,
        "node_auc": report.node_classification["auc"],
        "node_f1": report.node_classification["f1"],
        "cluster_count": report.cluster_summary["cluster_count"],
        "mean_tactic_coherence": report.cluster_summary["mean_tactic_coherence"],
        "mean_time_span_hours": report.cluster_summary["mean_time_span_hours"],
    }
    if report.ground_truth_clustering is not None:
        summary.update(report.ground_truth_clustering)
    return summary


def main() -> None:
    """Run all requested dataset benchmarks and write a summary file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset in args.datasets:
        try:
            summaries.append(
                run_benchmark(
                    dataset,
                    data_root=args.data_root,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    eps=args.eps,
                    min_samples=args.min_samples,
                    lanl_sample_days=args.lanl_sample_days,
                )
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping dataset %s: %s", dataset, exc)

    summary_path = args.output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    logger.info("Wrote benchmark summary to %s", summary_path)


if __name__ == "__main__":
    main()
