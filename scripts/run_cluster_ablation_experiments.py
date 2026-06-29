"""Run inference-time incident clustering ablations on saved AIT-ADS embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from clustering.incident_ablation_clusterer import (
    ClusterAblationStrategy,
    cluster_with_strategy,
    default_ablation_strategies,
)
from data.loaders.ait_ads_loader import load_ait_ads_graph
from evaluation.evaluator import evaluate_run


logger = logging.getLogger(__name__)


DEFAULT_SOURCES = {
    "temporal_causal_hgat": Path(
        "experiments/h1-incident-aware-hgat/results/all_variants_ait_ads_e6/temporal_causal_hgat"
    ),
    "multiview_hgat_v2": Path(
        "experiments/h1-incident-aware-hgat/results/v2_variants_ait_ads_e6/multiview_hgat_v2"
    ),
    "saved_hgat": Path("outputs/baseline_comparison_full/ait_ads/hgat"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIT-ADS clustering ablations over saved embeddings")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/ait_ads"))
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/h1-incident-aware-hgat/results/cluster_ablation_ait_ads"),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(DEFAULT_SOURCES),
        default=sorted(DEFAULT_SOURCES),
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = json.loads((path / "alert_predictions.json").read_text(encoding="utf-8"))
    alert_ids = [str(row["alert_id"]) for row in rows]
    probabilities = np.array([float(row["probability"]) for row in rows], dtype=np.float32)
    predictions = np.array([int(row["prediction"]) for row in rows], dtype=np.int64)
    return alert_ids, probabilities, predictions


def _load_threshold_and_mode(path: Path) -> tuple[float, str]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return 0.5, "prediction"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = float(summary.get("threshold", 0.5))
    selection_mode = str(summary.get("selection_mode", "probability"))
    return threshold, selection_mode


def _align_to_artifacts(
    *,
    artifact_alert_ids: list[str],
    source_alert_ids: list[str],
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_index = {alert_id: index for index, alert_id in enumerate(source_alert_ids)}
    missing = [alert_id for alert_id in artifact_alert_ids if alert_id not in source_index]
    if missing:
        raise ValueError(f"Source is missing {len(missing)} artifact alerts; first missing={missing[0]!r}")
    order = np.array([source_index[alert_id] for alert_id in artifact_alert_ids], dtype=np.int64)
    return embeddings[order], probabilities[order], predictions[order]


def _flatten_result(
    *,
    source: str,
    strategy: ClusterAblationStrategy,
    threshold: float,
    selection_mode: str,
    selected_count: int,
    report,
) -> dict[str, object]:
    gt = report.ground_truth_clustering or {}
    graph_uses_semantic = strategy.base == "graph" and strategy.relation_policy in {"semantic", "entity_semantic"}
    graph_uses_entity = strategy.base == "graph" and strategy.relation_policy in {"entity", "entity_semantic"}
    uses_semantic_metadata = strategy.split_tactic or graph_uses_semantic
    uses_entity_signal = strategy.split_entity or graph_uses_entity
    uses_time_signal = (
        strategy.split_time_gap_hours is not None
        or strategy.split_time_gap_quantile is not None
        or strategy.split_time_gap_estimator is not None
        or strategy.base in {"graph", "temporal", "bayesian_blocks"}
    )
    return {
        "source": source,
        "strategy": strategy.name,
        "base": strategy.base,
        "uses_time_signal": uses_time_signal,
        "uses_entity_signal": uses_entity_signal,
        "uses_semantic_metadata": uses_semantic_metadata,
        "threshold": threshold,
        "selection_mode": selection_mode,
        "selected_alert_count": selected_count,
        "node_auc": report.node_classification["auc"],
        "node_f1": report.node_classification["f1"],
        "cluster_count": report.cluster_summary["cluster_count"],
        "noise_alert_count": report.cluster_summary["noise_alert_count"],
        "mean_tactic_coherence": report.cluster_summary["mean_tactic_coherence"],
        "mean_time_span_hours": report.cluster_summary["mean_time_span_hours"],
        "cluster_precision": gt.get("cluster_precision"),
        "cluster_recall": gt.get("cluster_recall"),
        "cluster_f1": gt.get("cluster_f1"),
    }


def _selected_count(predictions: np.ndarray, probabilities: np.ndarray, threshold: float, selection_mode: str) -> int:
    if selection_mode == "probability":
        return int((probabilities >= threshold).sum())
    if selection_mode == "prediction":
        return int((predictions == 1).sum())
    return int(((probabilities >= threshold) | (predictions == 1)).sum())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_ait_ads_graph(args.data_root, max_records=args.max_records)
    strategies = default_ablation_strategies()
    all_results: list[dict[str, object]] = []

    for source in args.sources:
        source_path = DEFAULT_SOURCES[source]
        logger.info("Loading source=%s from %s", source, source_path)
        embeddings = np.load(source_path / "alert_embeddings.npy")
        source_alert_ids, probabilities, predictions = _load_predictions(source_path)
        threshold, selection_mode = _load_threshold_and_mode(source_path)
        aligned_embeddings, aligned_probabilities, aligned_predictions = _align_to_artifacts(
            artifact_alert_ids=artifacts.alert_ids,
            source_alert_ids=source_alert_ids,
            embeddings=embeddings,
            probabilities=probabilities,
            predictions=predictions,
        )
        selected_count = _selected_count(aligned_predictions, aligned_probabilities, threshold, selection_mode)

        for strategy in strategies:
            logger.info("source=%s strategy=%s", source, strategy.name)
            strategy_dir = args.output_dir / source / strategy.name
            clusters = cluster_with_strategy(
                strategy=strategy,
                embeddings=aligned_embeddings,
                alert_ids=artifacts.alert_ids,
                records=artifacts.alert_records,
                predictions=aligned_predictions,
                probabilities=aligned_probabilities,
                output_path=strategy_dir / "discovered_incidents.jsonl",
                threshold=threshold,
                selection_mode=selection_mode,
            )
            report = evaluate_run(
                artifacts,
                predictions=aligned_predictions,
                probabilities=aligned_probabilities,
                clusters=clusters,
                output_path=strategy_dir / "evaluation_report.json",
            )
            result = _flatten_result(
                source=source,
                strategy=strategy,
                threshold=threshold,
                selection_mode=selection_mode,
                selected_count=selected_count,
                report=report,
            )
            (strategy_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            all_results.append(result)

    all_results.sort(
        key=lambda row: (
            float(row.get("cluster_f1") or 0.0),
            float(row.get("cluster_precision") or 0.0),
        ),
        reverse=True,
    )
    fair_results = [row for row in all_results if not bool(row.get("uses_semantic_metadata"))]
    summary = {
        "best_overall": all_results[0] if all_results else None,
        "best_without_semantic_metadata": fair_results[0] if fair_results else None,
        "results": all_results,
    }
    (args.output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "ablation_summary.csv", all_results)
    logger.info("Wrote ablation summary to %s", args.output_dir / "ablation_summary.json")


if __name__ == "__main__":
    main()
