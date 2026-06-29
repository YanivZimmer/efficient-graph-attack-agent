"""Validate incident clustering policies on chronological AIT-ADS windows."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from clustering.incident_ablation_clusterer import ClusterAblationStrategy, cluster_with_strategy
from data.graph_builder import AlertGraphArtifacts
from data.loaders.ait_ads_loader import load_ait_ads_graph
from evaluation.evaluator import evaluate_run
try:
    from run_cluster_ablation_experiments import (
        DEFAULT_SOURCES,
        _align_to_artifacts,
        _load_predictions,
        _load_threshold_and_mode,
        _selected_count,
    )
except ModuleNotFoundError:
    from scripts.run_cluster_ablation_experiments import (
        DEFAULT_SOURCES,
        _align_to_artifacts,
        _load_predictions,
        _load_threshold_and_mode,
        _selected_count,
    )


logger = logging.getLogger(__name__)


HOLDOUT_STRATEGIES = [
    ClusterAblationStrategy(
        name="temporal_only_macro_elbow",
        base="temporal",
        split_time_gap_estimator="macro_elbow",
        split_time_gap_min_hours=0.25,
    ),
    ClusterAblationStrategy(
        name="temporal_only_macro_elbow_floor4",
        base="temporal",
        split_time_gap_estimator="macro_elbow",
        split_time_gap_min_hours=4.0,
    ),
    ClusterAblationStrategy(name="temporal_only_4h", base="temporal", split_time_gap_hours=4.0),
    ClusterAblationStrategy(name="temporal_only_6h", base="temporal", split_time_gap_hours=6.0),
    ClusterAblationStrategy(name="temporal_only_12h", base="temporal", split_time_gap_hours=12.0),
    ClusterAblationStrategy(name="temporal_only_24h", base="temporal", split_time_gap_hours=24.0),
    ClusterAblationStrategy(
        name="temporal_only_adaptive_q95",
        base="temporal",
        split_time_gap_quantile=0.95,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chronological holdout validation for AIT-ADS clustering")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/ait_ads"))
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/h1-incident-aware-hgat/results/temporal_holdout_ait_ads"),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(DEFAULT_SOURCES),
        default=sorted(DEFAULT_SOURCES),
    )
    return parser.parse_args()


def _timestamp_key(artifacts: AlertGraphArtifacts, index: int) -> tuple[float, int]:
    timestamp = artifacts.alert_records[index].timestamp
    return (timestamp.timestamp() if timestamp is not None else float("inf"), index)


def _chronological_folds(artifacts: AlertGraphArtifacts, folds: int) -> list[list[int]]:
    ordered = sorted(range(len(artifacts.alert_records)), key=lambda index: _timestamp_key(artifacts, index))
    return [chunk.astype(int).tolist() for chunk in np.array_split(np.array(ordered, dtype=np.int64), folds) if len(chunk)]


def _subset_ground_truth(
    artifacts: AlertGraphArtifacts,
    alert_id_set: set[str],
) -> dict[str, list[str]]:
    subset: dict[str, list[str]] = {}
    for incident_id, alert_ids in artifacts.ground_truth_incidents.items():
        kept = [alert_id for alert_id in alert_ids if alert_id in alert_id_set]
        if len(kept) >= 2:
            subset[incident_id] = kept
    return subset


def _subset_artifacts(artifacts: AlertGraphArtifacts, indices: list[int]) -> AlertGraphArtifacts:
    alert_ids = [artifacts.alert_ids[index] for index in indices]
    records = [artifacts.alert_records[index] for index in indices]
    labels = torch.tensor([record.label for record in records], dtype=torch.float32)
    data = HeteroData()
    data["alert"].x = torch.zeros((len(indices), 1), dtype=torch.float32)
    data["alert"].y = labels
    data["alert"].train_mask = torch.zeros(len(indices), dtype=torch.bool)
    data["alert"].val_mask = torch.ones(len(indices), dtype=torch.bool)
    return AlertGraphArtifacts(
        data=data,
        schema=artifacts.schema,
        alert_ids=alert_ids,
        alert_records=records,
        tactic_vocab=artifacts.tactic_vocab,
        technique_vocab=artifacts.technique_vocab,
        ground_truth_incidents=_subset_ground_truth(artifacts, set(alert_ids)),
    )


def _flatten_result(
    *,
    source: str,
    fold_index: int,
    fold_indices: list[int],
    strategy: ClusterAblationStrategy,
    selected_count: int,
    ground_truth_incident_count: int,
    report,
) -> dict[str, object]:
    gt = report.ground_truth_clustering or {}
    return {
        "source": source,
        "fold": fold_index,
        "strategy": strategy.name,
        "fold_alert_count": len(fold_indices),
        "selected_alert_count": selected_count,
        "ground_truth_incident_count": ground_truth_incident_count,
        "node_auc": report.node_classification["auc"],
        "node_f1": report.node_classification["f1"],
        "cluster_count": report.cluster_summary["cluster_count"],
        "noise_alert_count": report.cluster_summary["noise_alert_count"],
        "mean_time_span_hours": report.cluster_summary["mean_time_span_hours"],
        "cluster_precision": gt.get("cluster_precision"),
        "cluster_recall": gt.get("cluster_recall"),
        "cluster_f1": gt.get("cluster_f1"),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["strategy"]))].append(row)

    aggregates: list[dict[str, object]] = []
    for (source, strategy), group in grouped.items():
        f1s = np.array([float(row.get("cluster_f1") or 0.0) for row in group], dtype=np.float64)
        precisions = np.array([float(row.get("cluster_precision") or 0.0) for row in group], dtype=np.float64)
        recalls = np.array([float(row.get("cluster_recall") or 0.0) for row in group], dtype=np.float64)
        aggregates.append(
            {
                "source": source,
                "strategy": strategy,
                "folds": len(group),
                "mean_cluster_f1": float(np.mean(f1s)),
                "std_cluster_f1": float(np.std(f1s)),
                "min_cluster_f1": float(np.min(f1s)),
                "mean_cluster_precision": float(np.mean(precisions)),
                "mean_cluster_recall": float(np.mean(recalls)),
            }
        )
    aggregates.sort(
        key=lambda row: (
            float(row["mean_cluster_f1"]),
            float(row["mean_cluster_precision"]),
        ),
        reverse=True,
    )
    return aggregates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_ait_ads_graph(args.data_root, max_records=args.max_records)
    folds = _chronological_folds(artifacts, args.folds)
    rows: list[dict[str, object]] = []

    for source in args.sources:
        source_path = DEFAULT_SOURCES[source]
        logger.info("Loading source=%s from %s", source, source_path)
        embeddings = np.load(source_path / "alert_embeddings.npy")
        source_alert_ids, probabilities, predictions = _load_predictions(source_path)
        threshold, selection_mode = _load_threshold_and_mode(source_path)
        embeddings, probabilities, predictions = _align_to_artifacts(
            artifact_alert_ids=artifacts.alert_ids,
            source_alert_ids=source_alert_ids,
            embeddings=embeddings,
            probabilities=probabilities,
            predictions=predictions,
        )

        for fold_index, fold_indices in enumerate(folds):
            fold_dir = args.output_dir / source / f"fold_{fold_index:02d}"
            fold_artifacts = _subset_artifacts(artifacts, fold_indices)
            fold_embeddings = embeddings[fold_indices]
            fold_probabilities = probabilities[fold_indices]
            fold_predictions = predictions[fold_indices]
            selected_count = _selected_count(fold_predictions, fold_probabilities, threshold, selection_mode)

            for strategy in HOLDOUT_STRATEGIES:
                logger.info("source=%s fold=%s strategy=%s", source, fold_index, strategy.name)
                strategy_dir = fold_dir / strategy.name
                clusters = cluster_with_strategy(
                    strategy=strategy,
                    embeddings=fold_embeddings,
                    alert_ids=fold_artifacts.alert_ids,
                    records=fold_artifacts.alert_records,
                    predictions=fold_predictions,
                    probabilities=fold_probabilities,
                    output_path=strategy_dir / "discovered_incidents.jsonl",
                    threshold=threshold,
                    selection_mode=selection_mode,
                )
                report = evaluate_run(
                    fold_artifacts,
                    predictions=fold_predictions,
                    probabilities=fold_probabilities,
                    clusters=clusters,
                    output_path=strategy_dir / "evaluation_report.json",
                )
                row = _flatten_result(
                    source=source,
                    fold_index=fold_index,
                    fold_indices=fold_indices,
                    strategy=strategy,
                    selected_count=selected_count,
                    ground_truth_incident_count=len(fold_artifacts.ground_truth_incidents),
                    report=report,
                )
                (strategy_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(row)

    aggregates = _aggregate(rows)
    summary = {
        "folds": len(folds),
        "strategies": [strategy.name for strategy in HOLDOUT_STRATEGIES],
        "best": aggregates[0] if aggregates else None,
        "aggregates": aggregates,
        "rows": rows,
    }
    (args.output_dir / "temporal_holdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "temporal_holdout_rows.csv", rows)
    _write_csv(args.output_dir / "temporal_holdout_aggregates.csv", aggregates)
    logger.info("Wrote temporal holdout summary to %s", args.output_dir / "temporal_holdout_summary.json")


if __name__ == "__main__":
    main()
