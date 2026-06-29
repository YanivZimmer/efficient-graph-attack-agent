"""Run all research HGAT variants on AIT-ADS and compare against saved baselines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data.loaders.ait_ads_loader import load_ait_ads_graph
from training.research_variant_trainer import augment_time_harmonics, train_research_variant


logger = logging.getLogger(__name__)


VARIANTS = [
    "baseline_hgat",
    "weak_pair_hgat",
    "temporal_causal_hgat",
    "temporal_causal_hgat_v2",
    "differentiable_cluster_hgat",
    "prototype_memory_hgat",
    "multiview_hgat",
    "multiview_hgat_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGAT research variants on AIT-ADS")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/ait_ads"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/h1-incident-aware-hgat/results/all_variants_ait_ads"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
    )
    return parser.parse_args()


def _load_variant_artifacts(data_root: Path, *, variant: str, max_records: int):
    include_alert_alert_edges = variant in {
        "temporal_causal_hgat",
        "temporal_causal_hgat_v2",
        "differentiable_cluster_hgat",
        "prototype_memory_hgat",
    }
    artifacts = load_ait_ads_graph(
        data_root,
        max_records=max_records,
        include_alert_alert_edges=include_alert_alert_edges,
        alert_link_hours=6.0,
        max_alert_neighbors_per_relation=8,
    )
    if variant in {
        "temporal_causal_hgat",
        "temporal_causal_hgat_v2",
        "differentiable_cluster_hgat",
        "prototype_memory_hgat",
        "multiview_hgat",
        "multiview_hgat_v2",
    }:
        augment_time_harmonics(artifacts, num_frequencies=2)
    return artifacts


def _load_saved_baselines() -> list[dict[str, object]]:
    path = Path("outputs/baseline_comparison_full/ait_ads")
    saved = []
    for method in ("graphweaver", "hgat", "gnn_ids", "graph_ids", "anomal_e", "grain"):
        report_path = path / method / "evaluation_report.json"
        if not report_path.exists():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        saved.append(
            {
                "variant": method,
                "source": "saved_repo_baseline",
                "node_auc": payload["node_classification"]["auc"],
                "node_f1": payload["node_classification"]["f1"],
                "cluster_precision": (payload.get("ground_truth_clustering") or {}).get("cluster_precision"),
                "cluster_recall": (payload.get("ground_truth_clustering") or {}).get("cluster_recall"),
                "cluster_f1": (payload.get("ground_truth_clustering") or {}).get("cluster_f1"),
            }
        )
    calibrated_baseline_path = Path("experiments/h1-incident-aware-hgat/results/ait_ads_calibrated_e6/baseline/summary.json")
    if calibrated_baseline_path.exists():
        payload = json.loads(calibrated_baseline_path.read_text(encoding="utf-8"))
        saved.append(
            {
                "variant": "baseline_hgat_calibrated",
                "source": "local_calibrated_baseline",
                "node_auc": payload["node_classification"]["auc"],
                "node_f1": payload["node_classification"]["f1"],
                "cluster_precision": payload["ground_truth_clustering"]["cluster_precision"],
                "cluster_recall": payload["ground_truth_clustering"]["cluster_recall"],
                "cluster_f1": payload["ground_truth_clustering"]["cluster_f1"],
            }
        )
    return saved


def _flatten_variant_summary(summary: dict[str, object]) -> dict[str, object]:
    node = summary["node_classification"]
    gt = summary.get("ground_truth_clustering") or {}
    return {
        "variant": summary["variant"],
        "source": "research_variant",
        "node_auc": node["auc"],
        "node_f1": node["f1"],
        "cluster_precision": gt.get("cluster_precision"),
        "cluster_recall": gt.get("cluster_recall"),
        "cluster_f1": gt.get("cluster_f1"),
        "selected_alert_count": summary.get("selected_alert_count"),
        "threshold": summary.get("threshold"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_summaries: list[dict[str, object]] = []
    for variant in args.variants:
        logger.info("Preparing artifacts for variant=%s", variant)
        artifacts = _load_variant_artifacts(args.data_root, variant=variant, max_records=args.max_records)
        result = train_research_variant(
            variant,
            artifacts,
            output_dir=args.output_dir / variant,
            epochs=args.epochs,
            threshold_mode="tuned",
            selection_mode="probability",
        )
        variant_summaries.append(_flatten_variant_summary(result.summary))

    comparisons = variant_summaries + _load_saved_baselines()
    comparisons.sort(key=lambda row: float(row.get("cluster_f1") or 0.0), reverse=True)
    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps({"results": comparisons}, indent=2), encoding="utf-8")
    logger.info("Wrote comparison summary to %s", summary_path)


if __name__ == "__main__":
    main()
