"""Run the first AIT-ADS incident-aware HGAT experiment."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from clustering.incident_clusterer import cluster_incidents
from data.loaders.ait_ads_loader import load_ait_ads_graph
from evaluation.evaluator import evaluate_run
from training.trainer import train_model


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantConfig:
    name: str
    include_alert_alert_edges: bool
    use_incident_pair_loss: bool


VARIANTS = {
    "baseline": VariantConfig(
        name="baseline",
        include_alert_alert_edges=False,
        use_incident_pair_loss=False,
    ),
    "incident_v1": VariantConfig(
        name="incident_v1",
        include_alert_alert_edges=True,
        use_incident_pair_loss=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incident-aware HGAT on AIT-ADS")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/ait_ads"),
        help="Path to AIT-ADS dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/h1-incident-aware-hgat/results/ait_ads_first_test"),
        help="Output directory for experiment artifacts",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["baseline", "incident_v1"],
        help="One or more variants to run",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs per variant")
    parser.add_argument("--max-records", type=int, default=10_000, help="Maximum AIT-ADS records to load")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN epsilon")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Malicious threshold before clustering")
    parser.add_argument(
        "--selection-mode",
        choices=("or", "probability", "prediction"),
        default="probability",
        help="How to select alerts before clustering",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("fixed", "tuned"),
        default="tuned",
        help="Use the provided threshold directly or tune it on the validation split",
    )
    parser.add_argument("--alert-link-hours", type=float, default=6.0, help="Temporal window for alert-alert edges")
    parser.add_argument(
        "--max-alert-neighbors-per-relation",
        type=int,
        default=8,
        help="Maximum local alert neighbors per relation",
    )
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--lambda-pos", type=float, default=0.5)
    parser.add_argument("--lambda-neg", type=float, default=0.25)
    parser.add_argument("--tau-pos-hours", type=float, default=6.0)
    parser.add_argument("--tau-neg-hours", type=float, default=24.0)
    parser.add_argument("--margin-neg", type=float, default=0.2)
    parser.add_argument("--max-pos-pairs-per-anchor", type=int, default=8)
    parser.add_argument("--max-neg-pairs-per-anchor", type=int, default=16)
    return parser.parse_args()


def _candidate_thresholds(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(probabilities)
    if len(unique) <= 256:
        return unique
    quantiles = np.linspace(0.0, 1.0, 256)
    return np.unique(np.quantile(unique, quantiles))


def _tune_threshold(artifacts, probabilities: np.ndarray) -> float:
    labels = artifacts.data["alert"].y.detach().cpu().numpy().astype(int)
    val_mask = artifacts.data["alert"].val_mask.detach().cpu().numpy().astype(bool)
    val_probs = probabilities[val_mask]
    val_labels = labels[val_mask]
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in _candidate_thresholds(val_probs):
        predictions = (val_probs >= float(threshold)).astype(int)
        score = f1_score(val_labels, predictions, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def _node_metrics(artifacts, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = artifacts.data["alert"].y.detach().cpu().numpy().astype(int)
    val_mask = artifacts.data["alert"].val_mask.detach().cpu().numpy().astype(bool)
    val_labels = labels[val_mask]
    val_probs = probabilities[val_mask]
    val_predictions = predictions[val_mask]
    return {
        "auc": float(roc_auc_score(val_labels, val_probs)) if len(np.unique(val_labels)) > 1 else float("nan"),
        "f1": float(f1_score(val_labels, val_predictions, zero_division=0)),
        "precision": float(precision_score(val_labels, val_predictions, zero_division=0)),
        "recall": float(recall_score(val_labels, val_predictions, zero_division=0)),
    }


def run_variant(config: VariantConfig, args: argparse.Namespace) -> dict[str, object]:
    variant_dir = args.output_dir / config.name
    logger.info("Loading AIT-ADS for variant=%s", config.name)
    artifacts = load_ait_ads_graph(
        args.data_root,
        max_records=args.max_records,
        include_alert_alert_edges=config.include_alert_alert_edges,
        alert_link_hours=args.alert_link_hours,
        max_alert_neighbors_per_relation=args.max_alert_neighbors_per_relation,
    )

    logger.info("Training variant=%s", config.name)
    training_result = train_model(
        artifacts,
        output_dir=variant_dir,
        epochs=args.epochs,
        projection_dim=args.projection_dim,
        use_incident_pair_loss=config.use_incident_pair_loss,
        lambda_pos=args.lambda_pos,
        lambda_neg=args.lambda_neg,
        tau_pos_hours=args.tau_pos_hours,
        tau_neg_hours=args.tau_neg_hours,
        margin_neg=args.margin_neg,
        max_pos_pairs_per_anchor=args.max_pos_pairs_per_anchor,
        max_neg_pairs_per_anchor=args.max_neg_pairs_per_anchor,
    )

    threshold = args.threshold
    if args.threshold_mode == "tuned":
        threshold = _tune_threshold(artifacts, training_result.probabilities)
        logger.info("Tuned threshold for variant=%s to %.6f", config.name, threshold)

    thresholded_predictions = _threshold_predictions(training_result.probabilities, threshold)
    logger.info("Clustering variant=%s", config.name)
    clusters = cluster_incidents(
        embeddings=training_result.embeddings,
        alert_ids=artifacts.alert_ids,
        predictions=thresholded_predictions,
        probabilities=training_result.probabilities,
        output_path=variant_dir / "discovered_incidents.jsonl",
        threshold=threshold,
        eps=args.eps,
        min_samples=args.min_samples,
        selection_mode=args.selection_mode,
    )
    report = evaluate_run(
        artifacts,
        predictions=thresholded_predictions,
        probabilities=training_result.probabilities,
        clusters=clusters,
        output_path=variant_dir / "evaluation_report.json",
    )
    summary = {
        "variant": config.name,
        "config": asdict(config),
        "selection_mode": args.selection_mode,
        "threshold_mode": args.threshold_mode,
        "threshold": threshold,
        "selected_alert_count": int(thresholded_predictions.sum()) if args.selection_mode != "prediction" else int((thresholded_predictions == 1).sum()),
        "node_classification": _node_metrics(artifacts, training_result.probabilities, thresholded_predictions),
        "cluster_summary": report.cluster_summary,
        "ground_truth_clustering": report.ground_truth_clustering,
        "best_val_auc": training_result.best_val_auc,
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [run_variant(VARIANTS[variant_name], args) for variant_name in args.variants]
    combined = {"variants": summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    logger.info("Wrote experiment summary to %s", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
