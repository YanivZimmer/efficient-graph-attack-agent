"""Run HGAT and all baselines side-by-side across benchmark datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from clustering.graphweaver import run_graphweaver_baseline
from clustering.incident_clusterer import cluster_incidents
from benchmarks.dataset_registry import DATASET_LOADERS, load_dataset
from evaluation.evaluator import evaluate_run
from models.baselines.registry import (
    BASELINE_MODELS,
    COMPARISON_TIERS,
    DEFAULT_ALL_METHODS,
    NON_TRAINABLE_METHODS,
    SUPERVISED_UPPER_BOUND_METHODS,
)
from training.baseline_trainer import train_baseline
from training.supervised_baseline_trainer import train_supervised_baseline


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse baseline comparison CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare HGAT against all baselines")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_LOADERS),
        help="Single dataset to benchmark (use --datasets for multiple)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_LOADERS),
        help="One or more datasets to benchmark",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_ALL_METHODS,
        choices=sorted(BASELINE_MODELS),
        help="Models to compare",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_comparison"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--lanl-sample-days", type=int, default=5)
    parser.add_argument(
        "--ait-ads-max-records",
        type=int,
        default=10_000,
        help="Subsample AIT-ADS to this many alerts (keeps all malicious alerts first)",
    )
    parser.add_argument("--graphweaver-gap-minutes", type=int, default=120)
    parser.add_argument(
        "--graphweaver-triage",
        choices=("oracle_label", "severity"),
        default="oracle_label",
    )
    parser.add_argument(
        "--fail-on-missing-gt",
        action="store_true",
        help="Raise an error instead of skipping supervised baselines when incident GT is missing",
    )
    parser.add_argument(
        "--literature",
        type=Path,
        default=Path("benchmarks/literature_baselines.json"),
        help="Published reference numbers to append as read-only rows",
    )
    return parser.parse_args()


def _result_row(
    *,
    dataset_name: str,
    method: str,
    report,
    training_result=None,
    triage_mode: str | None = None,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> dict:
    row = {
        "dataset": dataset_name,
        "method": method,
        "description": BASELINE_MODELS[method],
        "comparison_tier": COMPARISON_TIERS.get(method, "literature"),
        "source": "skipped" if skipped else "local_run",
        "node_auc": None if skipped else report.node_classification["auc"],
        "node_f1": None if skipped else report.node_classification["f1"],
        "node_precision": None if skipped else report.node_classification["precision"],
        "node_recall": None if skipped else report.node_classification["recall"],
        "cluster_count": None if skipped else report.cluster_summary["cluster_count"],
        "mean_tactic_coherence": None if skipped else report.cluster_summary["mean_tactic_coherence"],
        "mean_time_span_hours": None if skipped else report.cluster_summary["mean_time_span_hours"],
        "cluster_precision": None
        if skipped
        else (report.ground_truth_clustering or {}).get("cluster_precision"),
        "cluster_recall": None if skipped else (report.ground_truth_clustering or {}).get("cluster_recall"),
        "cluster_f1": None if skipped else (report.ground_truth_clustering or {}).get("cluster_f1"),
    }
    if training_result is not None and hasattr(training_result, "best_val_auc"):
        row["best_val_auc"] = training_result.best_val_auc
    if triage_mode is not None:
        row["triage_mode"] = triage_mode
    if skip_reason:
        row["skip_reason"] = skip_reason
    return row


def _run_graphweaver(
    artifacts,
    *,
    dataset_name: str,
    output_dir: Path,
    max_gap_minutes: int,
    triage: str,
) -> dict:
    method_output = output_dir / "graphweaver"
    graphweaver_result = run_graphweaver_baseline(
        artifacts,
        max_gap_minutes=max_gap_minutes,
        triage=triage,
        output_path=method_output / "discovered_incidents.jsonl",
    )
    report = evaluate_run(
        artifacts,
        predictions=graphweaver_result.predictions,
        probabilities=graphweaver_result.probabilities,
        clusters=graphweaver_result.clusters,
        output_path=method_output / "evaluation_report.json",
    )
    return _result_row(
        dataset_name=dataset_name,
        method="graphweaver",
        report=report,
        triage_mode=triage,
    )


def _run_trainable_method(
    method: str,
    artifacts,
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
    return _result_row(
        dataset_name=dataset_name,
        method=method,
        report=report,
        training_result=training_result,
    )


def _run_supervised_method(
    method: str,
    artifacts,
    *,
    dataset_name: str,
    output_dir: Path,
    epochs: int,
    fail_on_missing_gt: bool,
) -> dict:
    if not artifacts.ground_truth_incidents:
        if fail_on_missing_gt:
            raise ValueError(f"{method} requires incident-level ground truth")
        logger.warning("Skipping %s on %s: no incident ground truth", method, dataset_name)
        return _result_row(
            dataset_name=dataset_name,
            method=method,
            report=evaluate_run(
                artifacts,
                predictions=artifacts.data["alert"].y.detach().cpu().numpy().astype(int),
                probabilities=artifacts.data["alert"].y.detach().cpu().numpy().astype(float),
                clusters=[{"incident_id": -1, "alert_ids": []}],
                output_path=output_dir / method / "evaluation_report.json",
            ),
            skipped=True,
            skip_reason="no_incident_ground_truth",
        )

    method_output = output_dir / method
    supervised_result = train_supervised_baseline(
        method,
        artifacts,
        output_dir=method_output,
        epochs=epochs,
    )
    report = evaluate_run(
        artifacts,
        predictions=supervised_result.predictions,
        probabilities=supervised_result.probabilities,
        clusters=supervised_result.clusters,
        output_path=method_output / "evaluation_report.json",
    )
    return _result_row(
        dataset_name=dataset_name,
        method=method,
        report=report,
        training_result=supervised_result,
    )


def _run_method(
    method: str,
    artifacts,
    *,
    dataset_name: str,
    output_dir: Path,
    epochs: int,
    pretrain_epochs: int,
    eps: float,
    min_samples: int,
    graphweaver_gap_minutes: int,
    graphweaver_triage: str,
    fail_on_missing_gt: bool,
) -> dict:
    if method in NON_TRAINABLE_METHODS:
        return _run_graphweaver(
            artifacts,
            dataset_name=dataset_name,
            output_dir=output_dir,
            max_gap_minutes=graphweaver_gap_minutes,
            triage=graphweaver_triage,
        )
    if method in SUPERVISED_UPPER_BOUND_METHODS:
        return _run_supervised_method(
            method,
            artifacts,
            dataset_name=dataset_name,
            output_dir=output_dir,
            epochs=epochs,
            fail_on_missing_gt=fail_on_missing_gt,
        )
    return _run_trainable_method(
        method,
        artifacts,
        dataset_name=dataset_name,
        output_dir=output_dir,
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
        eps=eps,
        min_samples=min_samples,
    )


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
                "comparison_tier": reference.get("comparison_tier", "literature"),
                "source": "literature",
                "node_auc": reference.get("node_auc"),
                "node_f1": reference.get("node_f1"),
                "cluster_f1": reference.get("cluster_f1"),
                "metric_source": reference.get("metric_source"),
            }
        )
    return rows


def render_markdown_table(rows: list[dict]) -> str:
    """Render a side-by-side markdown comparison table."""
    headers = [
        "Dataset",
        "Method",
        "Tier",
        "Source",
        "AUC",
        "F1",
        "Cluster F1",
        "Clusters",
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
                    str(row.get("dataset", "-")),
                    str(row.get("method", "-")),
                    str(row.get("comparison_tier", "-")),
                    str(row.get("source", "-")),
                    _fmt(row.get("node_auc")),
                    _fmt(row.get("node_f1")),
                    _fmt(row.get("cluster_f1")),
                    _fmt(row.get("cluster_count"), 0),
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


def _selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return args.datasets
    if args.dataset:
        return [args.dataset]
    return ["primary"]


def run_dataset_comparison(args: argparse.Namespace, dataset: str) -> list[dict]:
    """Run all requested methods for one dataset."""
    try:
        artifacts = load_dataset(
            dataset,
            args.data_root,
            lanl_sample_days=args.lanl_sample_days,
            ait_ads_max_records=args.ait_ads_max_records,
        )
    except FileNotFoundError as exc:
        logger.warning("Skipping dataset %s: %s", dataset, exc)
        return []
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Skipping dataset %s due to load error: %s", dataset, exc)
        return []

    comparison_rows: list[dict] = []
    for method in args.methods:
        logger.info("Running baseline comparison for method=%s dataset=%s", method, dataset)
        try:
            comparison_rows.append(
                _run_method(
                    method,
                    artifacts,
                    dataset_name=dataset,
                    output_dir=args.output_dir / dataset,
                    epochs=args.epochs,
                    pretrain_epochs=args.pretrain_epochs,
                    eps=args.eps,
                    min_samples=args.min_samples,
                    graphweaver_gap_minutes=args.graphweaver_gap_minutes,
                    graphweaver_triage=args.graphweaver_triage,
                    fail_on_missing_gt=args.fail_on_missing_gt,
                )
            )
        except ValueError as exc:
            logger.warning("Skipping method=%s dataset=%s: %s", method, dataset, exc)
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "description": BASELINE_MODELS[method],
                    "comparison_tier": COMPARISON_TIERS.get(method, "literature"),
                    "source": "skipped",
                    "skip_reason": str(exc),
                }
            )

    json_path = args.output_dir / f"{dataset}_baseline_comparison.json"
    md_path = args.output_dir / f"{dataset}_baseline_comparison.md"
    json_path.write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_table(comparison_rows), encoding="utf-8")
    logger.info("Wrote comparison JSON to %s", json_path)
    return comparison_rows


def main() -> None:
    """Run all requested methods and datasets; write combined comparison tables."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for dataset in _selected_datasets(args):
        all_rows.extend(run_dataset_comparison(args, dataset))

    literature_rows = _load_literature_rows(args.literature)
    combined = all_rows + literature_rows
    combined_json = args.output_dir / "all_datasets_baseline_comparison.json"
    combined_md = args.output_dir / "all_datasets_baseline_comparison.md"
    combined_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    combined_md.write_text(render_markdown_table(combined), encoding="utf-8")
    logger.info("Wrote combined comparison to %s", combined_json)


if __name__ == "__main__":
    main()
