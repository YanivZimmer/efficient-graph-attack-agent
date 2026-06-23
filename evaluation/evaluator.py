"""Evaluation metrics for node classification and discovered incidents."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from data.graph_builder import AlertGraphArtifacts


logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Structured evaluation output."""

    node_classification: dict[str, float]
    cluster_summary: dict[str, Any]
    cluster_metrics: list[dict[str, Any]]
    ground_truth_clustering: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "node_classification": self.node_classification,
            "cluster_summary": self.cluster_summary,
            "cluster_metrics": self.cluster_metrics,
        }
        if self.ground_truth_clustering is not None:
            payload["ground_truth_clustering"] = self.ground_truth_clustering
        return payload


def _cluster_entity_types(records_by_id: dict[str, Any], alert_ids: list[str]) -> int:
    entity_types: set[str] = set()
    for alert_id in alert_ids:
        record = records_by_id[alert_id]
        entity_types.update(record.entities.keys())
    return len(entity_types)


def _cluster_tactic_coherence(records_by_id: dict[str, Any], alert_ids: list[str]) -> float:
    tactics = [records_by_id[alert_id].tactic for alert_id in alert_ids]
    if not tactics:
        return 0.0
    dominant = max(set(tactics), key=tactics.count)
    return sum(1 for tactic in tactics if tactic == dominant) / len(tactics)


def _cluster_time_span_hours(records_by_id: dict[str, Any], alert_ids: list[str]) -> float:
    timestamps = [records_by_id[alert_id].timestamp for alert_id in alert_ids]
    valid = [timestamp for timestamp in timestamps if timestamp is not None]
    if len(valid) < 2:
        return 0.0
    delta = max(valid) - min(valid)
    return delta.total_seconds() / 3600.0


def _best_matching_score(
    predicted_alerts: set[str],
    ground_truth_groups: dict[str, list[str]],
) -> tuple[float, float, float]:
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for gt_alerts in ground_truth_groups.values():
        gt_set = set(gt_alerts)
        if not predicted_alerts:
            continue
        overlap = predicted_alerts & gt_set
        precision = len(overlap) / len(predicted_alerts)
        recall = len(overlap) / len(gt_set) if gt_set else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_precision, best_recall, best_f1


def evaluate_run(
    artifacts: AlertGraphArtifacts,
    *,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    clusters: list[dict[str, object]],
    output_path: Path,
) -> EvaluationReport:
    """Compute node-level and proxy cluster metrics."""
    labels = artifacts.data["alert"].y.detach().cpu().numpy()
    val_mask = artifacts.data["alert"].val_mask.detach().cpu().numpy().astype(bool)
    val_labels = labels[val_mask]
    val_probabilities = probabilities[val_mask]
    val_predictions = predictions[val_mask]

    node_metrics = {
        "auc": float(roc_auc_score(val_labels, val_probabilities)) if len(np.unique(val_labels)) > 1 else float("nan"),
        "f1": float(f1_score(val_labels, val_predictions)),
        "precision": float(precision_score(val_labels, val_predictions, zero_division=0)),
        "recall": float(recall_score(val_labels, val_predictions, zero_division=0)),
    }

    records_by_id = {record.alert_id: record for record in artifacts.alert_records}
    cluster_metrics: list[dict[str, Any]] = []
    coherence_scores: list[float] = []
    time_spans: list[float] = []
    entity_type_counts: list[int] = []

    for cluster in clusters:
        incident_id = int(cluster["incident_id"])
        alert_ids = [str(alert_id) for alert_id in cluster["alert_ids"]]
        if not alert_ids:
            continue
        coherence = _cluster_tactic_coherence(records_by_id, alert_ids)
        span_hours = _cluster_time_span_hours(records_by_id, alert_ids)
        entity_types = _cluster_entity_types(records_by_id, alert_ids)
        coherence_scores.append(coherence)
        time_spans.append(span_hours)
        entity_type_counts.append(entity_types)
        cluster_metrics.append(
            {
                "incident_id": incident_id,
                "alert_count": len(alert_ids),
                "tactic_coherence": coherence,
                "time_span_hours": span_hours,
                "distinct_entity_types": entity_types,
            }
        )

    cluster_summary = {
        "cluster_count": len([cluster for cluster in clusters if int(cluster["incident_id"]) >= 0]),
        "noise_alert_count": sum(
            len(cluster["alert_ids"]) for cluster in clusters if int(cluster["incident_id"]) == -1
        ),
        "mean_tactic_coherence": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
        "mean_time_span_hours": float(np.mean(time_spans)) if time_spans else 0.0,
        "mean_distinct_entity_types": float(np.mean(entity_type_counts)) if entity_type_counts else 0.0,
    }

    ground_truth_metrics = None
    if artifacts.ground_truth_incidents:
        gt_groups = artifacts.ground_truth_incidents
        matched_precisions: list[float] = []
        matched_recalls: list[float] = []
        matched_f1s: list[float] = []
        for cluster in clusters:
            incident_id = int(cluster["incident_id"])
            if incident_id < 0:
                continue
            precision, recall, f1 = _best_matching_score(set(cluster["alert_ids"]), gt_groups)
            matched_precisions.append(precision)
            matched_recalls.append(recall)
            matched_f1s.append(f1)
        ground_truth_metrics = {
            "cluster_precision": float(np.mean(matched_precisions)) if matched_precisions else 0.0,
            "cluster_recall": float(np.mean(matched_recalls)) if matched_recalls else 0.0,
            "cluster_f1": float(np.mean(matched_f1s)) if matched_f1s else 0.0,
        }

    report = EvaluationReport(
        node_classification=node_metrics,
        cluster_summary=cluster_summary,
        cluster_metrics=cluster_metrics,
        ground_truth_clustering=ground_truth_metrics,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    _log_summary(report)
    return report


def _log_summary(report: EvaluationReport) -> None:
    logger.info("Node classification: %s", report.node_classification)
    logger.info("Cluster summary: %s", report.cluster_summary)
    if report.ground_truth_clustering is not None:
        logger.info("Ground-truth clustering: %s", report.ground_truth_clustering)
