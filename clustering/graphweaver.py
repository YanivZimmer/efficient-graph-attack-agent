"""GraphWeaver-style rule-based alert correlation (entity overlap + time window)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from data.graph_builder import AlertGraphArtifacts, AlertRecord


logger = logging.getLogger(__name__)


@dataclass
class GraphWeaverResult:
    """Outputs from the GraphWeaver rule-based baseline."""

    predictions: np.ndarray
    probabilities: np.ndarray
    embeddings: np.ndarray
    clusters: list[dict[str, object]]


def _entity_keys(record: AlertRecord) -> list[str]:
    keys: list[str] = []
    for entity_type, value in record.entities.items():
        if value:
            keys.append(f"{entity_type}:{value}")
    return keys


def _union_find_cluster(
    records: list[AlertRecord],
    *,
    max_gap_minutes: int,
) -> dict[str, list[str]]:
    """Cluster alerts that share an entity within a sliding time window."""
    if not records:
        return {}

    parent = {record.alert_id: record.alert_id for record in records}

    def find(alert_id: str) -> str:
        while parent[alert_id] != alert_id:
            parent[alert_id] = parent[parent[alert_id]]
            alert_id = parent[alert_id]
        return alert_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    records_by_id = {record.alert_id: record for record in records}
    entity_alerts: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for entity_key in _entity_keys(record):
            entity_alerts[entity_key].append(record.alert_id)

    max_gap_seconds = max_gap_minutes * 60
    for alert_ids in entity_alerts.values():
        ordered = sorted(
            alert_ids,
            key=lambda alert_id: (
                records_by_id[alert_id].timestamp or datetime.min,
                alert_id,
            ),
        )
        for idx, alert_id in enumerate(ordered):
            current = records_by_id[alert_id]
            if current.timestamp is None:
                continue
            previous_idx = idx - 1
            while previous_idx >= 0:
                previous_id = ordered[previous_idx]
                previous = records_by_id[previous_id]
                if previous.timestamp is None:
                    previous_idx -= 1
                    continue
                gap = (current.timestamp - previous.timestamp).total_seconds()
                if gap > max_gap_seconds:
                    break
                union(alert_id, previous_id)
                previous_idx -= 1

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[find(record.alert_id)].append(record.alert_id)

    return {root: sorted(alert_ids) for root, alert_ids in grouped.items()}


def cluster_graphweaver(
    artifacts: AlertGraphArtifacts,
    *,
    malicious_indices: np.ndarray,
    max_gap_minutes: int = 120,
    output_path: Path | None = None,
) -> list[dict[str, object]]:
    """Cluster malicious alerts via shared-entity union-find (GraphWeaver-style)."""
    selected_records = [artifacts.alert_records[index] for index in malicious_indices]
    grouped = _union_find_cluster(selected_records, max_gap_minutes=max_gap_minutes)

    clusters: list[dict[str, object]] = []
    for cluster_index, alert_ids in enumerate(grouped.values()):
        if len(alert_ids) < 2:
            incident_id = -1
        else:
            incident_id = cluster_index
        clusters.append({"incident_id": incident_id, "alert_ids": alert_ids})

    singletons = [cluster for cluster in clusters if int(cluster["incident_id"]) == -1]
    grouped_clusters = [cluster for cluster in clusters if int(cluster["incident_id"]) >= 0]
    clusters = grouped_clusters + singletons

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in clusters:
                handle.write(json.dumps(row) + "\n")

    logger.info(
        "GraphWeaver produced %s incident clusters from %s malicious alerts",
        len([cluster for cluster in clusters if int(cluster["incident_id"]) >= 0]),
        len(selected_records),
    )
    return clusters


def run_graphweaver_baseline(
    artifacts: AlertGraphArtifacts,
    *,
    max_gap_minutes: int = 120,
    triage: str = "oracle_label",
    output_path: Path | None = None,
) -> GraphWeaverResult:
    """Run the rule-based GraphWeaver baseline end-to-end.

    Triage modes:
    - ``oracle_label``: treat ground-truth malicious labels as triage input. This
      matches industry alert-correlation papers that assume alerts are already
      in the analyst queue; node metrics are not comparable to learned models.
    - ``severity``: predict malicious when severity >= dataset median severity.
    """
    labels = artifacts.data["alert"].y.detach().cpu().numpy()
    severities = np.array([record.severity for record in artifacts.alert_records], dtype=np.float32)

    if triage == "oracle_label":
        probabilities = labels.astype(np.float32)
        predictions = labels.astype(int)
    elif triage == "severity":
        threshold = float(np.median(severities)) if len(severities) else 0.5
        probabilities = (severities >= threshold).astype(np.float32)
        predictions = probabilities.astype(int)
    else:
        raise ValueError(f"Unknown GraphWeaver triage mode {triage!r}")

    malicious_indices = np.where((probabilities >= 0.5) | (predictions == 1))[0]
    clusters = cluster_graphweaver(
        artifacts,
        malicious_indices=malicious_indices,
        max_gap_minutes=max_gap_minutes,
        output_path=output_path,
    )

    # Placeholder embeddings for API compatibility with DBSCAN-based methods.
    embeddings = np.zeros((len(artifacts.alert_ids), 1), dtype=np.float32)
    return GraphWeaverResult(
        predictions=predictions,
        probabilities=probabilities,
        embeddings=embeddings,
        clusters=clusters,
    )
