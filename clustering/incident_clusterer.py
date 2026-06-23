"""DBSCAN clustering over malicious alert embeddings."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


logger = logging.getLogger(__name__)


def cluster_incidents(
    *,
    embeddings: np.ndarray,
    alert_ids: list[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
    eps: float = 0.3,
    min_samples: int = 2,
) -> list[dict[str, object]]:
    """Cluster malicious-predicted alerts and write incident groups to JSONL."""
    malicious_indices = np.where((probabilities >= threshold) | (predictions == 1))[0]
    if len(malicious_indices) == 0:
        logger.warning("No malicious-predicted alerts found for clustering")
        clusters = [{"incident_id": -1, "alert_ids": []}]
        _write_jsonl(output_path, clusters)
        return clusters

    selected_embeddings = embeddings[malicious_indices]
    selected_alert_ids = [alert_ids[index] for index in malicious_indices]

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clusterer.fit_predict(selected_embeddings)

    grouped: dict[int, list[str]] = defaultdict(list)
    for alert_id, label in zip(selected_alert_ids, labels, strict=True):
        grouped[int(label)].append(alert_id)

    clusters = [
        {"incident_id": incident_id, "alert_ids": sorted(alert_ids)}
        for incident_id, alert_ids in sorted(grouped.items(), key=lambda item: item[0])
    ]

    _write_jsonl(output_path, clusters)
    logger.info("Wrote %s incident clusters to %s", len(clusters), output_path)
    return clusters


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
