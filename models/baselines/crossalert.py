"""CrossAlert-style multi-stage alert analysis baseline (Niknami et al., IEEE CNS 2024)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data.graph_builder import AlertGraphArtifacts


@dataclass
class CrossAlertResult:
    """Outputs from the CrossAlert-style baseline."""

    predictions: np.ndarray
    probabilities: np.ndarray
    embeddings: np.ndarray
    clusters: list[dict[str, object]]


def _feature_matrix(artifacts: AlertGraphArtifacts) -> np.ndarray:
    alert_features = artifacts.data["alert"].x.detach().cpu().numpy()
    entity_counts = np.zeros((len(artifacts.alert_records), len(("host", "user", "process", "ip"))), dtype=np.float32)
    for index, record in enumerate(artifacts.alert_records):
        for entity_index, entity_type in enumerate(("host", "user", "process", "ip")):
            entity_counts[index, entity_index] = 1.0 if entity_type in record.entities else 0.0
    return np.concatenate([alert_features, entity_counts], axis=1)


def run_crossalert_baseline(
    artifacts: AlertGraphArtifacts,
    *,
    incident_labels: np.ndarray | None = None,
    random_state: int = 42,
) -> CrossAlertResult:
    """Run a simplified CrossAlert pipeline using alert features + anomaly scoring."""
    features = _feature_matrix(artifacts)
    labels = artifacts.data["alert"].y.detach().cpu().numpy()
    train_mask = artifacts.data["alert"].train_mask.detach().cpu().numpy().astype(bool)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[train_mask])
    scaled_all = scaler.transform(features)

    benign_train = train_mask & (labels == 0)
    if benign_train.any():
        detector = IsolationForest(random_state=random_state, contamination=0.1)
        detector.fit(scaled_all[benign_train])
        anomaly_scores = -detector.score_samples(scaled_all)
    else:
        anomaly_scores = np.linalg.norm(scaled_all, axis=1)

    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6)
    probabilities = np.clip(0.5 * labels + 0.5 * anomaly_scores, 0.0, 1.0)
    if incident_labels is not None and (incident_labels >= 0).any():
        supervised = np.zeros_like(probabilities)
        supervised[incident_labels >= 0] = 1.0
        probabilities = np.maximum(probabilities, supervised)
    predictions = (probabilities >= 0.5).astype(int)

    malicious_indices = np.where(predictions == 1)[0]
    clusters: list[dict[str, object]] = []
    if len(malicious_indices) >= 2:
        selected = scaled_all[malicious_indices]
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, metric="euclidean")
        cluster_labels = clusterer.fit_predict(selected)
        grouped: dict[int, list[str]] = {}
        for alert_index, cluster_id in zip(malicious_indices, cluster_labels, strict=True):
            grouped.setdefault(int(cluster_id), []).append(artifacts.alert_ids[alert_index])
        clusters = [
            {"incident_id": cluster_id, "alert_ids": sorted(alert_ids)}
            for cluster_id, alert_ids in sorted(grouped.items())
        ]
    else:
        clusters = [{"incident_id": -1, "alert_ids": [artifacts.alert_ids[i] for i in malicious_indices]}]

    embeddings = np.concatenate([scaled_all, anomaly_scores.reshape(-1, 1)], axis=1)
    return CrossAlertResult(
        predictions=predictions,
        probabilities=probabilities,
        embeddings=embeddings,
        clusters=clusters,
    )
