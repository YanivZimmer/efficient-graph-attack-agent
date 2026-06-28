"""Training loop for supervised alert-correlation upper-bound baselines."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

from data.graph_builder import AlertGraphArtifacts
from data.incident_labels import build_incident_label_tensor, incident_class_count
from models.baselines.crossalert import run_crossalert_baseline
from models.baselines.eckhoff_gmn import EckhoffGMN
from models.baselines.grain import GRAIN
from models.baselines.registry import SUPERVISED_UPPER_BOUND_METHODS
from training.baseline_trainer import BaselineTrainingResult, _classification_metrics, _compute_auc


logger = logging.getLogger(__name__)


@dataclass
class SupervisedBaselineResult:
    """Unified result for supervised upper-bound baselines."""

    model_name: str
    embeddings: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    clusters: list[dict[str, object]]
    best_val_auc: float
    metrics: dict[str, float]
    requires_incident_gt: bool = True


def _require_incident_gt(artifacts: AlertGraphArtifacts, method: str) -> None:
    if not artifacts.ground_truth_incidents:
        raise ValueError(
            f"{method} requires incident-level ground truth. "
            f"Dataset has no ground_truth_incidents; skip this method or use an alert-domain benchmark dataset."
        )


def _clusters_from_assignments(
    artifacts: AlertGraphArtifacts,
    predictions: np.ndarray,
    assignments: np.ndarray,
) -> list[dict[str, object]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for index, assignment in enumerate(assignments):
        if predictions[index] != 1:
            continue
        grouped[int(assignment)].append(artifacts.alert_ids[index])
    clusters = [
        {"incident_id": cluster_id, "alert_ids": sorted(alert_ids)}
        for cluster_id, alert_ids in sorted(grouped.items())
        if len(alert_ids) >= 2
    ]
    singletons = [
        {"incident_id": -1, "alert_ids": [artifacts.alert_ids[index]]}
        for index, assignment in enumerate(assignments)
        if predictions[index] == 1 and int(assignment) not in {key for key, vals in grouped.items() if len(vals) >= 2}
    ]
    return clusters + singletons


def _train_grain(
    artifacts: AlertGraphArtifacts,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
) -> tuple[GRAIN, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    incident_labels = build_incident_label_tensor(artifacts)
    num_incidents = max(incident_class_count(artifacts), 1)
    alert_dim = int(artifacts.data["alert"].x.size(-1))
    model = GRAIN(
        in_channels=alert_dim,
        num_incidents=num_incidents,
    ).to(device)
    model.set_alert_records(artifacts.alert_records)

    data = artifacts.data.to(device)
    incident_labels = incident_labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    best_state = None
    best_val_auc = float("-inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        incident_logits = model.incident_logits(data)
        train_mask = data["alert"].train_mask
        loss = bce_loss(logits[train_mask], data["alert"].y[train_mask])
        supervised_mask = train_mask & (incident_labels >= 0)
        if supervised_mask.any():
            loss = loss + ce_loss(incident_logits[supervised_mask], incident_labels[supervised_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = sigmoid(logits)
            val_auc = _compute_auc(data["alert"].y, probabilities, data["alert"].val_mask)
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 20:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data).detach().cpu().numpy()
        logits = model(data)
        probabilities = sigmoid(logits).detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
        incident_preds = model.incident_logits(data).argmax(dim=-1).detach().cpu().numpy()
    return model, embeddings, predictions, probabilities, best_val_auc, incident_preds


def _train_eckhoff_gmn(
    artifacts: AlertGraphArtifacts,
    *,
    epochs: int,
    learning_rate: float,
    device: str,
) -> tuple[EckhoffGMN, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    incident_labels = build_incident_label_tensor(artifacts)
    alert_dim = int(artifacts.data["alert"].x.size(-1))
    model = EckhoffGMN(in_channels=alert_dim).to(device)
    data = artifacts.data.to(device)
    incident_labels = incident_labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_auc = float("-inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data)
        logits = model(data)
        train_mask = data["alert"].train_mask
        loss = bce_loss(logits[train_mask], data["alert"].y[train_mask])
        supervised_mask = train_mask & (incident_labels >= 0)
        if supervised_mask.any():
            for class_id in torch.unique(incident_labels[supervised_mask]):
                class_mask = supervised_mask & (incident_labels == class_id)
                if int(class_mask.sum()) < 2:
                    continue
                class_embeddings = embeddings[class_mask]
                centroid = class_embeddings.mean(dim=0, keepdim=True)
                loss = loss + 0.1 * ((class_embeddings - centroid) ** 2).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = sigmoid(model(data))
            val_auc = _compute_auc(data["alert"].y, probabilities, data["alert"].val_mask)
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 20:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data).detach().cpu().numpy()
        logits = model(data)
        probabilities = sigmoid(logits).detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)

    prototypes: dict[int, list[np.ndarray]] = defaultdict(list)
    train_mask = artifacts.data["alert"].train_mask.detach().cpu().numpy().astype(bool)
    incident_np = incident_labels.detach().cpu().numpy()
    for index, is_train in enumerate(train_mask):
        if not is_train or incident_np[index] < 0:
            continue
        prototypes[int(incident_np[index])].append(embeddings[index])
    prototype_matrix = np.stack(
        [np.mean(vectors, axis=0) for _, vectors in sorted(prototypes.items())],
        axis=0,
    )
    if prototype_matrix.size == 0:
        assignments = np.zeros(len(artifacts.alert_ids), dtype=int)
    else:
        normalized = prototype_matrix / (np.linalg.norm(prototype_matrix, axis=1, keepdims=True) + 1e-6)
        query = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
        assignments = np.argmax(query @ normalized.T, axis=1)

    return model, embeddings, predictions, probabilities, best_val_auc, assignments


def train_supervised_baseline(
    model_name: str,
    artifacts: AlertGraphArtifacts,
    *,
    output_dir: Path | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> SupervisedBaselineResult:
    """Train a supervised upper-bound baseline that requires incident GT."""
    if model_name not in SUPERVISED_UPPER_BOUND_METHODS:
        raise KeyError(f"Unknown supervised baseline {model_name!r}")

    _require_incident_gt(artifacts, model_name)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "crossalert":
        incident_labels = build_incident_label_tensor(artifacts).detach().cpu().numpy()
        result = run_crossalert_baseline(artifacts, incident_labels=incident_labels)
        metrics = _classification_metrics(
            artifacts.data["alert"].y,
            torch.tensor(result.probabilities),
            torch.tensor(result.predictions),
            artifacts.data["alert"].val_mask,
        )
        supervised = SupervisedBaselineResult(
            model_name=model_name,
            embeddings=result.embeddings,
            predictions=result.predictions,
            probabilities=result.probabilities,
            clusters=result.clusters,
            best_val_auc=metrics["auc"],
            metrics=metrics,
        )
    elif model_name == "grain":
        _, embeddings, predictions, probabilities, best_val_auc, assignments = _train_grain(
            artifacts,
            epochs=epochs,
            learning_rate=learning_rate,
            device=resolved_device,
        )
        metrics = _classification_metrics(
            artifacts.data["alert"].y,
            torch.tensor(probabilities),
            torch.tensor(predictions),
            artifacts.data["alert"].val_mask,
        )
        clusters = _clusters_from_assignments(artifacts, predictions, assignments)
        supervised = SupervisedBaselineResult(
            model_name=model_name,
            embeddings=embeddings,
            predictions=predictions,
            probabilities=probabilities,
            clusters=clusters,
            best_val_auc=best_val_auc,
            metrics=metrics,
        )
    elif model_name == "eckhoff_gmn":
        _, embeddings, predictions, probabilities, best_val_auc, assignments = _train_eckhoff_gmn(
            artifacts,
            epochs=epochs,
            learning_rate=learning_rate,
            device=resolved_device,
        )
        metrics = _classification_metrics(
            artifacts.data["alert"].y,
            torch.tensor(probabilities),
            torch.tensor(predictions),
            artifacts.data["alert"].val_mask,
        )
        clusters = _clusters_from_assignments(artifacts, predictions, assignments)
        supervised = SupervisedBaselineResult(
            model_name=model_name,
            embeddings=embeddings,
            predictions=predictions,
            probabilities=probabilities,
            clusters=clusters,
            best_val_auc=best_val_auc,
            metrics=metrics,
        )
    else:
        raise KeyError(model_name)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "alert_embeddings.npy", supervised.embeddings)
        (output_dir / "discovered_incidents.jsonl").write_text(
            "\n".join(json.dumps(row) for row in supervised.clusters) + ("\n" if supervised.clusters else ""),
            encoding="utf-8",
        )

    return supervised
