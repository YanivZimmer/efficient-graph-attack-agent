"""Training loop for heterogeneous GAT alert classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.functional import sigmoid

from data.graph_builder import AlertGraphArtifacts
from models.hgat import HeterogeneousGAT


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Artifacts produced by model training."""

    model: HeterogeneousGAT
    embeddings: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    history: list[dict[str, float]]
    best_val_auc: float


def _entity_counts(artifacts: AlertGraphArtifacts) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_type in ("host", "user", "process", "ip"):
        if node_type in artifacts.data.node_types:
            counts[node_type] = int(artifacts.data[node_type].num_nodes)
    return counts


def _compute_auc(labels: torch.Tensor, probabilities: torch.Tensor, mask: torch.Tensor) -> float:
    masked_labels = labels[mask].detach().cpu().numpy()
    masked_probs = probabilities[mask].detach().cpu().numpy()
    if len(np.unique(masked_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(masked_labels, masked_probs))


def train_model(
    artifacts: AlertGraphArtifacts,
    *,
    output_dir: Path,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str | None = None,
) -> TrainingResult:
    """Train the heterogeneous GAT and persist checkpoints and embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = artifacts.data.to(resolved_device)
    metadata = data.metadata()
    alert_dim = int(data["alert"].x.size(-1))

    model = HeterogeneousGAT(
        metadata=metadata,
        alert_in_channels=alert_dim,
        entity_counts=_entity_counts(artifacts),
    ).to(resolved_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits[data["alert"].train_mask], data["alert"].y[data["alert"].train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            probabilities = sigmoid(logits)
            val_auc = _compute_auc(data["alert"].y, probabilities, data["alert"].val_mask)

        history.append({"epoch": float(epoch), "loss": float(loss.item()), "val_auc": float(val_auc)})
        logger.info("Epoch %s | loss=%.4f | val_auc=%.4f", epoch, loss.item(), val_auc)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Early stopping triggered at epoch %s", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data).detach().cpu().numpy()
        logits = model(data)
        probabilities = sigmoid(logits).detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)

    checkpoint_path = output_dir / "model.pt"
    embeddings_path = output_dir / "alert_embeddings.npy"
    predictions_path = output_dir / "alert_predictions.json"

    torch.save(model.state_dict(), checkpoint_path)
    np.save(embeddings_path, embeddings)
    prediction_rows = [
        {
            "alert_id": alert_id,
            "probability": float(probabilities[index]),
            "prediction": int(predictions[index]),
            "label": int(artifacts.data["alert"].y[index].item()),
        }
        for index, alert_id in enumerate(artifacts.alert_ids)
    ]
    predictions_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")

    logger.info("Saved checkpoint to %s", checkpoint_path)
    logger.info("Saved embeddings to %s", embeddings_path)
    logger.info("Saved predictions to %s", predictions_path)

    return TrainingResult(
        model=model,
        embeddings=embeddings,
        predictions=predictions,
        probabilities=probabilities,
        history=history,
        best_val_auc=best_val_auc,
    )
