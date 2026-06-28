"""Generic training loop for HGAT and literature baseline models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.functional import sigmoid

from data.graph_builder import AlertGraphArtifacts
from models.baselines.anomal_e import AnomalE
from models.baselines.graph_ids import GraphIDS
from models.baselines.registry import WEAKLY_SUPERVISED_METHODS, build_model


logger = logging.getLogger(__name__)


@dataclass
class BaselineTrainingResult:
    """Training artifacts for any baseline model."""

    model_name: str
    model: nn.Module
    embeddings: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    history: list[dict[str, float]]
    best_val_auc: float
    metrics: dict[str, float]


def _compute_auc(labels: torch.Tensor, probabilities: torch.Tensor, mask: torch.Tensor) -> float:
    masked_labels = labels[mask].detach().cpu().numpy()
    masked_probs = probabilities[mask].detach().cpu().numpy()
    if len(np.unique(masked_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(masked_labels, masked_probs))


def _classification_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    masked_labels = labels[mask].detach().cpu().numpy()
    masked_probs = probabilities[mask].detach().cpu().numpy()
    masked_preds = predictions[mask].detach().cpu().numpy()
    return {
        "auc": float(roc_auc_score(masked_labels, masked_probs)) if len(np.unique(masked_labels)) > 1 else float("nan"),
        "f1": float(f1_score(masked_labels, masked_preds, zero_division=0)),
        "precision": float(precision_score(masked_labels, masked_preds, zero_division=0)),
        "recall": float(recall_score(masked_labels, masked_preds, zero_division=0)),
    }


def _self_supervised_loss(model: nn.Module, data: Any) -> torch.Tensor:
    train_mask = data["alert"].train_mask
    labels = data["alert"].y
    if isinstance(model, GraphIDS):
        benign_mask = train_mask & (labels == 0)
        return model.masked_reconstruction_loss(data, benign_mask)
    if isinstance(model, AnomalE):
        return model.dgi_loss(data, train_mask)
    raise TypeError(f"Model {type(model)!r} does not support self-supervised loss")


def train_baseline(
    model_name: str,
    artifacts: AlertGraphArtifacts,
    *,
    output_dir: Path | None = None,
    epochs: int = 100,
    pretrain_epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str | None = None,
) -> BaselineTrainingResult:
    """Train a registered baseline model with the appropriate objective."""
    if model_name not in WEAKLY_SUPERVISED_METHODS:
        raise KeyError(
            f"Unknown weakly-supervised model {model_name!r}. "
            f"Expected one of {sorted(WEAKLY_SUPERVISED_METHODS)}"
        )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = artifacts.data.to(resolved_device)
    model = build_model(model_name, artifacts).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = float("-inf")
    stale_epochs = 0

    total_epochs = pretrain_epochs + epochs if model_name in {"graph_ids", "anomal_e"} else epochs

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()

        use_self_supervised = model_name in {"graph_ids", "anomal_e"} and epoch <= pretrain_epochs
        if use_self_supervised:
            loss = _self_supervised_loss(model, data)
        else:
            logits = model(data)
            loss = bce_loss(logits[data["alert"].train_mask], data["alert"].y[data["alert"].train_mask])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            probabilities = sigmoid(logits)
            val_auc = _compute_auc(data["alert"].y, probabilities, data["alert"].val_mask)

        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "val_auc": float(val_auc),
                "phase": 1.0 if use_self_supervised else 2.0,
            }
        )
        logger.info(
            "%s epoch=%s phase=%s loss=%.4f val_auc=%.4f",
            model_name,
            epoch,
            "pretrain" if use_self_supervised else "finetune",
            loss.item(),
            val_auc,
        )

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience and epoch > pretrain_epochs:
                logger.info("Early stopping triggered for %s at epoch %s", model_name, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data).detach().cpu().numpy()
        logits = model(data)
        probabilities = sigmoid(logits).detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)

    metrics = _classification_metrics(
        data["alert"].y,
        torch.tensor(probabilities),
        torch.tensor(predictions),
        data["alert"].val_mask,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "model.pt")
        np.save(output_dir / "alert_embeddings.npy", embeddings)
        rows = [
            {
                "alert_id": alert_id,
                "probability": float(probabilities[index]),
                "prediction": int(predictions[index]),
                "label": int(data["alert"].y[index].item()),
            }
            for index, alert_id in enumerate(artifacts.alert_ids)
        ]
        (output_dir / "alert_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return BaselineTrainingResult(
        model_name=model_name,
        model=model,
        embeddings=embeddings,
        predictions=predictions,
        probabilities=probabilities,
        history=history,
        best_val_auc=best_val_auc,
        metrics=metrics,
    )
