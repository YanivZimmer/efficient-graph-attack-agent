"""Training loop for heterogeneous GAT alert classification."""

from __future__ import annotations

from collections import defaultdict
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


@dataclass(frozen=True)
class WeakPairSet:
    """Precomputed weak supervision pairs for incident-aware metric learning."""

    positive_pairs: np.ndarray
    negative_pairs: np.ndarray


def _shared_entity(record_a, record_b) -> bool:
    return any(
        value == record_b.entities.get(entity_type)
        for entity_type, value in record_a.entities.items()
        if value and entity_type in record_b.entities
    )


def _time_gap_hours(record_a, record_b) -> float:
    if record_a.timestamp is None or record_b.timestamp is None:
        return float("inf")
    return abs((record_a.timestamp - record_b.timestamp).total_seconds()) / 3600.0


def _mine_weak_pairs(
    artifacts: AlertGraphArtifacts,
    *,
    tau_pos_hours: float,
    tau_neg_hours: float,
    max_pos_per_anchor: int,
    max_neg_per_anchor: int,
    seed: int,
) -> WeakPairSet:
    """Mine weak positive/negative alert pairs from the training split."""
    labels = artifacts.data["alert"].y.detach().cpu().numpy().astype(int)
    train_mask = artifacts.data["alert"].train_mask.detach().cpu().numpy().astype(bool)
    records = artifacts.alert_records
    malicious_train = [index for index, is_train in enumerate(train_mask) if is_train and labels[index] == 1]
    benign_train = [index for index, is_train in enumerate(train_mask) if is_train and labels[index] == 0]

    positive_pairs: set[tuple[int, int]] = set()
    positive_counts: dict[int, int] = defaultdict(int)
    entity_buckets: dict[str, dict[str, list[int]]] = {
        entity_type: defaultdict(list) for entity_type in ("host", "user", "process", "ip")
    }
    for index in malicious_train:
        for entity_type, value in records[index].entities.items():
            if entity_type in entity_buckets and value:
                entity_buckets[entity_type][value].append(index)

    for buckets in entity_buckets.values():
        for alert_indices in buckets.values():
            if len(alert_indices) < 2:
                continue
            ordered = sorted(
                alert_indices,
                key=lambda index: (
                    records[index].timestamp.timestamp() if records[index].timestamp else float("-inf"),
                    index,
                ),
            )
            for offset, left_index in enumerate(ordered):
                if positive_counts[left_index] >= max_pos_per_anchor:
                    continue
                left_record = records[left_index]
                for right_index in ordered[offset + 1 :]:
                    right_record = records[right_index]
                    if _time_gap_hours(left_record, right_record) > tau_pos_hours:
                        break
                    pair = (min(left_index, right_index), max(left_index, right_index))
                    if pair in positive_pairs:
                        continue
                    if positive_counts[left_index] >= max_pos_per_anchor:
                        break
                    if positive_counts[right_index] >= max_pos_per_anchor:
                        continue
                    positive_pairs.add(pair)
                    positive_counts[left_index] += 1
                    positive_counts[right_index] += 1

    rng = np.random.default_rng(seed)
    negative_pairs: set[tuple[int, int]] = set()
    malicious_pool = np.array(malicious_train, dtype=np.int64)
    benign_pool = np.array(benign_train, dtype=np.int64)
    for anchor_index in malicious_train:
        anchor_record = records[anchor_index]
        anchor_negatives = 0

        if len(benign_pool) > 0:
            candidate_count = min(max_neg_per_anchor // 2, len(benign_pool))
            sampled_benign = rng.choice(benign_pool, size=candidate_count, replace=False)
            for sampled_index in sampled_benign.tolist():
                pair = (min(anchor_index, sampled_index), max(anchor_index, sampled_index))
                if pair in negative_pairs:
                    continue
                negative_pairs.add(pair)
                anchor_negatives += 1

        attempts = 0
        max_attempts = max(16, max_neg_per_anchor * 20)
        while anchor_negatives < max_neg_per_anchor and attempts < max_attempts and len(malicious_pool) > 1:
            attempts += 1
            sampled_index = int(rng.choice(malicious_pool))
            if sampled_index == anchor_index:
                continue
            sampled_record = records[sampled_index]
            if _shared_entity(anchor_record, sampled_record):
                continue
            if _time_gap_hours(anchor_record, sampled_record) < tau_neg_hours:
                continue
            if anchor_record.tactic == sampled_record.tactic:
                continue
            pair = (min(anchor_index, sampled_index), max(anchor_index, sampled_index))
            if pair in negative_pairs:
                continue
            negative_pairs.add(pair)
            anchor_negatives += 1

    positive_array = np.array(sorted(positive_pairs), dtype=np.int64) if positive_pairs else np.empty((0, 2), dtype=np.int64)
    negative_array = np.array(sorted(negative_pairs), dtype=np.int64) if negative_pairs else np.empty((0, 2), dtype=np.int64)
    logger.info(
        "Mined %s weak positive pairs and %s weak negative pairs",
        len(positive_array),
        len(negative_array),
    )
    return WeakPairSet(positive_pairs=positive_array, negative_pairs=negative_array)


def _positive_pair_loss(projected: torch.Tensor, pairs: np.ndarray) -> torch.Tensor:
    if len(pairs) == 0:
        return projected.new_tensor(0.0)
    pair_tensor = torch.as_tensor(pairs, device=projected.device, dtype=torch.long)
    similarities = (projected[pair_tensor[:, 0]] * projected[pair_tensor[:, 1]]).sum(dim=-1)
    return (1.0 - similarities).mean()


def _negative_pair_loss(projected: torch.Tensor, pairs: np.ndarray, *, margin: float) -> torch.Tensor:
    if len(pairs) == 0:
        return projected.new_tensor(0.0)
    pair_tensor = torch.as_tensor(pairs, device=projected.device, dtype=torch.long)
    similarities = (projected[pair_tensor[:, 0]] * projected[pair_tensor[:, 1]]).sum(dim=-1)
    return torch.relu(similarities - margin).mean()


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
    projection_dim: int = 32,
    use_incident_pair_loss: bool = False,
    lambda_pos: float = 0.5,
    lambda_neg: float = 0.25,
    tau_pos_hours: float = 6.0,
    tau_neg_hours: float = 24.0,
    margin_neg: float = 0.2,
    max_pos_pairs_per_anchor: int = 8,
    max_neg_pairs_per_anchor: int = 16,
    pair_seed: int = 42,
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
        projection_dim=projection_dim,
        entity_counts=_entity_counts(artifacts),
    ).to(resolved_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    weak_pairs = (
        _mine_weak_pairs(
            artifacts,
            tau_pos_hours=tau_pos_hours,
            tau_neg_hours=tau_neg_hours,
            max_pos_per_anchor=max_pos_pairs_per_anchor,
            max_neg_per_anchor=max_neg_pairs_per_anchor,
            seed=pair_seed,
        )
        if use_incident_pair_loss
        else WeakPairSet(
            positive_pairs=np.empty((0, 2), dtype=np.int64),
            negative_pairs=np.empty((0, 2), dtype=np.int64),
        )
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data)
        logits = model.classify_embeddings(embeddings)
        cls_loss = loss_fn(logits[data["alert"].train_mask], data["alert"].y[data["alert"].train_mask])
        pos_loss = embeddings.new_tensor(0.0)
        neg_loss = embeddings.new_tensor(0.0)
        loss = cls_loss
        if use_incident_pair_loss:
            projected = model.project_embeddings(embeddings)
            pos_loss = _positive_pair_loss(projected, weak_pairs.positive_pairs)
            neg_loss = _negative_pair_loss(projected, weak_pairs.negative_pairs, margin=margin_neg)
            loss = loss + lambda_pos * pos_loss + lambda_neg * neg_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            probabilities = sigmoid(logits)
            val_auc = _compute_auc(data["alert"].y, probabilities, data["alert"].val_mask)

        history_entry = {
            "epoch": float(epoch),
            "loss": float(loss.item()),
            "classification_loss": float(cls_loss.item()),
            "val_auc": float(val_auc),
        }
        if use_incident_pair_loss:
            history_entry["positive_pair_loss"] = float(pos_loss.item())
            history_entry["negative_pair_loss"] = float(neg_loss.item())
        history.append(history_entry)
        logger.info(
            "Epoch %s | loss=%.4f | cls=%.4f | pos=%.4f | neg=%.4f | val_auc=%.4f",
            epoch,
            loss.item(),
            cls_loss.item(),
            pos_loss.item(),
            neg_loss.item(),
            val_auc,
        )

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
