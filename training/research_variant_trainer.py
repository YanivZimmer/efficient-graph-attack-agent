"""Trainer for experimental HGAT-family research variants."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.functional import cross_entropy, mse_loss, sigmoid

from clustering.incident_clusterer import cluster_incidents
from data.graph_builder import AlertGraphArtifacts
from evaluation.evaluator import evaluate_run
from models.baselines.graph_utils import build_semantic_alert_graph, build_sparse_causal_alert_graph
from models.hgat import HeterogeneousGAT
from models.research_variants import (
    DifferentiableClusterHGAT,
    MultiViewHGAT,
    MultiViewHGATv2,
    PrototypeMemoryHGAT,
    TemporalCausalHGATv2,
)
from training.trainer import (
    TrainingResult,
    WeakPairSet,
    _entity_counts,
    _mine_weak_pairs,
    _negative_pair_loss,
    _positive_pair_loss,
)


logger = logging.getLogger(__name__)


@dataclass
class ResearchVariantResult:
    """Artifacts and metrics produced by a research variant run."""

    variant: str
    embeddings: np.ndarray
    probabilities: np.ndarray
    threshold: float
    predictions: np.ndarray
    report_path: Path
    summary_path: Path
    summary: dict[str, object]


def _clone_artifacts(artifacts: AlertGraphArtifacts) -> AlertGraphArtifacts:
    return copy.deepcopy(artifacts)


def augment_time_harmonics(artifacts: AlertGraphArtifacts, *, num_frequencies: int = 2) -> None:
    """Append sinusoidal time features to alert node features."""
    timestamps = [record.timestamp for record in artifacts.alert_records]
    valid_times = [timestamp.timestamp() for timestamp in timestamps if timestamp is not None]
    if not valid_times:
        return
    min_time = min(valid_times)
    max_time = max(valid_times)
    scale = max(max_time - min_time, 1.0)
    normalized = []
    for timestamp in timestamps:
        if timestamp is None:
            normalized.append(0.0)
        else:
            normalized.append((timestamp.timestamp() - min_time) / scale)
    extra = []
    for value in normalized:
        row: list[float] = []
        for frequency in range(1, num_frequencies + 1):
            row.append(float(np.sin(2 * np.pi * frequency * value)))
            row.append(float(np.cos(2 * np.pi * frequency * value)))
        extra.append(row)
    extra_tensor = torch.tensor(extra, dtype=artifacts.data["alert"].x.dtype)
    artifacts.data["alert"].x = torch.cat([artifacts.data["alert"].x, extra_tensor], dim=1)


def _threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def _candidate_thresholds(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(probabilities)
    if len(unique) <= 256:
        return unique
    quantiles = np.linspace(0.0, 1.0, 256)
    return np.unique(np.quantile(unique, quantiles))


def tune_threshold(artifacts: AlertGraphArtifacts, probabilities: np.ndarray) -> float:
    """Pick the validation threshold that maximizes F1."""
    labels = artifacts.data["alert"].y.detach().cpu().numpy().astype(int)
    val_mask = artifacts.data["alert"].val_mask.detach().cpu().numpy().astype(bool)
    val_probs = probabilities[val_mask]
    val_labels = labels[val_mask]
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in _candidate_thresholds(val_probs):
        preds = (val_probs >= float(threshold)).astype(int)
        score = f1_score(val_labels, preds, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _normalized_time_targets(artifacts: AlertGraphArtifacts, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    timestamps = [record.timestamp.timestamp() for record in artifacts.alert_records if record.timestamp is not None]
    if not timestamps:
        zeros = torch.zeros(len(artifacts.alert_records), dtype=torch.float32, device=device)
        return zeros, torch.zeros(len(artifacts.alert_records), dtype=torch.bool, device=device)
    min_time = min(timestamps)
    max_time = max(timestamps)
    scale = max(max_time - min_time, 1.0)
    normalized = []
    valid_mask = []
    for record in artifacts.alert_records:
        if record.timestamp is None:
            normalized.append(0.0)
            valid_mask.append(False)
        else:
            normalized.append((record.timestamp.timestamp() - min_time) / scale)
            valid_mask.append(True)
    return (
        torch.tensor(normalized, dtype=torch.float32, device=device),
        torch.tensor(valid_mask, dtype=torch.bool, device=device),
    )


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()


def _temporal_v2_losses(
    model: TemporalCausalHGATv2,
    embeddings: torch.Tensor,
    auxiliary: dict[str, object],
    train_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    projected = model.encoder.project_embeddings(embeddings)
    time_scores = model.time_scores(embeddings)
    causal_edge_index = auxiliary["causal_graph"].edge_index
    time_targets = auxiliary["time_targets"]
    time_valid_mask = auxiliary["time_valid_mask"]

    train_time_mask = train_mask & time_valid_mask
    regression_loss = (
        mse_loss(time_scores[train_time_mask], time_targets[train_time_mask])
        if int(train_time_mask.sum().item()) > 0
        else embeddings.new_tensor(0.0)
    )

    if causal_edge_index.numel() == 0:
        zero = embeddings.new_tensor(0.0)
        return {
            "time_regression_loss": regression_loss,
            "continuity_loss": zero,
            "ordering_loss": zero,
        }

    src = causal_edge_index[0]
    dst = causal_edge_index[1]
    edge_mask = train_mask[src] & train_mask[dst] & time_valid_mask[src] & time_valid_mask[dst]
    if int(edge_mask.sum().item()) == 0:
        zero = embeddings.new_tensor(0.0)
        return {
            "time_regression_loss": regression_loss,
            "continuity_loss": zero,
            "ordering_loss": zero,
        }

    src = src[edge_mask]
    dst = dst[edge_mask]
    gap_targets = (time_targets[dst] - time_targets[src]).clamp_min(0.0)
    similarities = (projected[src] * projected[dst]).sum(dim=-1)
    continuity_weights = torch.exp(-4.0 * gap_targets)
    continuity_loss = _safe_mean((1.0 - similarities) * continuity_weights)

    predicted_gap = time_scores[dst] - time_scores[src]
    required_margin = 0.02 + 0.25 * gap_targets
    ordering_loss = _safe_mean(torch.relu(required_margin - predicted_gap))
    return {
        "time_regression_loss": regression_loss,
        "continuity_loss": continuity_loss,
        "ordering_loss": ordering_loss,
    }


def _contrastive_alignment_loss(left: torch.Tensor, right: torch.Tensor, *, temperature: float = 0.2) -> torch.Tensor:
    if left.size(0) < 2 or right.size(0) < 2:
        return left.new_tensor(0.0)
    logits = (left @ right.T) / temperature
    labels = torch.arange(left.size(0), device=left.device)
    return (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2.0


def _multiview_v2_losses(
    extras: dict[str, object],
    positive_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    projections = extras["views"]["projections"]
    gates = extras["views"]["gates"]
    selected_mask = positive_mask if int(positive_mask.sum().item()) >= 2 else fallback_mask

    hetero = projections["hetero"][selected_mask]
    temporal = projections["temporal"][selected_mask]
    semantic = projections["semantic"][selected_mask]
    alignment_loss = (
        _contrastive_alignment_loss(hetero, temporal)
        + _contrastive_alignment_loss(hetero, semantic)
        + _contrastive_alignment_loss(temporal, semantic)
    ) / 3.0

    mean_gates = gates[fallback_mask].mean(dim=0) if int(fallback_mask.sum().item()) > 0 else gates.mean(dim=0)
    uniform = torch.full_like(mean_gates, 1.0 / mean_gates.numel())
    gate_balance_loss = torch.sum((mean_gates - uniform) ** 2)
    return {
        "alignment_loss": alignment_loss,
        "gate_balance_loss": gate_balance_loss,
    }


def _node_metrics(artifacts: AlertGraphArtifacts, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = artifacts.data["alert"].y.detach().cpu().numpy().astype(int)
    val_mask = artifacts.data["alert"].val_mask.detach().cpu().numpy().astype(bool)
    val_labels = labels[val_mask]
    val_probs = probabilities[val_mask]
    val_preds = predictions[val_mask]
    return {
        "auc": float(roc_auc_score(val_labels, val_probs)) if len(np.unique(val_labels)) > 1 else float("nan"),
        "f1": float(f1_score(val_labels, val_preds, zero_division=0)),
        "precision": float(precision_score(val_labels, val_preds, zero_division=0)),
        "recall": float(recall_score(val_labels, val_preds, zero_division=0)),
    }


def _assignment_clusters(
    artifacts: AlertGraphArtifacts,
    selected_mask: np.ndarray,
    assignments: np.ndarray,
) -> list[dict[str, object]]:
    grouped: dict[int, list[str]] = {}
    for index, is_selected in enumerate(selected_mask.tolist()):
        if not is_selected:
            continue
        grouped.setdefault(int(assignments[index]), []).append(artifacts.alert_ids[index])
    clusters = []
    for cluster_id, alert_ids in sorted(grouped.items()):
        if len(alert_ids) >= 2:
            clusters.append({"incident_id": cluster_id, "alert_ids": sorted(alert_ids)})
        else:
            clusters.append({"incident_id": -1, "alert_ids": sorted(alert_ids)})
    return clusters


def _save_predictions(
    output_dir: Path,
    artifacts: AlertGraphArtifacts,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    assignments: np.ndarray | None = None,
) -> None:
    rows = []
    labels = artifacts.data["alert"].y.detach().cpu().numpy().astype(int)
    for index, alert_id in enumerate(artifacts.alert_ids):
        row = {
            "alert_id": alert_id,
            "probability": float(probabilities[index]),
            "prediction": int(predictions[index]),
            "label": int(labels[index]),
        }
        if assignments is not None:
            row["assignment"] = int(assignments[index])
        rows.append(row)
    (output_dir / "alert_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_clusters(path: Path, clusters: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for cluster in clusters:
            handle.write(json.dumps(cluster) + "\n")


def _build_model_and_auxiliary(
    variant: str,
    artifacts: AlertGraphArtifacts,
    *,
    device: str,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
    num_clusters: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    metadata = artifacts.data.metadata()
    alert_dim = int(artifacts.data["alert"].x.size(-1))
    entity_counts = _entity_counts(artifacts)
    if variant in {"baseline_hgat", "weak_pair_hgat", "temporal_causal_hgat"}:
        model = HeterogeneousGAT(
            metadata=metadata,
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        ).to(device)
        return model, {}
    if variant == "temporal_causal_hgat_v2":
        model = TemporalCausalHGATv2(
            metadata=metadata,
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        ).to(device)
        causal_graph = build_sparse_causal_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=6.0,
            max_neighbors_per_alert=12,
        ).to(device)
        time_targets, time_valid_mask = _normalized_time_targets(artifacts, device=device)
        return model, {
            "causal_graph": causal_graph,
            "time_targets": time_targets,
            "time_valid_mask": time_valid_mask,
        }
    if variant == "differentiable_cluster_hgat":
        model = DifferentiableClusterHGAT(
            metadata=metadata,
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
            num_slots=num_clusters,
        ).to(device)
        return model, {}
    if variant == "prototype_memory_hgat":
        model = PrototypeMemoryHGAT(
            metadata=metadata,
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
            num_prototypes=num_clusters,
        ).to(device)
        return model, {}
    if variant == "multiview_hgat":
        model = MultiViewHGAT(
            metadata=metadata,
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        ).to(device)
        temporal_graph = build_sparse_causal_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=6.0,
            max_neighbors_per_alert=8,
        ).to(device)
        semantic_graph = build_semantic_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=24.0,
            max_neighbors_per_alert=8,
        ).to(device)
        return model, {"temporal_graph": temporal_graph, "semantic_graph": semantic_graph}
    if variant == "multiview_hgat_v2":
        model = MultiViewHGATv2(
            metadata=metadata,
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        ).to(device)
        temporal_graph = build_sparse_causal_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=8.0,
            max_neighbors_per_alert=12,
        ).to(device)
        semantic_graph = build_semantic_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=24.0,
            max_neighbors_per_alert=12,
        ).to(device)
        return model, {"temporal_graph": temporal_graph, "semantic_graph": semantic_graph}
    raise KeyError(f"Unknown research variant {variant!r}")


def _compute_variant_forward(
    variant: str,
    model: torch.nn.Module,
    data,
    auxiliary: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if variant in {"multiview_hgat", "multiview_hgat_v2"}:
        embeddings, views = model.encode(
            data,
            temporal_graph=auxiliary["temporal_graph"],
            semantic_graph=auxiliary["semantic_graph"],
        )
        logits = model.classifier(embeddings).squeeze(-1)
        return embeddings, logits, {"views": views}
    embeddings = model.encode(data)
    if hasattr(model, "encoder") and hasattr(model.encoder, "classify_embeddings"):
        logits = model.encoder.classify_embeddings(embeddings)
    elif hasattr(model, "classify_embeddings"):
        logits = model.classify_embeddings(embeddings)
    else:
        logits = model(data)
    return embeddings, logits, {}


def train_research_variant(
    variant: str,
    artifacts: AlertGraphArtifacts,
    *,
    output_dir: Path,
    epochs: int = 6,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str | None = None,
    hidden_channels: int = 128,
    out_channels: int = 64,
    dropout: float = 0.3,
    num_clusters: int = 8,
    threshold_mode: str = "tuned",
    threshold: float = 0.5,
    selection_mode: str = "probability",
) -> ResearchVariantResult:
    """Train a research variant and evaluate it on AIT-ADS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = artifacts.data.to(resolved_device)
    train_mask = data["alert"].train_mask
    train_positive_mask = train_mask & (data["alert"].y == 1)

    model, auxiliary = _build_model_and_auxiliary(
        variant,
        artifacts,
        device=resolved_device,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
        num_clusters=num_clusters,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    weak_pairs = (
        _mine_weak_pairs(
            artifacts,
            tau_pos_hours=6.0,
            tau_neg_hours=24.0,
            max_pos_per_anchor=8,
            max_neg_per_anchor=16,
            seed=42,
        )
        if variant == "weak_pair_hgat"
        else WeakPairSet(np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64))
    )

    best_state = None
    best_val_auc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings, logits, extras = _compute_variant_forward(variant, model, data, auxiliary)
        cls_loss = bce_loss(logits[train_mask], data["alert"].y[train_mask])
        loss = cls_loss
        history_entry = {"epoch": float(epoch), "classification_loss": float(cls_loss.item())}

        if variant == "weak_pair_hgat":
            projected = model.project_embeddings(embeddings)
            pos_loss = _positive_pair_loss(projected, weak_pairs.positive_pairs)
            neg_loss = _negative_pair_loss(projected, weak_pairs.negative_pairs, margin=0.2)
            loss = loss + 2.0 * pos_loss + 0.1 * neg_loss
            history_entry["positive_pair_loss"] = float(pos_loss.item())
            history_entry["negative_pair_loss"] = float(neg_loss.item())
        elif variant == "differentiable_cluster_hgat":
            loss_terms = model.clustering_loss(embeddings, train_positive_mask)
            loss = loss + 0.8 * loss_terms["compactness"] + 0.1 * loss_terms["entropy"] + 0.3 * loss_terms["balance"] + 0.05 * loss_terms["slot_diversity"]
            history_entry.update({key: float(value.item()) for key, value in loss_terms.items()})
        elif variant == "prototype_memory_hgat":
            loss_terms = model.prototype_loss(embeddings, train_positive_mask)
            loss = loss + 0.8 * loss_terms["compactness"] + 0.05 * loss_terms["prototype_diversity"]
            history_entry.update({key: float(value.item()) for key, value in loss_terms.items()})
        elif variant == "multiview_hgat":
            align_loss = model.alignment_loss(extras["views"], train_mask)
            loss = loss + 0.1 * align_loss
            history_entry["alignment_loss"] = float(align_loss.item())
        elif variant == "temporal_causal_hgat_v2":
            loss_terms = _temporal_v2_losses(model, embeddings, auxiliary, train_mask)
            loss = (
                loss
                + 0.2 * loss_terms["time_regression_loss"]
                + 0.25 * loss_terms["continuity_loss"]
                + 0.15 * loss_terms["ordering_loss"]
            )
            history_entry.update({key: float(value.item()) for key, value in loss_terms.items()})
        elif variant == "multiview_hgat_v2":
            loss_terms = _multiview_v2_losses(extras, train_positive_mask, train_mask)
            loss = loss + 0.2 * loss_terms["alignment_loss"] + 0.05 * loss_terms["gate_balance_loss"]
            history_entry.update({key: float(value.item()) for key, value in loss_terms.items()})

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings, logits, _ = _compute_variant_forward(variant, model, data, auxiliary)
            probabilities = sigmoid(logits)
            labels = data["alert"].y.detach().cpu().numpy()
            val_mask = data["alert"].val_mask.detach().cpu().numpy().astype(bool)
            val_labels = labels[val_mask]
            val_probs = probabilities[val_mask].detach().cpu().numpy()
            val_auc = float(roc_auc_score(val_labels, val_probs)) if len(np.unique(val_labels)) > 1 else float("nan")

        history_entry["loss"] = float(loss.item())
        history_entry["val_auc"] = float(val_auc)
        history.append(history_entry)
        logger.info("%s epoch=%s loss=%.4f val_auc=%.4f", variant, epoch, loss.item(), val_auc)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings, logits, _ = _compute_variant_forward(variant, model, data, auxiliary)
        probabilities = sigmoid(logits).detach().cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()

    chosen_threshold = threshold if threshold_mode == "fixed" else tune_threshold(artifacts, probabilities)
    predictions = _threshold_predictions(probabilities, chosen_threshold)
    selected_mask = predictions.astype(bool)

    assignments = None
    if variant == "differentiable_cluster_hgat":
        with torch.no_grad():
            assignments = model.hard_assignments(embeddings).detach().cpu().numpy()
        clusters = _assignment_clusters(artifacts, selected_mask, assignments)
        _write_clusters(output_dir / "discovered_incidents.jsonl", clusters)
    elif variant == "prototype_memory_hgat":
        with torch.no_grad():
            assignments = model.hard_assignments(embeddings).detach().cpu().numpy()
        clusters = _assignment_clusters(artifacts, selected_mask, assignments)
        _write_clusters(output_dir / "discovered_incidents.jsonl", clusters)
    else:
        clusters = cluster_incidents(
            embeddings=embeddings_np,
            alert_ids=artifacts.alert_ids,
            predictions=predictions,
            probabilities=probabilities,
            output_path=output_dir / "discovered_incidents.jsonl",
            threshold=chosen_threshold,
            eps=0.3,
            min_samples=2,
            selection_mode=selection_mode,
        )

    report = evaluate_run(
        artifacts,
        predictions=predictions,
        probabilities=probabilities,
        clusters=clusters,
        output_path=output_dir / "evaluation_report.json",
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    np.save(output_dir / "alert_embeddings.npy", embeddings_np)
    _save_predictions(output_dir, artifacts, probabilities, predictions, assignments=assignments)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "variant": variant,
        "threshold_mode": threshold_mode,
        "threshold": chosen_threshold,
        "selection_mode": selection_mode,
        "selected_alert_count": int(predictions.sum()),
        "node_classification": _node_metrics(artifacts, probabilities, predictions),
        "cluster_summary": report.cluster_summary,
        "ground_truth_clustering": report.ground_truth_clustering,
        "best_val_auc": best_val_auc,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ResearchVariantResult(
        variant=variant,
        embeddings=embeddings_np,
        probabilities=probabilities,
        threshold=chosen_threshold,
        predictions=predictions,
        report_path=output_dir / "evaluation_report.json",
        summary_path=summary_path,
        summary=summary,
    )
