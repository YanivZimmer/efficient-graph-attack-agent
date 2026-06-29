"""Experimental HGAT-family research variants for incident discovery."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import normalize
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Linear, SAGEConv

from models.hgat import HeterogeneousGAT


class AlertGraphSAGEEncoder(nn.Module):
    """Lightweight GraphSAGE encoder for alert-only graph views."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.input_encoder = Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, data: Data) -> torch.Tensor:
        features = self.input_encoder(data.x)
        features = self.conv1(features, data.edge_index)
        features = torch.relu(features)
        features = nn.functional.dropout(features, p=self.dropout, training=self.training)
        return self.conv2(features, data.edge_index)


class DifferentiableClusterHGAT(nn.Module):
    """HGAT encoder with trainable cluster slots and soft assignments."""

    def __init__(
        self,
        *,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        alert_in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        entity_counts: dict[str, int] | None = None,
        dropout: float = 0.3,
        num_slots: int = 8,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.encoder = HeterogeneousGAT(
            metadata=metadata,
            alert_in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        )
        self.cluster_slots = nn.Parameter(torch.randn(num_slots, out_channels))

    def encode(self, data: HeteroData) -> torch.Tensor:
        return self.encoder.encode(data)

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.encoder(data)

    def assignment_probabilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = normalize(embeddings, dim=-1)
        normalized_slots = normalize(self.cluster_slots, dim=-1)
        similarities = normalized_embeddings @ normalized_slots.T
        return torch.softmax(similarities / self.temperature, dim=-1)

    def clustering_loss(self, embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if int(mask.sum()) == 0:
            zero = embeddings.new_tensor(0.0)
            return {"compactness": zero, "entropy": zero, "balance": zero, "slot_diversity": zero}
        selected = embeddings[mask]
        q = self.assignment_probabilities(selected)
        normalized_embeddings = normalize(selected, dim=-1)
        normalized_slots = normalize(self.cluster_slots, dim=-1)
        similarities = normalized_embeddings @ normalized_slots.T
        compactness = 1.0 - torch.sum(q * similarities, dim=-1).mean()
        entropy = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=-1).mean()
        usage = q.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / usage.numel())
        balance = torch.sum((usage - uniform) ** 2)
        slot_gram = normalized_slots @ normalized_slots.T
        slot_diversity = ((slot_gram - torch.eye(slot_gram.size(0), device=slot_gram.device)) ** 2).mean()
        return {
            "compactness": compactness,
            "entropy": entropy,
            "balance": balance,
            "slot_diversity": slot_diversity,
        }

    def hard_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.assignment_probabilities(embeddings).argmax(dim=-1)


class PrototypeMemoryHGAT(nn.Module):
    """HGAT encoder with trainable incident prototypes."""

    def __init__(
        self,
        *,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        alert_in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        entity_counts: dict[str, int] | None = None,
        dropout: float = 0.3,
        num_prototypes: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = HeterogeneousGAT(
            metadata=metadata,
            alert_in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, out_channels))

    def encode(self, data: HeteroData) -> torch.Tensor:
        return self.encoder.encode(data)

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.encoder(data)

    def prototype_loss(self, embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if int(mask.sum()) == 0:
            zero = embeddings.new_tensor(0.0)
            return {"compactness": zero, "prototype_diversity": zero}
        selected = normalize(embeddings[mask], dim=-1)
        normalized_prototypes = normalize(self.prototypes, dim=-1)
        similarities = selected @ normalized_prototypes.T
        compactness = 1.0 - similarities.max(dim=-1).values.mean()
        proto_gram = normalized_prototypes @ normalized_prototypes.T
        prototype_diversity = ((proto_gram - torch.eye(proto_gram.size(0), device=proto_gram.device)) ** 2).mean()
        return {"compactness": compactness, "prototype_diversity": prototype_diversity}

    def hard_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = normalize(embeddings, dim=-1)
        normalized_prototypes = normalize(self.prototypes, dim=-1)
        return (normalized_embeddings @ normalized_prototypes.T).argmax(dim=-1)


class MultiViewHGAT(nn.Module):
    """Fuse heterogeneous, temporal, and semantic alert views."""

    def __init__(
        self,
        *,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        alert_in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        entity_counts: dict[str, int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hetero_encoder = HeterogeneousGAT(
            metadata=metadata,
            alert_in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        )
        self.temporal_encoder = AlertGraphSAGEEncoder(
            in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.semantic_encoder = AlertGraphSAGEEncoder(
            in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.gate = Linear(out_channels * 3, 3)
        self.classifier = Linear(out_channels, 1)

    def encode(
        self,
        data: HeteroData,
        *,
        temporal_graph: Data,
        semantic_graph: Data,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hetero = self.hetero_encoder.encode(data)
        temporal = self.temporal_encoder.encode(temporal_graph)
        semantic = self.semantic_encoder.encode(semantic_graph)
        concatenated = torch.cat([hetero, temporal, semantic], dim=-1)
        gates = torch.softmax(self.gate(concatenated), dim=-1)
        fused = (
            gates[:, 0:1] * hetero
            + gates[:, 1:2] * temporal
            + gates[:, 2:3] * semantic
        )
        return fused, {"hetero": hetero, "temporal": temporal, "semantic": semantic, "gates": gates}

    def forward(
        self,
        data: HeteroData,
        *,
        temporal_graph: Data,
        semantic_graph: Data,
    ) -> torch.Tensor:
        embeddings, _ = self.encode(data, temporal_graph=temporal_graph, semantic_graph=semantic_graph)
        return self.classifier(embeddings).squeeze(-1)

    def alignment_loss(self, view_embeddings: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        if int(mask.sum()) == 0:
            return view_embeddings["hetero"].new_tensor(0.0)
        hetero = normalize(view_embeddings["hetero"][mask], dim=-1)
        temporal = normalize(view_embeddings["temporal"][mask], dim=-1)
        semantic = normalize(view_embeddings["semantic"][mask], dim=-1)
        return (
            (1.0 - (hetero * temporal).sum(dim=-1)).mean()
            + (1.0 - (hetero * semantic).sum(dim=-1)).mean()
            + (1.0 - (temporal * semantic).sum(dim=-1)).mean()
        ) / 3.0


class TemporalCausalHGATv2(nn.Module):
    """HGAT with auxiliary temporal heads for causal continuity and ordering."""

    def __init__(
        self,
        *,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        alert_in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        entity_counts: dict[str, int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = HeterogeneousGAT(
            metadata=metadata,
            alert_in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        )
        self.time_head = Linear(out_channels, 1)

    def encode(self, data: HeteroData) -> torch.Tensor:
        return self.encoder.encode(data)

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.encoder(data)

    def classify_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.encoder.classify_embeddings(embeddings)

    def time_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.time_head(embeddings).squeeze(-1)


class MultiViewHGATv2(nn.Module):
    """Stronger multiview HGAT with common-space projections and residual fusion."""

    def __init__(
        self,
        *,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        alert_in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        entity_counts: dict[str, int] | None = None,
        dropout: float = 0.3,
        projection_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hetero_encoder = HeterogeneousGAT(
            metadata=metadata,
            alert_in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=entity_counts,
            dropout=dropout,
        )
        self.temporal_encoder = AlertGraphSAGEEncoder(
            in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.semantic_encoder = AlertGraphSAGEEncoder(
            in_channels=alert_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.hetero_proj = Linear(out_channels, projection_dim)
        self.temporal_proj = Linear(out_channels, projection_dim)
        self.semantic_proj = Linear(out_channels, projection_dim)
        self.gate = nn.Sequential(
            Linear(out_channels * 3, out_channels),
            nn.ReLU(),
            Linear(out_channels, 3),
        )
        self.residual = Linear(out_channels * 3, out_channels)
        self.classifier = Linear(out_channels, 1)

    def encode(
        self,
        data: HeteroData,
        *,
        temporal_graph: Data,
        semantic_graph: Data,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hetero = self.hetero_encoder.encode(data)
        temporal = self.temporal_encoder.encode(temporal_graph)
        semantic = self.semantic_encoder.encode(semantic_graph)
        stacked = torch.cat([hetero, temporal, semantic], dim=-1)
        gates = torch.softmax(self.gate(stacked), dim=-1)
        fused = (
            0.5 * hetero
            + 0.5
            * (
                gates[:, 0:1] * hetero
                + gates[:, 1:2] * temporal
                + gates[:, 2:3] * semantic
                + self.residual(stacked)
            )
        )
        projections = {
            "hetero": normalize(self.hetero_proj(hetero), dim=-1),
            "temporal": normalize(self.temporal_proj(temporal), dim=-1),
            "semantic": normalize(self.semantic_proj(semantic), dim=-1),
        }
        return fused, {
            "hetero": hetero,
            "temporal": temporal,
            "semantic": semantic,
            "gates": gates,
            "projections": projections,
        }

    def forward(
        self,
        data: HeteroData,
        *,
        temporal_graph: Data,
        semantic_graph: Data,
    ) -> torch.Tensor:
        embeddings, _ = self.encode(data, temporal_graph=temporal_graph, semantic_graph=semantic_graph)
        return self.classifier(embeddings).squeeze(-1)
