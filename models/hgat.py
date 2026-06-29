"""Heterogeneous graph attention model for alert classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import normalize
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


class HeterogeneousGAT(nn.Module):
    """Two-layer heterogeneous GAT with per-edge-type attention."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        alert_in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        projection_dim: int = 32,
        entity_counts: dict[str, int] | None = None,
        heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the heterogeneous GAT backbone and classifier head."""
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout
        node_types, _edge_types = metadata
        entity_counts = entity_counts or {}

        self.entity_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(max(entity_counts.get(node_type, 1), 1), hidden_channels)
                for node_type in node_types
                if node_type != "alert"
            }
        )
        self.alert_encoder = Linear(alert_in_channels, hidden_channels)

        self.conv1 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
        )
        self.conv2 = HGTConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            heads=heads,
        )
        self.projection = Linear(out_channels, projection_dim)
        self.classifier = Linear(out_channels, 1)

    def _initial_node_features(self, data: HeteroData) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        if "alert" in data.node_types:
            features["alert"] = self.alert_encoder(data["alert"].x)

        for node_type, embedding in self.entity_embeddings.items():
            count = int(data[node_type].num_nodes)
            device = data["alert"].x.device
            node_indices = torch.arange(count, device=device)
            features[node_type] = embedding(node_indices)
        return features

    def encode(self, data: HeteroData) -> torch.Tensor:
        """Return 64-dimensional alert embeddings."""
        x_dict = self._initial_node_features(data)
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {node_type: torch.relu(features) for node_type, features in x_dict.items()}
        x_dict = {
            node_type: nn.functional.dropout(features, p=self.dropout, training=self.training)
            for node_type, features in x_dict.items()
        }
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        return x_dict["alert"]

    def classify_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return alert logits from precomputed alert embeddings."""
        return self.classifier(embeddings).squeeze(-1)

    def project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return normalized projected alert embeddings for metric learning."""
        return normalize(self.projection(embeddings), dim=-1)

    def project(self, data: HeteroData) -> torch.Tensor:
        """Encode and project alerts into the metric-learning space."""
        return self.project_embeddings(self.encode(data))

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Return alert logits for binary classification."""
        embeddings = self.encode(data)
        return self.classify_embeddings(embeddings)
