"""Supervised GraphSAGE baseline inspired by GNN-IDS (ARES 2024)."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Linear, SAGEConv

from models.baselines.graph_utils import build_alert_homogeneous_graph


class GNNIDS(nn.Module):
    """Two-layer GraphSAGE classifier on an alert connectivity graph."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the GNN-IDS baseline."""
        super().__init__()
        self.dropout = dropout
        self.input_encoder = Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.classifier = Linear(out_channels, 1)
        self._homo_cache: Data | None = None

    def _homogeneous(self, data: HeteroData | Data) -> Data:
        if isinstance(data, Data):
            return data
        return build_alert_homogeneous_graph(data)

    def encode(self, data: HeteroData | Data) -> torch.Tensor:
        """Return alert embeddings after GraphSAGE message passing."""
        homo = self._homogeneous(data)
        features = self.input_encoder(homo.x)
        features = self.conv1(features, homo.edge_index)
        features = torch.relu(features)
        features = nn.functional.dropout(features, p=self.dropout, training=self.training)
        features = self.conv2(features, homo.edge_index)
        return features

    def forward(self, data: HeteroData | Data) -> torch.Tensor:
        """Return binary classification logits."""
        embeddings = self.encode(data)
        return self.classifier(embeddings).squeeze(-1)
