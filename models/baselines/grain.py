"""GRAIN-style supervised causal-graph GNN (Xiao et al., Computers & Security 2025)."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Linear, SAGEConv

from data.graph_builder import AlertRecord
from models.baselines.graph_utils import build_causal_alert_graph


class GRAIN(nn.Module):
    """Simplified GRAIN: causal alert graph encoder + binary and incident heads."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_incidents: int = 1,
        dropout: float = 0.3,
        max_gap_hours: float = 2.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.max_gap_hours = max_gap_hours
        self.alert_records: list[AlertRecord] = []
        self.input_encoder = Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.classifier = Linear(out_channels, 1)
        self.incident_head = Linear(out_channels, max(num_incidents, 1))

    def set_alert_records(self, alert_records: list[AlertRecord]) -> None:
        self.alert_records = alert_records

    def _homogeneous(self, data: HeteroData | Data) -> Data:
        if isinstance(data, Data):
            return data
        return build_causal_alert_graph(
            data,
            self.alert_records,
            max_gap_hours=self.max_gap_hours,
        )

    def encode(self, data: HeteroData | Data) -> torch.Tensor:
        homo = self._homogeneous(data)
        features = self.input_encoder(homo.x)
        features = self.conv1(features, homo.edge_index)
        features = torch.relu(features)
        features = nn.functional.dropout(features, p=self.dropout, training=self.training)
        features = self.conv2(features, homo.edge_index)
        return features

    def forward(self, data: HeteroData | Data) -> torch.Tensor:
        embeddings = self.encode(data)
        return self.classifier(embeddings).squeeze(-1)

    def incident_logits(self, data: HeteroData | Data) -> torch.Tensor:
        return self.incident_head(self.encode(data))
