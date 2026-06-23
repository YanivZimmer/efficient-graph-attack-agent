"""Self-supervised Anomal-E baseline: E-GraphSAGE + Deep Graph Infomax."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Linear, SAGEConv

from models.baselines.graph_utils import build_alert_homogeneous_graph


class AnomalE(nn.Module):
    """GraphSAGE encoder trained with a DGI-style mutual information objective."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the Anomal-E baseline."""
        super().__init__()
        self.dropout = dropout
        self.input_encoder = Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.summary = Linear(out_channels, out_channels)
        self.discriminator = nn.Bilinear(out_channels, out_channels, 1)

    def _homogeneous(self, data: HeteroData | Data) -> Data:
        if isinstance(data, Data):
            return data
        return build_alert_homogeneous_graph(data)

    def encode(self, data: HeteroData | Data) -> torch.Tensor:
        """Return alert embeddings from the E-GraphSAGE encoder."""
        homo = self._homogeneous(data)
        features = self.input_encoder(homo.x)
        features = torch.relu(self.conv1(features, homo.edge_index))
        features = nn.functional.dropout(features, p=self.dropout, training=self.training)
        features = self.conv2(features, homo.edge_index)
        return features

    def _summary_vector(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.summary(torch.mean(embeddings, dim=0, keepdim=True)))

    def dgi_loss(self, data: HeteroData | Data, train_mask: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss on the training alerts."""
        homo = self._homogeneous(data)
        positive = self.encode(data)[train_mask]
        if positive.numel() == 0:
            return torch.tensor(0.0, device=homo.x.device)

        summary = self._summary_vector(positive)
        positive_scores = self.discriminator(positive, summary.expand(positive.size(0), -1)).squeeze(-1)

        homo = self._homogeneous(data)
        corrupted_features = homo.x.clone()
        permuted = train_mask.nonzero(as_tuple=False).view(-1)
        shuffled_indices = permuted[torch.randperm(permuted.size(0), device=permuted.device)]
        corrupted_features[permuted] = homo.x[shuffled_indices]
        corrupted_graph = Data(
            x=corrupted_features,
            edge_index=homo.edge_index,
            y=homo.y,
            train_mask=homo.train_mask,
            val_mask=homo.val_mask,
        )
        negative = self.encode(corrupted_graph)[train_mask]
        negative_scores = self.discriminator(negative, summary.expand(negative.size(0), -1)).squeeze(-1)

        positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores,
            torch.ones_like(positive_scores),
        )
        negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            negative_scores,
            torch.zeros_like(negative_scores),
        )
        return positive_loss + negative_loss

    def anomaly_scores(self, data: HeteroData | Data) -> torch.Tensor:
        """Return per-alert anomaly scores (lower discriminator score means more anomalous)."""
        embeddings = self.encode(data)
        train_mask = data["alert"].train_mask if isinstance(data, HeteroData) else data.train_mask
        summary = self._summary_vector(embeddings[train_mask])
        scores = self.discriminator(embeddings, summary.expand(embeddings.size(0), -1)).squeeze(-1)
        return -scores

    def forward(self, data: HeteroData | Data) -> torch.Tensor:
        """Return logits for malicious classification."""
        return self.anomaly_scores(data)
