"""Self-supervised GraphIDS baseline: E-GraphSAGE + Transformer MAE."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Linear, SAGEConv

from models.baselines.graph_utils import build_alert_homogeneous_graph


class _TransformerBlock(nn.Module):
    """Single-head Transformer encoder block for flow embedding reconstruction."""

    def __init__(self, hidden_channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels),
        )

    def forward(self, tokens: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=mask)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens


class GraphIDS(nn.Module):
    """GraphSAGE encoder with masked Transformer autoencoder reconstruction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
        mask_ratio: float = 0.3,
    ) -> None:
        """Initialize the GraphIDS baseline."""
        super().__init__()
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.input_encoder = Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.transformer = _TransformerBlock(out_channels, dropout=dropout)
        self.decoder = Linear(out_channels, out_channels)
        self.score_head = Linear(out_channels, 1)

    def _homogeneous(self, data: HeteroData | Data) -> Data:
        if isinstance(data, Data):
            return data
        return build_alert_homogeneous_graph(data)

    def encode(self, data: HeteroData | Data) -> torch.Tensor:
        """Return GraphSAGE alert embeddings."""
        homo = self._homogeneous(data)
        features = self.input_encoder(homo.x)
        features = torch.relu(self.conv1(features, homo.edge_index))
        features = nn.functional.dropout(features, p=self.dropout, training=self.training)
        features = self.conv2(features, homo.edge_index)
        return features

    def reconstruction_error(self, data: HeteroData | Data) -> torch.Tensor:
        """Compute per-alert reconstruction error (higher means more anomalous)."""
        embeddings = self.encode(data)
        batch = embeddings.unsqueeze(0)
        reconstructed = self.decoder(self.transformer(batch)).squeeze(0)
        return torch.mean((embeddings - reconstructed) ** 2, dim=1)

    def forward(self, data: HeteroData | Data) -> torch.Tensor:
        """Return logits where higher values indicate maliciousness."""
        error = self.reconstruction_error(data)
        embeddings = self.encode(data)
        calibrated = self.score_head(embeddings).squeeze(-1)
        return calibrated - error

    def masked_reconstruction_loss(self, data: HeteroData | Data, mask: torch.Tensor) -> torch.Tensor:
        """Train reconstruction on a subset of alerts (typically benign train rows)."""
        homo = self._homogeneous(data)
        selected = mask.nonzero(as_tuple=False).view(-1)
        if selected.numel() == 0:
            return torch.tensor(0.0, device=homo.x.device)

        embeddings = self.encode(data)[selected]
        batch = embeddings.unsqueeze(0)
        num_tokens = embeddings.size(0)
        num_masked = max(1, int(math.ceil(num_tokens * self.mask_ratio)))
        masked_positions = torch.randperm(num_tokens, device=embeddings.device)[:num_masked]

        attention_mask = torch.zeros(num_tokens, dtype=torch.bool, device=embeddings.device)
        attention_mask[masked_positions] = True
        corrupted = batch.clone()
        corrupted[:, masked_positions, :] = 0.0

        reconstructed = self.decoder(
            self.transformer(corrupted, mask=attention_mask.unsqueeze(0))
        )
        target = batch[:, masked_positions, :]
        prediction = reconstructed[:, masked_positions, :]
        return torch.mean((prediction - target) ** 2)
