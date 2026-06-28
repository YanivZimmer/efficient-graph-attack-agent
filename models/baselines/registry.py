"""Baseline model registry and factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from data.graph_builder import AlertGraphArtifacts
from data.incident_labels import incident_class_count
from models.baselines.anomal_e import AnomalE
from models.baselines.eckhoff_gmn import EckhoffGMN
from models.baselines.gnn_ids import GNNIDS
from models.baselines.graph_ids import GraphIDS
from models.baselines.grain import GRAIN
from models.hgat import HeterogeneousGAT


ModelBuilder = Callable[[AlertGraphArtifacts, dict[str, Any]], nn.Module]

BASELINE_MODELS: dict[str, str] = {
    "graphweaver": "GraphWeaver: rule-based entity-overlap correlation (CIKM 2024 style)",
    "hgat": "Heterogeneous GAT (project model)",
    "gnn_ids": "GNN-IDS: supervised 2-layer GraphSAGE (ARES 2024 style)",
    "graph_ids": "GraphIDS: E-GraphSAGE + Transformer MAE reconstruction",
    "anomal_e": "Anomal-E: E-GraphSAGE + Deep Graph Infomax",
    "grain": "GRAIN: causal alert-graph GNN with incident supervision (Computers & Security 2025)",
    "eckhoff_gmn": "Eckhoff et al.: graph-matching alert contextualisation (2025)",
    "crossalert": "CrossAlert: multi-stage alert feature fusion (IEEE CNS 2024)",
}

NON_TRAINABLE_METHODS: frozenset[str] = frozenset({"graphweaver"})

SUPERVISED_UPPER_BOUND_METHODS: frozenset[str] = frozenset({"grain", "eckhoff_gmn", "crossalert"})

WEAKLY_SUPERVISED_METHODS: frozenset[str] = frozenset({"hgat", "gnn_ids", "graph_ids", "anomal_e"})

COMPARISON_TIERS: dict[str, str] = {
    "graphweaver": "rule_based_lower_bound",
    "hgat": "weakly_supervised",
    "gnn_ids": "flow_level_gnn",
    "graph_ids": "flow_level_gnn",
    "anomal_e": "flow_level_gnn",
    "grain": "supervised_upper_bound",
    "eckhoff_gmn": "supervised_upper_bound",
    "crossalert": "supervised_upper_bound",
}

DEFAULT_ALL_METHODS = [
    "graphweaver",
    "hgat",
    "gnn_ids",
    "graph_ids",
    "anomal_e",
    "grain",
    "eckhoff_gmn",
    "crossalert",
]


def _entity_counts(artifacts: AlertGraphArtifacts) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_type in ("host", "user", "process", "ip"):
        if node_type in artifacts.data.node_types:
            counts[node_type] = int(artifacts.data[node_type].num_nodes)
    return counts


def build_model(model_name: str, artifacts: AlertGraphArtifacts, **kwargs: Any) -> nn.Module:
    """Instantiate a registered baseline model."""
    alert_dim = int(artifacts.data["alert"].x.size(-1))
    hidden_channels = int(kwargs.get("hidden_channels", 128))
    out_channels = int(kwargs.get("out_channels", 64))
    dropout = float(kwargs.get("dropout", 0.3))

    if model_name == "hgat":
        return HeterogeneousGAT(
            metadata=artifacts.data.metadata(),
            alert_in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            entity_counts=_entity_counts(artifacts),
            dropout=dropout,
        )
    if model_name == "gnn_ids":
        return GNNIDS(
            in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
    if model_name == "graph_ids":
        return GraphIDS(
            in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
    if model_name == "anomal_e":
        return AnomalE(
            in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
    if model_name == "grain":
        model = GRAIN(
            in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_incidents=max(incident_class_count(artifacts), 1),
            dropout=dropout,
        )
        model.set_alert_records(artifacts.alert_records)
        return model
    if model_name == "eckhoff_gmn":
        return EckhoffGMN(
            in_channels=alert_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
    raise KeyError(f"Unknown trainable model {model_name!r}. Expected one of {sorted(BASELINE_MODELS)}")
