"""Shared dataset loader registry for GNN benchmark runners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts
from data.loaders.ait_ads_loader import load_ait_ads_graph
from data.loaders.cicids_loader import load_cicids_graph
from data.loaders.darpa2000_loader import load_darpa2000_graph
from data.loaders.darpa_tc_loader import load_darpa_tc_graph
from data.loaders.excytin_loader import load_excytin_graph
from data.loaders.iscx2012_loader import load_iscx2012_graph
from data.loaders.lanl_loader import load_lanl_graph
from data.loaders.primary_loader import load_primary_graph

DATASET_LOADERS: dict[str, Callable[..., AlertGraphArtifacts]] = {
    "primary": load_primary_graph,
    "ait_ads": load_ait_ads_graph,
    "darpa2000": load_darpa2000_graph,
    "darpa_tc": load_darpa_tc_graph,
    "excytin": load_excytin_graph,
    "iscx2012": load_iscx2012_graph,
    "lanl": load_lanl_graph,
    "cicids": load_cicids_graph,
}

ALERT_DOMAIN_DATASETS = frozenset({"primary", "ait_ads", "darpa2000", "iscx2012", "darpa_tc", "excytin"})


def resolve_dataset_path(dataset: str, data_root: Path) -> Path:
    if dataset == "primary":
        return data_root / "0b1972fe_backup" / "training_data_rich_examples.jsonl"
    return data_root / dataset


def load_dataset(
    dataset: str,
    data_root: Path,
    *,
    lanl_sample_days: int = 5,
    ait_ads_max_records: int = 10_000,
) -> AlertGraphArtifacts:
    path = resolve_dataset_path(dataset, data_root)
    if dataset == "lanl":
        return load_lanl_graph(path, sample_days=lanl_sample_days)
    if dataset == "ait_ads":
        return load_ait_ads_graph(path, max_records=ait_ads_max_records)
    return DATASET_LOADERS[dataset](path)
