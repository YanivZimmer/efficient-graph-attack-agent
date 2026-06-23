"""Convert heterogeneous alert graphs into homogeneous PyG graphs."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch_geometric.data import Data, HeteroData


def build_alert_homogeneous_graph(data: HeteroData) -> Data:
    """Project alert-entity edges into an alert-alert graph via shared entities."""
    num_alerts = int(data["alert"].x.size(0))
    edge_pairs: set[tuple[int, int]] = set()

    for edge_type in data.edge_types:
        source_type, _relation, target_type = edge_type
        if source_type != "alert":
            continue
        edge_index = data[edge_type].edge_index
        entity_to_alerts: dict[int, list[int]] = defaultdict(list)
        for alert_index, entity_index in zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
            strict=True,
        ):
            entity_to_alerts[int(entity_index)].append(int(alert_index))

        for alert_indices in entity_to_alerts.values():
            unique_alerts = sorted(set(alert_indices))
            for left in range(len(unique_alerts)):
                for right in range(left + 1, len(unique_alerts)):
                    source = unique_alerts[left]
                    target = unique_alerts[right]
                    edge_pairs.add((source, target))
                    edge_pairs.add((target, source))

    if not edge_pairs and num_alerts > 1:
        for index in range(num_alerts - 1):
            edge_pairs.add((index, index + 1))
            edge_pairs.add((index + 1, index))

    if edge_pairs:
        edge_index = torch.tensor(sorted(edge_pairs), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    homo = Data(
        x=data["alert"].x,
        edge_index=edge_index,
        y=data["alert"].y,
        train_mask=data["alert"].train_mask,
        val_mask=data["alert"].val_mask,
    )
    homo.num_nodes = num_alerts
    return homo
