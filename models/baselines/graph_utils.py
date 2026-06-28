"""Convert heterogeneous alert graphs into homogeneous PyG graphs."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch_geometric.data import Data, HeteroData

from data.graph_builder import AlertRecord


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


def build_causal_alert_graph(
    data: HeteroData,
    alert_records: list[AlertRecord],
    *,
    max_gap_hours: float = 2.0,
) -> Data:
    """Build a directed causal alert graph (entity overlap + temporal precedence)."""
    homo = build_alert_homogeneous_graph(data)
    edge_pairs: set[tuple[int, int]] = set(
        zip(homo.edge_index[0].tolist(), homo.edge_index[1].tolist(), strict=True)
    )

    for left_index, left_record in enumerate(alert_records):
        if left_record.timestamp is None:
            continue
        for right_index, right_record in enumerate(alert_records):
            if left_index == right_index or right_record.timestamp is None:
                continue
            if right_record.timestamp <= left_record.timestamp:
                continue
            gap_hours = (right_record.timestamp - left_record.timestamp).total_seconds() / 3600.0
            if gap_hours > max_gap_hours:
                continue
            shares_entity = any(
                left_record.entities.get(entity_type) == right_record.entities.get(entity_type)
                for entity_type in left_record.entities
                if entity_type in right_record.entities
            )
            if shares_entity:
                edge_pairs.add((left_index, right_index))

    if edge_pairs:
        homo.edge_index = torch.tensor(sorted(edge_pairs), dtype=torch.long).t().contiguous()
    return homo
