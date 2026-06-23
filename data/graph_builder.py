"""Build PyG HeteroData graphs from normalized alert records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

from data.schema import (
    InferredSchema,
    extract_mitre_fields,
    get_nested_value,
    infer_schema,
    label_to_binary,
    parse_timestamp,
    severity_to_float,
)


NODE_TYPES = ("alert", "host", "user", "process", "ip")
EDGE_REL = "connects_to"
REV_EDGE_REL = "rev_connects_to"


class AlertRecord(BaseModel):
    """Normalized alert record used for graph construction."""

    alert_id: str
    label: int
    tactic: str = "<unknown>"
    technique: str = "<unknown>"
    severity: float = 0.0
    timestamp: datetime | None = None
    entities: dict[str, str] = Field(default_factory=dict)


@dataclass
class AlertGraphArtifacts:
    """Graph artifacts produced by the builder."""

    data: HeteroData
    schema: InferredSchema
    alert_ids: list[str]
    alert_records: list[AlertRecord]
    tactic_vocab: list[str]
    technique_vocab: list[str]
    ground_truth_incidents: dict[str, list[str]] = field(default_factory=dict)


def _normalize_record(record: dict[str, Any], schema: InferredSchema) -> AlertRecord | None:
    alert_id = get_nested_value(record, schema.alert_id_field)
    label_value = get_nested_value(record, schema.label_field)
    if not isinstance(alert_id, str) or not alert_id.strip():
        return None
    label = label_to_binary(label_value)
    if label is None:
        return None

    tactic, technique = extract_mitre_fields(record, schema)
    severity_value = get_nested_value(record, schema.severity_field)
    timestamp_value = get_nested_value(record, schema.timestamp_field)
    if timestamp_value is None and schema.traces_path:
        traces = get_nested_value(record, ".".join(schema.traces_path))
        if isinstance(traces, list) and traces:
            timestamp_value = traces[0].get("timestamp")

    entities: dict[str, str] = {}
    for entity_type, entity_path in schema.entity_fields.items():
        value = get_nested_value(record, entity_path)
        if isinstance(value, str) and value.strip():
            entities[entity_type] = value.strip()

    return AlertRecord(
        alert_id=alert_id.strip(),
        label=label,
        tactic=(tactic or "<unknown>").strip(),
        technique=(technique or "<unknown>").strip(),
        severity=severity_to_float(severity_value),
        timestamp=parse_timestamp(timestamp_value),
        entities=entities,
    )


def normalize_records(records: list[dict[str, Any]], schema: InferredSchema | None = None) -> tuple[list[AlertRecord], InferredSchema]:
    """Convert raw JSON records into normalized alert records."""
    resolved_schema = schema or infer_schema(records)
    normalized: list[AlertRecord] = []
    for record in records:
        parsed = _normalize_record(record, resolved_schema)
        if parsed is not None:
            normalized.append(parsed)
    if not normalized:
        raise ValueError("No valid alert records found after schema normalization")
    return normalized, resolved_schema


def _index_entities(records: list[AlertRecord]) -> dict[str, dict[str, int]]:
    indices: dict[str, dict[str, int]] = {entity_type: {} for entity_type in ("host", "user", "process", "ip")}
    for record in records:
        for entity_type, value in record.entities.items():
            if entity_type not in indices:
                continue
            bucket = indices[entity_type]
            if value not in bucket:
                bucket[value] = len(bucket)
    return indices


def _one_hot(values: list[str], vocab: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(vocab)}
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for row_index, value in enumerate(values):
        column = lookup.get(value)
        if column is not None:
            matrix[row_index, column] = 1.0
    return matrix


def _build_alert_features(records: list[AlertRecord]) -> tuple[torch.Tensor, list[str], list[str]]:
    tactic_vocab = sorted({record.tactic for record in records})
    technique_vocab = sorted({record.technique for record in records})
    timestamps = [record.timestamp for record in records]
    valid_times = [timestamp for timestamp in timestamps if timestamp is not None]
    base_time = min(valid_times) if valid_times else None

    time_deltas: list[float] = []
    for timestamp in timestamps:
        if timestamp is None or base_time is None:
            time_deltas.append(0.0)
            continue
        delta_hours = (timestamp - base_time).total_seconds() / 3600.0
        time_deltas.append(delta_hours)

    max_delta = max(time_deltas) if time_deltas else 1.0
    if max_delta <= 0:
        max_delta = 1.0
    normalized_deltas = [delta / max_delta for delta in time_deltas]
    severities = [record.severity for record in records]

    tactic_matrix = _one_hot([record.tactic for record in records], tactic_vocab)
    technique_matrix = _one_hot([record.technique for record in records], technique_vocab)
    numeric = np.array(
        [[severity, delta] for severity, delta in zip(severities, normalized_deltas, strict=True)],
        dtype=np.float32,
    )
    features = np.concatenate([tactic_matrix, technique_matrix, numeric], axis=1)
    return torch.tensor(features, dtype=torch.float32), tactic_vocab, technique_vocab


def _build_edge_index(
    source_indices: list[int],
    target_indices: list[int],
) -> torch.Tensor:
    if not source_indices:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([source_indices, target_indices], dtype=torch.long)


def build_graph_from_records(
    records: list[dict[str, Any]],
    *,
    schema: InferredSchema | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    ground_truth_incidents: dict[str, list[str]] | None = None,
) -> AlertGraphArtifacts:
    """Build a heterogeneous alert graph from raw JSON records."""
    normalized, resolved_schema = normalize_records(records, schema=schema)
    entity_indices = _index_entities(normalized)
    alert_features, tactic_vocab, technique_vocab = _build_alert_features(normalized)

    labels = torch.tensor([record.label for record in normalized], dtype=torch.float32)
    indices = np.arange(len(normalized))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        stratify=[record.label for record in normalized],
    )
    train_mask = torch.zeros(len(normalized), dtype=torch.bool)
    val_mask = torch.zeros(len(normalized), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    data = HeteroData()
    data["alert"].x = alert_features
    data["alert"].y = labels
    data["alert"].train_mask = train_mask
    data["alert"].val_mask = val_mask

    for entity_type, mapping in entity_indices.items():
        count = len(mapping)
        data[entity_type].num_nodes = max(count, 1)

    edge_pairs: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    for alert_index, record in enumerate(normalized):
        for entity_type, value in record.entities.items():
            entity_index = entity_indices[entity_type][value]
            forward = (("alert", EDGE_REL, entity_type))
            reverse = ((entity_type, REV_EDGE_REL, "alert"))
            edge_pairs.setdefault(forward, ([], []))[0].append(alert_index)
            edge_pairs.setdefault(forward, ([], []))[1].append(entity_index)
            edge_pairs.setdefault(reverse, ([], []))[0].append(entity_index)
            edge_pairs.setdefault(reverse, ([], []))[1].append(alert_index)

    for edge_type, (sources, targets) in edge_pairs.items():
        data[edge_type].edge_index = _build_edge_index(sources, targets)

    return AlertGraphArtifacts(
        data=data,
        schema=resolved_schema,
        alert_ids=[record.alert_id for record in normalized],
        alert_records=normalized,
        tactic_vocab=tactic_vocab,
        technique_vocab=technique_vocab,
        ground_truth_incidents=ground_truth_incidents or {},
    )
