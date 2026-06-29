"""Experimental incident clustering policies for ablation studies."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from data.graph_builder import AlertRecord


@dataclass(frozen=True)
class ClusterAblationStrategy:
    """Configuration for one inference-time clustering ablation."""

    name: str
    base: str = "dbscan"
    relation_policy: str = "entity"
    max_gap_hours: float = 24.0
    max_neighbors_per_bucket: int = 12
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2
    min_cluster_size: int = 2
    split_tactic: bool = False
    split_entity: bool = False
    split_time_gap_hours: float | None = None
    split_time_gap_quantile: float | None = None
    split_time_gap_estimator: str | None = None
    split_time_gap_min_hours: float = 1.0
    split_time_gap_max_hours: float = 72.0
    macro_gap_ratio_threshold: float = 2.5
    macro_gap_boundary_scale: float = 0.95
    cosine_threshold: float | None = None
    bayesian_blocks_prior: float = 8.0


def default_ablation_strategies() -> list[ClusterAblationStrategy]:
    """Return a compact ablation grid for giant-cluster splitting and graph clustering."""
    return [
        ClusterAblationStrategy(name="dbscan_current"),
        ClusterAblationStrategy(
            name="dbscan_split_time_macro_elbow",
            split_time_gap_estimator="macro_elbow",
            split_time_gap_min_hours=0.25,
        ),
        ClusterAblationStrategy(
            name="dbscan_split_time_macro_elbow_floor4",
            split_time_gap_estimator="macro_elbow",
            split_time_gap_min_hours=4.0,
        ),
        ClusterAblationStrategy(name="dbscan_split_time_4h", split_time_gap_hours=4.0),
        ClusterAblationStrategy(name="dbscan_split_time_6h", split_time_gap_hours=6.0),
        ClusterAblationStrategy(name="dbscan_split_time_12h", split_time_gap_hours=12.0),
        ClusterAblationStrategy(name="dbscan_split_time_24h", split_time_gap_hours=24.0),
        ClusterAblationStrategy(name="dbscan_split_time_48h", split_time_gap_hours=48.0),
        ClusterAblationStrategy(
            name="dbscan_split_time_adaptive_q90",
            split_time_gap_quantile=0.90,
        ),
        ClusterAblationStrategy(
            name="dbscan_split_time_adaptive_q95",
            split_time_gap_quantile=0.95,
        ),
        ClusterAblationStrategy(name="dbscan_split_tactic", split_tactic=True),
        ClusterAblationStrategy(name="dbscan_split_entity_24h", split_entity=True, max_gap_hours=24.0),
        ClusterAblationStrategy(
            name="dbscan_split_tactic_time_24h",
            split_tactic=True,
            split_time_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="dbscan_split_entity_time_24h",
            split_entity=True,
            split_time_gap_hours=24.0,
            max_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="dbscan_split_tactic_entity_time_24h",
            split_tactic=True,
            split_entity=True,
            split_time_gap_hours=24.0,
            max_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="graph_entity_6h",
            base="graph",
            relation_policy="entity",
            max_gap_hours=6.0,
        ),
        ClusterAblationStrategy(
            name="graph_entity_24h",
            base="graph",
            relation_policy="entity",
            max_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="graph_semantic_6h",
            base="graph",
            relation_policy="semantic",
            max_gap_hours=6.0,
        ),
        ClusterAblationStrategy(
            name="graph_semantic_24h",
            base="graph",
            relation_policy="semantic",
            max_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="graph_entity_semantic_6h",
            base="graph",
            relation_policy="entity_semantic",
            max_gap_hours=6.0,
        ),
        ClusterAblationStrategy(
            name="graph_entity_semantic_24h",
            base="graph",
            relation_policy="entity_semantic",
            max_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="graph_entity_semantic_cos_24h",
            base="graph",
            relation_policy="entity_semantic",
            max_gap_hours=24.0,
            cosine_threshold=0.95,
        ),
        ClusterAblationStrategy(
            name="graph_entity_semantic_time_split_24h",
            base="graph",
            relation_policy="entity_semantic",
            max_gap_hours=24.0,
            split_time_gap_hours=24.0,
        ),
        ClusterAblationStrategy(
            name="temporal_only_macro_elbow",
            base="temporal",
            split_time_gap_estimator="macro_elbow",
            split_time_gap_min_hours=0.25,
        ),
        ClusterAblationStrategy(
            name="temporal_only_macro_elbow_floor4",
            base="temporal",
            split_time_gap_estimator="macro_elbow",
            split_time_gap_min_hours=4.0,
        ),
        ClusterAblationStrategy(
            name="temporal_only_4h",
            base="temporal",
            split_time_gap_hours=4.0,
        ),
        ClusterAblationStrategy(
            name="temporal_only_6h",
            base="temporal",
            split_time_gap_hours=6.0,
        ),
        ClusterAblationStrategy(
            name="temporal_only_12h",
            base="temporal",
            split_time_gap_hours=12.0,
        ),
        ClusterAblationStrategy(
            name="temporal_only_adaptive_q90",
            base="temporal",
            split_time_gap_quantile=0.90,
        ),
        ClusterAblationStrategy(
            name="temporal_only_adaptive_q95",
            base="temporal",
            split_time_gap_quantile=0.95,
        ),
        ClusterAblationStrategy(
            name="bayesian_blocks_p4",
            base="bayesian_blocks",
            bayesian_blocks_prior=4.0,
        ),
        ClusterAblationStrategy(
            name="bayesian_blocks_p6",
            base="bayesian_blocks",
            bayesian_blocks_prior=6.0,
        ),
        ClusterAblationStrategy(
            name="bayesian_blocks_p8",
            base="bayesian_blocks",
            bayesian_blocks_prior=8.0,
        ),
        ClusterAblationStrategy(
            name="bayesian_blocks_p10",
            base="bayesian_blocks",
            bayesian_blocks_prior=10.0,
        ),
    ]


def cluster_with_strategy(
    *,
    strategy: ClusterAblationStrategy,
    embeddings: np.ndarray,
    alert_ids: list[str],
    records: list[AlertRecord],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    threshold: float,
    selection_mode: str = "probability",
) -> list[dict[str, object]]:
    """Cluster selected alerts with an ablation strategy and write JSONL output."""
    selected_indices = _select_indices(predictions, probabilities, threshold, selection_mode)
    if len(selected_indices) == 0:
        clusters = [{"incident_id": -1, "alert_ids": []}]
        _write_jsonl(output_path, clusters)
        return clusters

    if strategy.base == "dbscan":
        groups = _dbscan_groups(
            embeddings=embeddings,
            selected_indices=selected_indices,
            eps=strategy.dbscan_eps,
            min_samples=strategy.dbscan_min_samples,
        )
    elif strategy.base == "graph":
        groups = _relation_component_groups(
            selected_indices=selected_indices,
            embeddings=embeddings,
            records=records,
            relation_policy=strategy.relation_policy,
            max_gap_hours=strategy.max_gap_hours,
            max_neighbors_per_bucket=strategy.max_neighbors_per_bucket,
            cosine_threshold=strategy.cosine_threshold,
        )
    elif strategy.base == "temporal":
        groups = [selected_indices.astype(int).tolist()]
    elif strategy.base == "bayesian_blocks":
        groups = _bayesian_block_groups(
            selected_indices=selected_indices,
            records=records,
            prior=strategy.bayesian_blocks_prior,
        )
    else:
        raise ValueError(f"Unknown clustering base {strategy.base!r}")

    groups = _apply_splitters(groups, records, strategy)
    clusters = _format_clusters(groups, alert_ids, min_cluster_size=strategy.min_cluster_size)
    _write_jsonl(output_path, clusters)
    return clusters


def _select_indices(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    selection_mode: str,
) -> np.ndarray:
    if selection_mode == "or":
        return np.where((probabilities >= threshold) | (predictions == 1))[0]
    if selection_mode == "probability":
        return np.where(probabilities >= threshold)[0]
    if selection_mode == "prediction":
        return np.where(predictions == 1)[0]
    raise ValueError(f"Unknown selection_mode {selection_mode!r}")


def _dbscan_groups(
    *,
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    eps: float,
    min_samples: int,
) -> list[list[int]]:
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings[selected_indices])
    grouped: dict[int, list[int]] = defaultdict(list)
    for alert_index, label in zip(selected_indices.tolist(), labels.tolist(), strict=True):
        grouped[int(label)].append(int(alert_index))
    return [indices for _label, indices in sorted(grouped.items(), key=lambda item: item[0])]


def _relation_component_groups(
    *,
    selected_indices: np.ndarray,
    embeddings: np.ndarray,
    records: list[AlertRecord],
    relation_policy: str,
    max_gap_hours: float,
    max_neighbors_per_bucket: int,
    cosine_threshold: float | None,
) -> list[list[int]]:
    selected_set = set(int(index) for index in selected_indices.tolist())
    adjacency: dict[int, set[int]] = {index: set() for index in selected_set}
    buckets: dict[str, list[int]] = defaultdict(list)

    for index in selected_set:
        record = records[index]
        if relation_policy in {"entity", "entity_semantic"}:
            for entity_type, value in record.entities.items():
                if value:
                    buckets[f"entity:{entity_type}:{value}"].append(index)
        if relation_policy in {"semantic", "entity_semantic"}:
            buckets[f"tactic:{record.tactic}"].append(index)
            buckets[f"technique:{record.technique}"].append(index)

    normalized_embeddings = _normalize_rows(embeddings) if cosine_threshold is not None else embeddings
    max_gap_seconds = max_gap_hours * 3600.0
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        ordered = sorted(bucket_indices, key=lambda index: (_timestamp_seconds(records[index]), index))
        for offset, left_index in enumerate(ordered):
            neighbor_count = 0
            left_time = _timestamp_seconds(records[left_index])
            for right_index in ordered[offset + 1 :]:
                right_time = _timestamp_seconds(records[right_index])
                if np.isfinite(left_time) and np.isfinite(right_time):
                    gap_seconds = right_time - left_time
                    if gap_seconds < 0:
                        continue
                    if gap_seconds > max_gap_seconds:
                        break
                if cosine_threshold is not None:
                    similarity = float(normalized_embeddings[left_index] @ normalized_embeddings[right_index])
                    if similarity < cosine_threshold:
                        continue
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                neighbor_count += 1
                if neighbor_count >= max_neighbors_per_bucket:
                    break

    return _connected_components(adjacency)


def _apply_splitters(
    groups: list[list[int]],
    records: list[AlertRecord],
    strategy: ClusterAblationStrategy,
) -> list[list[int]]:
    split_groups = groups
    if strategy.split_tactic:
        split_groups = _split_by_tactic(split_groups, records)
    if strategy.split_entity:
        split_groups = _split_by_entity_components(
            split_groups,
            records,
            max_gap_hours=strategy.max_gap_hours,
            max_neighbors_per_bucket=strategy.max_neighbors_per_bucket,
        )
    if (
        strategy.split_time_gap_hours is not None
        or strategy.split_time_gap_quantile is not None
        or strategy.split_time_gap_estimator is not None
    ):
        split_groups = _split_by_temporal_gaps(
            split_groups,
            records,
            max_gap_hours=strategy.split_time_gap_hours,
            gap_quantile=strategy.split_time_gap_quantile,
            gap_estimator=strategy.split_time_gap_estimator,
            min_gap_hours=strategy.split_time_gap_min_hours,
            max_gap_hours_clip=strategy.split_time_gap_max_hours,
            macro_gap_ratio_threshold=strategy.macro_gap_ratio_threshold,
            macro_gap_boundary_scale=strategy.macro_gap_boundary_scale,
        )
    return split_groups


def _split_by_tactic(groups: list[list[int]], records: list[AlertRecord]) -> list[list[int]]:
    split: list[list[int]] = []
    for group in groups:
        buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in group:
            record = records[index]
            buckets[(record.tactic, record.technique)].append(index)
        split.extend(buckets.values())
    return split


def _split_by_entity_components(
    groups: list[list[int]],
    records: list[AlertRecord],
    *,
    max_gap_hours: float,
    max_neighbors_per_bucket: int,
) -> list[list[int]]:
    split: list[list[int]] = []
    max_gap_seconds = max_gap_hours * 3600.0
    for group in groups:
        group_set = set(group)
        adjacency: dict[int, set[int]] = {index: set() for index in group_set}
        buckets: dict[str, list[int]] = defaultdict(list)
        for index in group_set:
            for entity_type, value in records[index].entities.items():
                if value:
                    buckets[f"{entity_type}:{value}"].append(index)
        for bucket_indices in buckets.values():
            ordered = sorted(bucket_indices, key=lambda index: (_timestamp_seconds(records[index]), index))
            for offset, left_index in enumerate(ordered):
                left_time = _timestamp_seconds(records[left_index])
                neighbor_count = 0
                for right_index in ordered[offset + 1 :]:
                    right_time = _timestamp_seconds(records[right_index])
                    if np.isfinite(left_time) and np.isfinite(right_time):
                        gap_seconds = right_time - left_time
                        if gap_seconds > max_gap_seconds:
                            break
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)
                    neighbor_count += 1
                    if neighbor_count >= max_neighbors_per_bucket:
                        break
        components = _connected_components(adjacency)
        split.extend(components)
    return split


def _split_by_temporal_gaps(
    groups: list[list[int]],
    records: list[AlertRecord],
    *,
    max_gap_hours: float | None,
    gap_quantile: float | None,
    gap_estimator: str | None,
    min_gap_hours: float,
    max_gap_hours_clip: float,
    macro_gap_ratio_threshold: float,
    macro_gap_boundary_scale: float,
) -> list[list[int]]:
    split: list[list[int]] = []
    for group in groups:
        ordered = sorted(group, key=lambda index: (_timestamp_seconds(records[index]), index))
        max_gap_seconds = _resolve_temporal_gap_seconds(
            ordered,
            records,
            max_gap_hours=max_gap_hours,
            gap_quantile=gap_quantile,
            gap_estimator=gap_estimator,
            min_gap_hours=min_gap_hours,
            max_gap_hours_clip=max_gap_hours_clip,
            macro_gap_ratio_threshold=macro_gap_ratio_threshold,
            macro_gap_boundary_scale=macro_gap_boundary_scale,
        )
        current: list[int] = []
        previous_time: float | None = None
        for index in ordered:
            current_time = _timestamp_seconds(records[index])
            starts_new = False
            if previous_time is not None and np.isfinite(previous_time) and np.isfinite(current_time):
                starts_new = current_time - previous_time > max_gap_seconds
            if starts_new and current:
                split.append(current)
                current = []
            current.append(index)
            previous_time = current_time
        if current:
            split.append(current)
    return split


def _resolve_temporal_gap_seconds(
    ordered_group: list[int],
    records: list[AlertRecord],
    *,
    max_gap_hours: float | None,
    gap_quantile: float | None,
    gap_estimator: str | None,
    min_gap_hours: float,
    max_gap_hours_clip: float,
    macro_gap_ratio_threshold: float,
    macro_gap_boundary_scale: float,
) -> float:
    if max_gap_hours is not None:
        return max_gap_hours * 3600.0
    if gap_estimator is not None:
        if gap_estimator == "macro_elbow":
            return _resolve_macro_gap_elbow_seconds(
                ordered_group,
                records,
                min_gap_hours=min_gap_hours,
                max_gap_hours_clip=max_gap_hours_clip,
                ratio_threshold=macro_gap_ratio_threshold,
                boundary_scale=macro_gap_boundary_scale,
            )
        raise ValueError(f"Unknown temporal gap estimator {gap_estimator!r}")
    if gap_quantile is None:
        return max_gap_hours_clip * 3600.0

    gaps = _positive_gap_hours(ordered_group, records)
    if not gaps:
        return max_gap_hours_clip * 3600.0
    quantile_gap_hours = float(np.quantile(np.array(gaps, dtype=np.float64), gap_quantile))
    clipped_gap_hours = min(max(quantile_gap_hours, min_gap_hours), max_gap_hours_clip)
    return clipped_gap_hours * 3600.0


def _positive_gap_hours(ordered_group: list[int], records: list[AlertRecord]) -> list[float]:
    gaps: list[float] = []
    previous_time: float | None = None
    for index in ordered_group:
        current_time = _timestamp_seconds(records[index])
        if previous_time is not None and np.isfinite(previous_time) and np.isfinite(current_time):
            gap_hours = (current_time - previous_time) / 3600.0
            if gap_hours > 0:
                gaps.append(gap_hours)
        previous_time = current_time
    return gaps


def _resolve_macro_gap_elbow_seconds(
    ordered_group: list[int],
    records: list[AlertRecord],
    *,
    min_gap_hours: float,
    max_gap_hours_clip: float,
    ratio_threshold: float,
    boundary_scale: float,
) -> float:
    gaps = sorted(
        gap
        for gap in _positive_gap_hours(ordered_group, records)
        if min_gap_hours <= gap <= max_gap_hours_clip
    )
    if len(gaps) < 2:
        return max_gap_hours_clip * 3600.0

    gap_array = np.array(gaps, dtype=np.float64)
    ratios = gap_array[1:] / np.clip(gap_array[:-1], 1e-9, None)
    prominent = np.where(ratios >= ratio_threshold)[0]
    boundary_index = int(prominent[0]) if len(prominent) else int(np.argmax(ratios))
    boundary_hours = float(gap_array[boundary_index]) * boundary_scale
    clipped_gap_hours = min(max(boundary_hours, min_gap_hours), max_gap_hours_clip)
    return clipped_gap_hours * 3600.0


def _bayesian_block_groups(
    *,
    selected_indices: np.ndarray,
    records: list[AlertRecord],
    prior: float,
) -> list[list[int]]:
    ordered = [
        int(index)
        for index in sorted(selected_indices.tolist(), key=lambda idx: (_timestamp_seconds(records[int(idx)]), int(idx)))
        if np.isfinite(_timestamp_seconds(records[int(index)]))
    ]
    if len(ordered) <= 1:
        return [ordered]

    times = np.array([_timestamp_seconds(records[index]) for index in ordered], dtype=np.float64)
    times = np.maximum.accumulate(times + np.arange(len(times), dtype=np.float64) * 1e-6)
    cell_edges = np.empty(len(times) + 1, dtype=np.float64)
    cell_edges[1:-1] = 0.5 * (times[1:] + times[:-1])
    cell_edges[0] = times[0] - max((times[1] - times[0]) / 2.0, 1e-6)
    cell_edges[-1] = times[-1] + max((times[-1] - times[-2]) / 2.0, 1e-6)

    best = np.zeros(len(times), dtype=np.float64)
    last = np.zeros(len(times), dtype=np.int64)
    for right in range(len(times)):
        block_lengths = np.clip(cell_edges[right + 1] - cell_edges[: right + 1], 1e-6, None)
        block_counts = np.arange(right + 1, 0, -1, dtype=np.float64)
        fitness = block_counts * (np.log(block_counts) - np.log(block_lengths))
        scores = fitness - prior
        if right > 0:
            scores[1:] += best[:right]
        best_start = int(np.argmax(scores))
        best[right] = scores[best_start]
        last[right] = best_start

    change_points = [len(times)]
    index = len(times)
    while index > 0:
        index = int(last[index - 1])
        change_points.append(index)
    change_points = sorted(set(change_points))
    return [
        ordered[start:end]
        for start, end in zip(change_points[:-1], change_points[1:], strict=True)
        if end > start
    ]


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    components: list[list[int]] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component: list[int] = []
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    return components


def _format_clusters(
    groups: list[list[int]],
    alert_ids: list[str],
    *,
    min_cluster_size: int,
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    incident_id = 0
    noise_alerts: list[str] = []
    for group in sorted(groups, key=lambda indices: (min(indices) if indices else -1, len(indices))):
        group_alert_ids = sorted(alert_ids[index] for index in group)
        if len(group_alert_ids) < min_cluster_size:
            noise_alerts.extend(group_alert_ids)
            continue
        clusters.append({"incident_id": incident_id, "alert_ids": group_alert_ids})
        incident_id += 1
    if noise_alerts:
        clusters.append({"incident_id": -1, "alert_ids": sorted(noise_alerts)})
    return clusters or [{"incident_id": -1, "alert_ids": []}]


def _timestamp_seconds(record: AlertRecord) -> float:
    if record.timestamp is None:
        return float("inf")
    return record.timestamp.timestamp()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
