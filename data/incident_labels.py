"""Map ground-truth incident groupings to per-alert supervision tensors."""

from __future__ import annotations

import torch

from data.graph_builder import AlertGraphArtifacts


def build_incident_label_tensor(artifacts: AlertGraphArtifacts) -> torch.Tensor:
    """Return per-alert incident class indices; ``-1`` when unknown."""
    if not artifacts.ground_truth_incidents:
        return torch.full((len(artifacts.alert_ids),), -1, dtype=torch.long)

    alert_to_class: dict[str, int] = {}
    for class_index, alert_ids in enumerate(sorted(artifacts.ground_truth_incidents.values(), key=len, reverse=True)):
        for alert_id in alert_ids:
            alert_to_class[alert_id] = class_index

    return torch.tensor(
        [alert_to_class.get(alert_id, -1) for alert_id in artifacts.alert_ids],
        dtype=torch.long,
    )


def incident_class_count(artifacts: AlertGraphArtifacts) -> int:
    """Count distinct ground-truth incident groups."""
    return len(artifacts.ground_truth_incidents)
