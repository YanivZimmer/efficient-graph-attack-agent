"""Loader for ExCyTIn-Bench / SecRL incident graph exports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"ExCyTIn data not found at {path}. "
            f"Provide {description} from https://github.com/microsoft/SecRL."
        )


def _load_graphml_like_json(path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    alerts: list[dict] = []
    incidents: dict[str, list[str]] = defaultdict(list)

    if isinstance(payload, dict) and "alerts" in payload:
        alert_rows = payload["alerts"]
        incident_rows = payload.get("incidents", {})
    elif isinstance(payload, list):
        alert_rows = payload
        incident_rows = {}
    else:
        alert_rows = payload.get("nodes", [])
        incident_rows = payload.get("incidents", {})

    for row in alert_rows:
        if not isinstance(row, dict):
            continue
        alert_id = row.get("alert_id") or row.get("SystemAlertId") or row.get("id")
        if not alert_id:
            continue
        alerts.append(
            {
                "alert_id": str(alert_id),
                "label": row.get("label") or row.get("Classification") or row.get("AlertSeverity"),
                "timestamp": row.get("timestamp") or row.get("TimeGenerated"),
                "severity": row.get("severity") or row.get("AlertSeverity"),
                "tactic": row.get("tactic") or row.get("Tactics"),
                "technique": row.get("technique") or row.get("Techniques"),
                "host": row.get("host") or row.get("DeviceName"),
                "user": row.get("user") or row.get("AccountName"),
                "process": row.get("process") or row.get("ProcessName"),
                "ip": row.get("ip") or row.get("RemoteIP") or row.get("SourceIP"),
            }
        )
        incident_id = row.get("incident_id") or row.get("IncidentId")
        if incident_id:
            incidents[str(incident_id)].append(str(alert_id))

    if isinstance(incident_rows, dict):
        for incident_id, alert_ids in incident_rows.items():
            incidents[str(incident_id)].extend(str(alert_id) for alert_id in alert_ids)

    return alerts, dict(incidents)


def load_excytin_graph(data_root: Path) -> AlertGraphArtifacts:
    """Load ExCyTIn incident graph JSON exports."""
    candidate = data_root
    if data_root.is_dir():
        json_files = sorted(data_root.glob("*.json"))
        if not json_files:
            _require_path(data_root / "incident_graph.json", "an incident graph JSON export")
        candidate = json_files[0]
    _require_path(candidate, "an incident graph JSON export")
    alerts, ground_truth = _load_graphml_like_json(candidate)
    return build_graph_from_records(alerts, ground_truth_incidents=ground_truth)
