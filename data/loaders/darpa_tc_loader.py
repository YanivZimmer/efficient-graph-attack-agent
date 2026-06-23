"""Loader for DARPA Transparent Computing provenance datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"DARPA TC data not found at {data_root}. "
            "Download E3/E5 CADETS, THEIA, or TRACE parsed CDM JSON and pass --data-root."
        )


def _load_json_records(data_root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(data_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            records.append(payload)
    return records


def _normalize_darpa_record(raw: dict) -> dict:
    """Map common DARPA CDM fields into the generic alert schema."""
    node_type = str(raw.get("type") or raw.get("objectType") or raw.get("subject_type") or "").lower()
    label = raw.get("label") or raw.get("malicious") or raw.get("ground_truth")
    if isinstance(label, bool):
        label = "malicious" if label else "benign"
    elif label is None:
        label = "benign"

    record = {
        "alert_id": str(raw.get("uuid") or raw.get("id") or raw.get("event_id")),
        "label": label,
        "timestamp": raw.get("timestamp") or raw.get("time"),
        "severity": raw.get("severity") or raw.get("score") or 0.5,
        "tactic": raw.get("tactic") or raw.get("attack_stage") or "<unknown>",
        "technique": raw.get("technique") or raw.get("operation") or "<unknown>",
    }

    if "process" in node_type or raw.get("process_name"):
        record["process"] = raw.get("process_name") or raw.get("name") or raw.get("path")
    if "file" in node_type or raw.get("path"):
        record["host"] = raw.get("host") or raw.get("hostname") or raw.get("path")
    if "socket" in node_type or raw.get("remote_ip") or raw.get("remoteAddress"):
        record["ip"] = raw.get("remote_ip") or raw.get("remoteAddress") or raw.get("localAddress")
    if raw.get("user"):
        record["user"] = raw["user"]
    return record


def _build_ground_truth(records: list[dict]) -> dict[str, list[str]]:
    incidents: dict[str, list[str]] = defaultdict(list)
    for record in records:
        incident_id = record.get("incident_id") or record.get("attack_id") or record.get("campaign_id")
        alert_id = record.get("uuid") or record.get("id") or record.get("event_id")
        if incident_id and alert_id:
            incidents[str(incident_id)].append(str(alert_id))
    return dict(incidents)


def load_darpa_tc_graph(data_root: Path) -> AlertGraphArtifacts:
    """Load a DARPA TC dataset directory into graph artifacts."""
    _require_data_root(data_root)
    raw_records = _load_json_records(data_root)
    normalized = [_normalize_darpa_record(record) for record in raw_records]
    ground_truth = _build_ground_truth(raw_records)
    return build_graph_from_records(normalized, ground_truth_incidents=ground_truth)
