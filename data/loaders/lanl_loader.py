"""Loader for LANL Comprehensive Multi-Source Cyber-Security Events."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"LANL data not found at {data_root}. "
            "Download the LANL cyber1 dataset and pass --data-root."
        )


def _parse_day(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value[:10], fmt)
        except ValueError:
            continue
    return None


def _within_window(timestamp: datetime | None, start: datetime, end: datetime) -> bool:
    return timestamp is not None and start <= timestamp <= end


def _load_redteam(data_root: Path, sample_days: int) -> tuple[list[dict], dict[str, list[str]]]:
    redteam_path = next(data_root.glob("*redteam*"), None)
    if redteam_path is None:
        redteam_path = data_root / "redteam.txt"
    _require_data_root(redteam_path)

    records: list[dict] = []
    incidents: dict[str, list[str]] = defaultdict(list)
    timestamps: list[datetime] = []

    with redteam_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            timestamp = _parse_day(row[0]) or datetime.fromtimestamp(int(row[0]))
            timestamps.append(timestamp)
            alert_id = f"lanl-red-{len(records)}"
            session_id = row[3] if len(row) > 3 else "session-0"
            records.append(
                {
                    "alert_id": alert_id,
                    "label": "malicious",
                    "timestamp": timestamp.isoformat(),
                    "severity": "high",
                    "tactic": row[2] if len(row) > 2 else "<unknown>",
                    "technique": row[1] if len(row) > 1 else "<unknown>",
                    "user": row[1] if len(row) > 1 else None,
                    "host": row[2] if len(row) > 2 else None,
                }
            )
            incidents[str(session_id)].append(alert_id)

    if not timestamps:
        return records, dict(incidents)

    start = min(timestamps)
    end = start + timedelta(days=sample_days)
    filtered = [record for record in records if _within_window(_parse_day(str(record["timestamp"])), start, end)]
    filtered_incidents = {
        incident_id: [alert_id for alert_id in alert_ids if alert_id in {row["alert_id"] for row in filtered}]
        for incident_id, alert_ids in incidents.items()
    }
    filtered_incidents = {key: value for key, value in filtered_incidents.items() if value}
    return filtered, filtered_incidents


def load_lanl_graph(data_root: Path, sample_days: int = 5) -> AlertGraphArtifacts:
    """Load a time slice of the LANL dataset into graph artifacts."""
    alerts, ground_truth = _load_redteam(data_root, sample_days=sample_days)
    return build_graph_from_records(alerts, ground_truth_incidents=ground_truth)
