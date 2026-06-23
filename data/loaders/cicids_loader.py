"""Loader for CIC-IDS 2017 flow CSV exports."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"CIC-IDS 2017 data not found at {data_root}. "
            "Download the CSV exports and pass --data-root."
        )


def _attack_day_label(path: Path) -> str:
    name = path.stem.lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if day in name:
            return day
    return path.stem


def _load_csv(path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    records: list[dict] = []
    incidents: dict[str, list[str]] = defaultdict(list)
    day_label = _attack_day_label(path)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            label_value = row.get("Label") or row.get("label") or "BENIGN"
            label = "benign" if str(label_value).upper() == "BENIGN" else "malicious"
            src_ip = row.get("Source IP") or row.get("Src IP") or row.get("source_ip")
            dst_ip = row.get("Destination IP") or row.get("Dst IP") or row.get("destination_ip")
            alert_id = f"cicids-{path.stem}-{index}"
            records.append(
                {
                    "alert_id": alert_id,
                    "label": label,
                    "severity": row.get("Flow Duration") or 0.5,
                    "tactic": str(label_value),
                    "technique": str(label_value),
                    "ip": src_ip or dst_ip,
                }
            )
            if label == "malicious":
                incidents[f"{day_label}:{label_value}"].append(alert_id)

    return records, dict(incidents)


def load_cicids_graph(data_root: Path) -> AlertGraphArtifacts:
    """Load one or more CIC-IDS 2017 CSV files into graph artifacts."""
    _require_data_root(data_root)
    csv_files = sorted(data_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_root}")

    all_records: list[dict] = []
    ground_truth: dict[str, list[str]] = {}
    for csv_path in csv_files:
        records, incidents = _load_csv(csv_path)
        all_records.extend(records)
        ground_truth.update(incidents)

    return build_graph_from_records(all_records, ground_truth_incidents=ground_truth)
