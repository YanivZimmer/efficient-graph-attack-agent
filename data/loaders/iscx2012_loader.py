"""Loader for UNB ISCX IDS 2012 alert-correlation exports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"ISCX 2012 data not found at {data_root}. "
            "Place alert CSV/JSON exports under datasets/iscx2012/."
        )


def _normalize_row(row: dict, *, source: str, index: int) -> dict:
    scenario = (
        row.get("scenario")
        or row.get("attack_scenario")
        or row.get("AttackScenario")
        or row.get("alert_type")
        or "<unknown>"
    )
    label_value = row.get("label") or row.get("Label") or row.get("class") or "malicious"
    label = "benign" if str(label_value).lower() in {"benign", "normal", "legitimate", "0"} else "malicious"
    src_ip = row.get("src_ip") or row.get("source_ip") or row.get("SourceIP")
    dst_ip = row.get("dst_ip") or row.get("destination_ip") or row.get("DestinationIP")
    return {
        "alert_id": str(row.get("alert_id") or row.get("AlertID") or f"iscx2012-{source}-{index}"),
        "label": label,
        "timestamp": row.get("timestamp") or row.get("Time") or row.get("date"),
        "severity": row.get("severity") or row.get("Priority") or 0.6,
        "tactic": str(scenario),
        "technique": str(row.get("signature") or row.get("Signature") or scenario),
        "ip": src_ip or dst_ip,
        "host": row.get("host") or row.get("Host"),
        "user": row.get("user") or row.get("User"),
    }


def _load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append(_normalize_row(row, source=path.stem, index=index))
    return rows


def _load_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, dict):
        raw_rows = payload.get("alerts") or payload.get("records") or []
    else:
        raw_rows = []
    return [_normalize_row(row, source=path.stem, index=index) for index, row in enumerate(raw_rows)]


def load_iscx2012_graph(data_root: Path) -> AlertGraphArtifacts:
    """Load ISCX IDS 2012 alert exports into graph artifacts."""
    _require_data_root(data_root)
    files = sorted(list(data_root.glob("*.csv")) + list(data_root.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No ISCX 2012 CSV/JSON files found under {data_root}")

    records: list[dict] = []
    ground_truth: dict[str, list[str]] = defaultdict(list)
    for path in files:
        loaded = _load_csv(path) if path.suffix.lower() == ".csv" else _load_json(path)
        records.extend(loaded)
        for record in loaded:
            if record["label"] != "malicious":
                continue
            scenario = str(record.get("tactic") or path.stem)
            ground_truth[f"{path.stem}:{scenario}"].append(record["alert_id"])

    return build_graph_from_records(records, ground_truth_incidents=dict(ground_truth))
