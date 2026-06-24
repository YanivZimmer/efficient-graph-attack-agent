"""Loader for DARPA 2000 LLDOS Snort alert exports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


LLDOS_PHASES = (
    "ip_sweep",
    "sadmind_probe",
    "sadmind_exploit",
    "ddos_install",
    "ddos_launch",
)


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"DARPA 2000 data not found at {data_root}. "
            "Place Snort alert CSV/JSON exports under datasets/darpa2000/."
        )


def _normalize_row(row: dict, *, source: str, index: int) -> dict:
    signature = (
        row.get("signature")
        or row.get("Signature")
        or row.get("msg")
        or row.get("alert_type")
        or "<unknown>"
    )
    phase = str(row.get("phase") or row.get("attack_phase") or signature).strip().lower()
    label = "benign" if str(row.get("label", "malicious")).lower() in {"benign", "normal", "0"} else "malicious"
    src_ip = row.get("src_ip") or row.get("source_ip") or row.get("sip")
    dst_ip = row.get("dst_ip") or row.get("destination_ip") or row.get("dip")
    return {
        "alert_id": str(row.get("alert_id") or row.get("sid") or f"darpa2000-{source}-{index}"),
        "label": label,
        "timestamp": row.get("timestamp") or row.get("time"),
        "severity": row.get("priority") or row.get("severity") or 0.7,
        "tactic": phase,
        "technique": str(signature),
        "ip": src_ip or dst_ip,
        "host": row.get("host") or dst_ip,
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


def load_darpa2000_graph(data_root: Path) -> AlertGraphArtifacts:
    """Load DARPA 2000 LLDOS alert exports into graph artifacts."""
    _require_data_root(data_root)
    files = sorted(list(data_root.glob("*.csv")) + list(data_root.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No DARPA 2000 CSV/JSON files found under {data_root}")

    records: list[dict] = []
    ground_truth: dict[str, list[str]] = defaultdict(list)
    for path in files:
        loaded = _load_csv(path) if path.suffix.lower() == ".csv" else _load_json(path)
        records.extend(loaded)
        for record in loaded:
            if record["label"] != "malicious":
                continue
            phase = str(record.get("tactic") or "unknown_phase").lower()
            matched_phase = next((known for known in LLDOS_PHASES if known in phase), phase)
            ground_truth[f"{path.stem}:{matched_phase}"].append(record["alert_id"])

    return build_graph_from_records(records, ground_truth_incidents=dict(ground_truth))
