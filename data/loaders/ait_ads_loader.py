"""Loader for AIT Alert Data Set (AIT-ADS, Landauer et al., CSET 2024)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"AIT-ADS data not found at {data_root}. "
            "Download from https://zenodo.org/records/8263181 and extract under datasets/ait_ads/."
        )


def _parse_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_label(record: dict) -> tuple[str, str]:
    label_value = record.get("Label") or record.get("label") or record.get("time_label")
    if isinstance(label_value, list):
        label_value = label_value[0] if label_value else ""
    label_text = str(label_value or "").strip().lower()
    if not label_text or label_text in {"benign", "normal", "false positive"}:
        return "benign", ""
    return "malicious", str(label_value).strip()


def _extract_entities(record: dict) -> dict[str, str]:
    entities: dict[str, str] = {}
    aminer = record.get("AMiner")
    if isinstance(aminer, dict):
        host = aminer.get("ID") or aminer.get("Host")
        if isinstance(host, str) and host.strip():
            entities["host"] = host.strip()
    for key in ("src_ip", "source_ip", "SourceIP", "dest_ip", "destination_ip"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            entities.setdefault("ip", value.strip())
    hostname = record.get("hostname") or record.get("Host")
    if isinstance(hostname, str) and hostname.strip():
        entities.setdefault("host", hostname.strip())
    return entities


def _normalize_alert(record: dict, *, source_file: str, index: int) -> dict:
    label, attack_step = _extract_label(record)
    timestamp = _parse_timestamp(
        record.get("timestamp")
        or record.get("@timestamp")
        or record.get("Time")
        or record.get("time")
    )
    alert_id = str(record.get("alert_id") or record.get("id") or f"ait-{source_file}-{index}")
    tactic = attack_step or str(record.get("AnalysisComponentName") or record.get("description") or "<unknown>")
    return {
        "alert_id": alert_id,
        "label": label,
        "timestamp": timestamp,
        "severity": record.get("priority") or record.get("severity") or 0.5,
        "tactic": tactic,
        "technique": attack_step or tactic,
        **(_extract_entities(record)),
    }


def _load_attack_times(data_root: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Load optional attacktimes.csv mapping scenario+step to [start, end]."""
    attack_times_path = data_root / "attacktimes.csv"
    if not attack_times_path.exists():
        return {}
    windows: dict[tuple[str, str], tuple[float, float]] = {}
    with attack_times_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario = (row.get("scenario") or row.get("Scenario") or "").strip()
            attack = (row.get("attack") or row.get("Attack") or row.get("step") or "").strip()
            start = row.get("start") or row.get("Start")
            end = row.get("end") or row.get("End")
            if scenario and attack and start and end:
                windows[(scenario, attack)] = (float(start), float(end))
    return windows


def _load_json_alerts(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("alerts", "records", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported AIT-ADS JSON structure in {path}")


def load_ait_ads_graph(data_root: Path) -> AlertGraphArtifacts:
    """Load AIT-ADS alert JSON exports into graph artifacts."""
    _require_data_root(data_root)
    json_files = sorted(data_root.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No AIT-ADS JSON files found under {data_root}")

    records: list[dict] = []
    ground_truth: dict[str, list[str]] = defaultdict(list)
    for json_path in json_files:
        raw_alerts = _load_json_alerts(json_path)
        for index, raw in enumerate(raw_alerts):
            normalized = _normalize_alert(raw, source_file=json_path.stem, index=index)
            records.append(normalized)
            if normalized["label"] == "malicious":
                step = normalized.get("technique") or normalized.get("tactic") or json_path.stem
                ground_truth[f"{json_path.stem}:{step}"].append(normalized["alert_id"])

    return build_graph_from_records(records, ground_truth_incidents=dict(ground_truth))
