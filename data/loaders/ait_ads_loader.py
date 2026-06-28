"""Loader for AIT Alert Data Set (AIT-ADS, Landauer et al., CSET 2024)."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


DEFAULT_MAX_RECORDS = 10_000


def _require_data_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(
            f"AIT-ADS data not found at {data_root}. "
            "Download from https://zenodo.org/records/8263181 and extract under datasets/ait_ads/."
        )


def _parse_epoch(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return None
    return None


def _epoch_to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _load_label_windows(data_root: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Load attack-step time windows from labels.csv or attacktimes.csv."""
    windows: dict[tuple[str, str], tuple[float, float]] = {}
    for filename in ("labels.csv", "attacktimes.csv"):
        path = data_root / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scenario = (row.get("scenario") or row.get("Scenario") or "").strip()
                attack = (row.get("attack") or row.get("Attack") or row.get("step") or "").strip()
                start = row.get("start") or row.get("Start")
                end = row.get("end") or row.get("End")
                if scenario and attack and start and end:
                    windows[(scenario, attack)] = (float(start), float(end))
    return windows


def _scenario_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def _attack_step_for_timestamp(
    scenario: str,
    epoch: float | None,
    windows: dict[tuple[str, str], tuple[float, float]],
) -> str:
    if epoch is None:
        return ""
    for (window_scenario, attack), (start, end) in windows.items():
        if window_scenario == scenario and start <= epoch <= end:
            return attack
    return ""


def _extract_timestamp_epoch(record: dict) -> float | None:
    for key in ("timestamp", "@timestamp", "Time", "time"):
        epoch = _parse_epoch(record.get(key))
        if epoch is not None:
            return epoch

    log_data = record.get("LogData")
    if isinstance(log_data, dict):
        for key in ("Timestamps", "DetectionTimestamp"):
            values = log_data.get(key)
            if isinstance(values, list) and values:
                epoch = _parse_epoch(values[0])
                if epoch is not None:
                    return epoch
    return None


def _extract_entities(record: dict) -> dict[str, str]:
    entities: dict[str, str] = {}
    aminer = record.get("AMiner")
    if isinstance(aminer, dict):
        host = aminer.get("ID") or aminer.get("Host")
        if isinstance(host, str) and host.strip():
            entities["host"] = host.strip()

    agent = record.get("agent")
    if isinstance(agent, dict):
        ip = agent.get("ip")
        if isinstance(ip, str) and ip.strip():
            entities.setdefault("host", ip.strip())

    predecoder = record.get("predecoder")
    if isinstance(predecoder, dict):
        hostname = predecoder.get("hostname")
        if isinstance(hostname, str) and hostname.strip():
            entities.setdefault("host", hostname.strip())

    for key in ("src_ip", "source_ip", "SourceIP", "dest_ip", "destination_ip"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            entities.setdefault("ip", value.strip())
    return entities


def _extract_severity(record: dict) -> float:
    rule = record.get("rule")
    if isinstance(rule, dict) and rule.get("level") is not None:
        return min(float(rule["level"]) / 15.0, 1.0)
    return 0.5


def _extract_signature(record: dict) -> str:
    rule = record.get("rule")
    if isinstance(rule, dict) and rule.get("description"):
        return str(rule["description"])
    analysis = record.get("AnalysisComponent")
    if isinstance(analysis, dict):
        return str(analysis.get("AnalysisComponentName") or analysis.get("Message") or "<unknown>")
    return "<unknown>"


def _normalize_alert(
    record: dict,
    *,
    source_file: str,
    index: int,
    scenario: str,
    label_windows: dict[tuple[str, str], tuple[float, float]],
) -> dict | None:
    epoch = _extract_timestamp_epoch(record)
    attack_step = _attack_step_for_timestamp(scenario, epoch, label_windows)
    label_value = record.get("Label") or record.get("label")
    if isinstance(label_value, list):
        label_value = label_value[0] if label_value else ""
    if label_value:
        label_text = str(label_value).strip().lower()
        label = "benign" if label_text in {"benign", "normal", "false positive", ""} else "malicious"
        attack_step = attack_step or str(label_value).strip()
    else:
        label = "malicious" if attack_step else "benign"

    alert_id = str(record.get("alert_id") or record.get("id") or f"ait-{source_file}-{index}")
    signature = _extract_signature(record)
    return {
        "alert_id": alert_id,
        "label": label,
        "timestamp": _epoch_to_iso(epoch) if epoch is not None else None,
        "severity": _extract_severity(record),
        "tactic": attack_step or signature,
        "technique": attack_step or signature,
        **(_extract_entities(record)),
        "_scenario": scenario,
        "_attack_step": attack_step,
    }


def _load_json_alerts(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid NDJSON in {path} at line {line_number}") from exc
        return records

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("alerts", "records", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported AIT-ADS JSON structure in {path}")


def _subsample_records(
    records: list[dict],
    *,
    max_records: int,
    random_state: int,
) -> list[dict]:
    if len(records) <= max_records:
        return records
    malicious = [record for record in records if record["label"] == "malicious"]
    benign = [record for record in records if record["label"] != "malicious"]
    rng = random.Random(random_state)

    if not benign:
        return rng.sample(malicious, max_records)

    max_malicious = min(len(malicious), int(max_records * 0.85))
    max_benign = max_records - max_malicious
    sampled_malicious = rng.sample(malicious, max_malicious) if len(malicious) > max_malicious else malicious
    sampled_benign = rng.sample(benign, min(max_benign, len(benign)))
    return sampled_malicious + sampled_benign


def _build_ground_truth(records: list[dict]) -> dict[str, list[str]]:
    ground_truth: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["label"] != "malicious":
            continue
        attack_step = record.get("_attack_step") or record.get("technique") or record.get("tactic")
        scenario = record.get("_scenario") or "unknown"
        if not attack_step:
            continue
        ground_truth[f"{scenario}:{attack_step}"].append(record["alert_id"])
    return dict(ground_truth)


def _strip_internal_fields(records: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for record in records:
        cleaned.append({key: value for key, value in record.items() if not key.startswith("_")})
    return cleaned


def load_ait_ads_graph(data_root: Path, *, max_records: int = DEFAULT_MAX_RECORDS) -> AlertGraphArtifacts:
    """Load AIT-ADS alert JSON/NDJSON exports into graph artifacts."""
    _require_data_root(data_root)
    json_files = sorted(path for path in data_root.glob("*.json") if path.name != "package.json")
    if not json_files:
        raise FileNotFoundError(f"No AIT-ADS JSON files found under {data_root}")

    label_windows = _load_label_windows(data_root)
    records: list[dict] = []
    for json_path in json_files:
        scenario = _scenario_from_filename(json_path)
        raw_alerts = _load_json_alerts(json_path)
        for index, raw in enumerate(raw_alerts):
            normalized = _normalize_alert(
                raw,
                source_file=json_path.stem,
                index=index,
                scenario=scenario,
                label_windows=label_windows,
            )
            if normalized is not None:
                records.append(normalized)

    records = _subsample_records(records, max_records=max_records, random_state=42)
    ground_truth = _build_ground_truth(records)
    return build_graph_from_records(_strip_internal_fields(records), ground_truth_incidents=ground_truth)
