"""Runtime schema inference for alert JSONL records."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


MALICIOUS_LABEL_TOKENS = ("malicious",)
BENIGN_LABEL_TOKENS = ("benign", "false positive")

ENTITY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "host": ("hostname", "host", "host_name", "computer", "device", "machine"),
    "user": ("username", "user", "user_name", "account", "principal"),
    "process": ("process", "process_name", "filename", "file_name", "commandline", "command_line"),
    "ip": ("localip", "remoteip", "ip", "source_ip", "dest_ip", "destination_ip", "src_ip", "dst_ip"),
}

ALERT_ID_CANDIDATES = ("alert_id", "id", "systemalertid", "alertid")
LABEL_CANDIDATES = ("label", "verdict", "classification", "ground_truth")
TACTIC_CANDIDATES = ("tactic", "mitre_tactic", "attack_tactic")
TECHNIQUE_CANDIDATES = ("technique", "mitre_technique", "attack_technique")
SEVERITY_CANDIDATES = ("severity", "severityname", "severity_name", "risk_score")
TIMESTAMP_CANDIDATES = ("timestamp", "event_time", "eventcreationtime", "time", "created_at")


class InferredSchema(BaseModel):
    """Field mappings inferred from sample records."""

    alert_id_field: str
    label_field: str
    tactic_field: str | None = None
    technique_field: str | None = None
    severity_field: str | None = None
    timestamp_field: str | None = None
    entity_fields: dict[str, str] = Field(default_factory=dict)
    mitre_attack_path: tuple[str, ...] | None = None
    traces_path: tuple[str, ...] | None = None
    raw_event_path: tuple[str, ...] | None = None


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _flatten_paths(record: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Collect leaf values with their nested key paths."""
    paths: dict[tuple[str, ...], Any] = {}
    if isinstance(record, dict):
        for key, value in record.items():
            child_prefix = prefix + (key,)
            if isinstance(value, dict):
                paths.update(_flatten_paths(value, child_prefix))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                paths.update(_flatten_paths(value[0], child_prefix + ("0",)))
            else:
                paths[child_prefix] = value
    return paths


DISFAVORED_PATH_TOKENS = ("feedback", "metadata", "output", "scores", "comment", "verified")
FAVORED_PATH_TOKENS = ("rawpayload", "event", "input", "raw_payload")


def _path_context_adjustment(path: tuple[str, ...]) -> int:
    joined = ".".join(part.lower() for part in path)
    score = 0
    for token in FAVORED_PATH_TOKENS:
        if token in joined:
            score += 12
    for token in DISFAVORED_PATH_TOKENS:
        if token in joined:
            score -= 25
    if len(path) >= 3 and path[0] == "traces" and path[-1].lower() == "timestamp":
        score += 40
    return score


def _score_field(path: tuple[str, ...], candidates: tuple[str, ...]) -> int:
    leaf = _normalize_key(path[-1])
    for index, candidate in enumerate(candidates):
        if leaf == _normalize_key(candidate):
            return 100 - index
    return 0


def _pick_best_path(
    paths: dict[tuple[str, ...], Any],
    candidates: tuple[str, ...],
    *,
    value_predicate: Any | None = None,
    records: list[dict[str, Any]] | None = None,
) -> str | None:
    scored: list[tuple[int, tuple[str, ...]]] = []
    for path in paths:
        if value_predicate is not None and records is not None:
            dotted = ".".join(path)
            sample_values = [get_nested_value(record, dotted) for record in records]
            if not any(value_predicate(value) for value in sample_values):
                continue
        elif value_predicate is not None:
            value = paths[path]
            if not value_predicate(value):
                continue
        score = _score_field(path, candidates) + _path_context_adjustment(path)
        if score > 0:
            scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return ".".join(scored[0][1])


def _pick_entity_fields(records: list[dict[str, Any]]) -> dict[str, str]:
    entity_fields: dict[str, str] = {}
    for entity_type, candidates in ENTITY_CANDIDATES.items():
        merged_paths = {}
        for record in records:
            merged_paths.update(_flatten_paths(record))
        field = _pick_best_path(
            merged_paths,
            candidates,
            value_predicate=lambda value: isinstance(value, str) and value.strip(),
            records=records,
        )
        if field:
            entity_fields[entity_type] = field
    return entity_fields


def _find_mitre_attack_path(record: dict[str, Any]) -> tuple[str, ...] | None:
    paths = _flatten_paths(record)
    for path, value in paths.items():
        if _normalize_key(path[-1]) != "mitreattack":
            continue
        if isinstance(value, (list, dict)):
            return path
    return None


def _find_traces_path(record: dict[str, Any]) -> tuple[str, ...] | None:
    if "traces" in record and isinstance(record["traces"], list):
        return ("traces",)
    return None


def _find_raw_event_path(record: dict[str, Any]) -> tuple[str, ...] | None:
    paths = _flatten_paths(record)
    for path in paths:
        if _normalize_key(path[-1]) == "event" and path[-2].lower() in {"rawpayload", "raw_payload"}:
            return path
    return None


def _counter_key(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def infer_schema(records: list[dict[str, Any]], sample_size: int = 5) -> InferredSchema:
    """Infer alert and entity field names from the first records."""
    if not records:
        raise ValueError("Cannot infer schema from an empty record list")

    sample = records[:sample_size]
    merged_paths: dict[tuple[str, ...], Counter[Any]] = {}
    for record in sample:
        for path, value in _flatten_paths(record).items():
            merged_paths.setdefault(path, Counter())[_counter_key(value)] += 1

    flat_paths = {
        path: next(key for key, _count in counter.most_common(1))
        for path, counter in merged_paths.items()
    }

    alert_id_field = _pick_best_path(
        flat_paths,
        ALERT_ID_CANDIDATES,
        value_predicate=lambda value: isinstance(value, str) and value.strip(),
        records=sample,
    )
    label_field = _pick_best_path(
        flat_paths,
        LABEL_CANDIDATES,
        value_predicate=lambda value: isinstance(value, str) and value.strip(),
        records=sample,
    )
    if not alert_id_field or not label_field:
        raise ValueError("Could not infer alert_id and label fields from sample records")

    tactic_field = _pick_best_path(
        flat_paths,
        TACTIC_CANDIDATES,
        value_predicate=lambda value: isinstance(value, str) and value.strip(),
    )
    technique_field = _pick_best_path(
        flat_paths,
        TECHNIQUE_CANDIDATES,
        value_predicate=lambda value: isinstance(value, str) and value.strip(),
    )
    severity_field = _pick_best_path(flat_paths, SEVERITY_CANDIDATES)
    timestamp_field = _pick_best_path(flat_paths, TIMESTAMP_CANDIDATES)
    if _find_traces_path(sample[0]):
        timestamp_field = "traces.0.timestamp"

    entity_fields = _pick_entity_fields(sample)

    return InferredSchema(
        alert_id_field=alert_id_field,
        label_field=label_field,
        tactic_field=tactic_field,
        technique_field=technique_field,
        severity_field=severity_field,
        timestamp_field=timestamp_field,
        entity_fields=entity_fields,
        mitre_attack_path=_find_mitre_attack_path(sample[0]),
        traces_path=_find_traces_path(sample[0]),
        raw_event_path=_find_raw_event_path(sample[0]),
    )


def get_nested_value(record: dict[str, Any], dotted_path: str | None) -> Any:
    """Read a nested value using a dot-separated path."""
    if not dotted_path:
        return None
    current: Any = record
    for part in dotted_path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                return None
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def parse_timestamp(value: Any) -> datetime | None:
    """Parse common timestamp formats into UTC-aware datetimes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000.0 if value > 1_000_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def label_to_binary(label: Any) -> int | None:
    """Map free-form labels to binary malicious/benign classes."""
    if label is None:
        return None
    text = str(label).strip().lower()
    if any(token in text for token in MALICIOUS_LABEL_TOKENS):
        return 1
    if any(token in text for token in BENIGN_LABEL_TOKENS):
        return 0
    return None


def extract_mitre_fields(record: dict[str, Any], schema: InferredSchema) -> tuple[str | None, str | None]:
    """Extract tactic and technique from direct fields or MitreAttack blocks."""
    tactic = get_nested_value(record, schema.tactic_field)
    technique = get_nested_value(record, schema.technique_field)
    if isinstance(tactic, str) and isinstance(technique, str):
        return tactic, technique

    mitre = get_nested_value(record, ".".join(schema.mitre_attack_path) if schema.mitre_attack_path else None)
    if isinstance(mitre, list) and mitre:
        mitre = mitre[0]
    if isinstance(mitre, dict):
        return mitre.get("Tactic") or mitre.get("tactic"), mitre.get("Technique") or mitre.get("technique")
    return None, None


def severity_to_float(value: Any) -> float:
    """Normalize severity values to a 0-1 range."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return min(max(float(value) / 100.0, 0.0), 1.0)
    text = str(value).strip().lower()
    mapping = {
        "informational": 0.1,
        "low": 0.25,
        "medium": 0.5,
        "high": 0.75,
        "critical": 1.0,
    }
    return mapping.get(text, 0.5)
