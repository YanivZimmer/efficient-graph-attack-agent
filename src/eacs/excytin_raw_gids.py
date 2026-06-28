from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .gids import (
    DetectorEvaluation,
    GIDSEdge,
    GIDSDetector,
    HIDDEN_SEVERITY_SCORE,
    _evaluate_detector,
    _gids_local_verdict_evaluation,
    _run_gids_rare,
)
from .models import Alert, Entity, EntityType
from .secrl import (
    DEFAULT_SECRL_ROOT,
    alert_from_security_alert_row,
    load_incident_alert_ids,
    load_incident_graph_alert_ids,
    _iter_security_alert_rows,
    _scope_folders,
)


KIND_TO_GIDS_TACTIC = {
    "initial_access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege_escalation": "Privilege Escalation",
    "credential_access": "Credential Access",
    "defense_evasion": "Defense Evasion",
    "discovery": "Discovery",
    "lateral_movement": "Lateral Movement",
    "collection": "Collection",
    "command_and_control": "Command And Control",
    "data_exfiltration": "Exfiltration",
    "impact": "Impact",
}
RAW_LABEL_FIELDS = {"IsIncident", "IncidentName", "IncidentNumber", "AlertIds"}


@dataclass(frozen=True)
class ExcytinRawGIDSReport:
    data_root: str
    scope: str
    ground_truth_source: str
    hide_severity: bool
    candidate_alerts: int
    ground_truth_alerts: int
    available_ground_truth_alerts: int
    ground_truth_incidents: int
    gids_pattern_matches: int
    gids_communities_considered: int
    gids_rare_relationships: int
    tactic_counts: dict[str, int]
    detectors: list[DetectorEvaluation]
    notes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# ExCyTIn Raw Alert GIDS Evaluation",
            "",
            f"- Data root: `{self.data_root}`",
            f"- Scope: `{self.scope}`",
            f"- Ground truth: `{self.ground_truth_source}`",
            f"- Severity hidden: `{'yes' if self.hide_severity else 'no'}`",
            f"- Candidate raw alerts: `{self.candidate_alerts}`",
            f"- Ground-truth graph alerts: `{self.ground_truth_alerts}`",
            f"- Available ground-truth alerts in scope: `{self.available_ground_truth_alerts}`",
            f"- Ground-truth incidents in scope: `{self.ground_truth_incidents}`",
            f"- GIDS pattern matches: `{self.gids_pattern_matches}`",
            f"- GIDS communities considered: `{self.gids_communities_considered}`",
            f"- GIDS rare relationships: `{self.gids_rare_relationships}`",
            "",
            "## Notes",
            "",
        ]
        lines.extend(f"- {note}" for note in self.notes)
        lines.extend(
            [
                "",
                "## Detector Comparison",
                "",
                "| Detector | Status | Selected | TP | FP | FN | Precision | Recall | F1 | Clusters | GT Clusters | New Clusters | Incident Any | Incident All | Mean Alert Recall |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in self.detectors:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.detector,
                        row.status,
                        str(row.selected_alerts),
                        str(row.true_positives),
                        str(row.false_positives),
                        str(row.false_negatives),
                        f"{row.precision:.3f}",
                        f"{row.recall:.3f}",
                        f"{row.f1:.3f}",
                        str(row.clusters_reported),
                        str(row.known_overlap_clusters),
                        str(row.candidate_new_clusters),
                        f"{row.incident_recall_any:.3f}",
                        f"{row.incident_recall_all:.3f}",
                        f"{row.mean_incident_alert_recall:.3f}",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Tactic Counts",
                "",
                "| Tactic | Alerts |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {_escape_table(tactic)} | {count} |" for tactic, count in self.tactic_counts.items())
        return "\n".join(lines) + "\n"


def evaluate_excytin_raw_gids(
    data_root: Path,
    *,
    scope: str = "full",
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    limit: Optional[int] = None,
    hide_severity: bool = False,
) -> ExcytinRawGIDSReport:
    raw_rows = list(iter_excytin_raw_alert_rows(data_root, scope=scope, limit=limit))
    edges = [
        edge_from_security_alert_row(row, source_name=source_name, hide_severity=hide_severity)
        for source_name, row in raw_rows
    ]
    candidate_ids = {edge.alert_id for edge in edges}
    ground_truth_ids = _load_ground_truth(data_root, scope, ground_truth, secrl_root)
    available_ground_truth = {
        incident_id: alert_ids & candidate_ids
        for incident_id, alert_ids in ground_truth_ids.items()
        if alert_ids & candidate_ids
    }
    available_positive_ids = set().union(*available_ground_truth.values(), set())
    labels = {edge.alert_id: edge.alert_id in available_positive_ids for edge in edges}

    gids_run = GIDSDetector().run(edges)
    gids_eval = _evaluate_detector(
        detector="GIDS",
        status="ok",
        clusters={incident.incident_id: set(incident.alert_ids) for incident in gids_run.incidents},
        labels=labels,
        ground_truth=available_ground_truth,
        notes=["GIDS structural detector over raw ExCyTIn/SecRL SecurityAlert rows."],
    )
    gids_local_eval, _ = _gids_local_verdict_evaluation(
        gids_run,
        edges,
        labels,
        available_ground_truth,
        use_severity=not hide_severity,
    )
    gids_rare_run, rare_edges = _run_gids_rare(edges)
    gids_rare_eval = _evaluate_detector(
        detector="gids_rare",
        status="ok",
        clusters={incident.incident_id: set(incident.alert_ids) for incident in gids_rare_run.incidents},
        labels=labels,
        ground_truth=available_ground_truth,
        notes=["GIDS with relationship rarity as an additional candidate source and score feature."],
    )
    gids_rare_agent_eval, _ = _gids_local_verdict_evaluation(
        gids_rare_run,
        rare_edges,
        labels,
        available_ground_truth,
        use_severity=not hide_severity,
        use_rarity=True,
        detector_name="gids_rare_with_agent",
    )

    tactic_counts = Counter(edge.tactic or "<none>" for edge in edges)
    return ExcytinRawGIDSReport(
        data_root=str(data_root),
        scope=scope,
        ground_truth_source=ground_truth,
        hide_severity=hide_severity,
        candidate_alerts=len(edges),
        ground_truth_alerts=len(set().union(*ground_truth_ids.values(), set())),
        available_ground_truth_alerts=len(available_positive_ids),
        ground_truth_incidents=len(available_ground_truth),
        gids_pattern_matches=gids_run.pattern_matches,
        gids_communities_considered=gids_run.communities_considered,
        gids_rare_relationships=gids_rare_run.rare_relationships,
        tactic_counts=dict(sorted(tactic_counts.items(), key=lambda item: (-item[1], item[0]))),
        detectors=[
            gids_eval,
            gids_local_eval,
            gids_rare_eval,
            gids_rare_agent_eval,
        ],
        notes=[
            "This evaluation uses raw SecRL/ExCyTIn SecurityAlert rows only; question, answer, and solution fields are not loaded.",
            "Incident graph labels are used only after detector prediction for scoring.",
            "Raw SecurityAlert `IsIncident` and incident identifier fields are stripped from GIDS edge raw metadata.",
            "Recall is computed against ground-truth graph alert IDs that are present in the selected raw-alert scope.",
            (
                f"`AlertSeverity` was hidden and replaced with neutral score {HIDDEN_SEVERITY_SCORE}."
                if hide_severity
                else "`AlertSeverity` was available as a detector feature."
            ),
        ],
    )


def iter_excytin_raw_alert_rows(
    data_root: Path,
    *,
    scope: str = "full",
    limit: Optional[int] = None,
) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[str, dict[str, str]]] = []
    for folder in _scope_folders(data_root, scope):
        for row in _iter_security_alert_rows(folder):
            rows.append((folder.name, row))
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def edge_from_security_alert_row(
    row: dict[str, str],
    *,
    source_name: str,
    hide_severity: bool = False,
) -> GIDSEdge:
    alert = alert_from_security_alert_row(row, source_name)
    source = _entity_value(alert.source)
    target = _entity_value(alert.target) if alert.target is not None else source
    user = _first_entity_value(alert, EntityType.USER)
    return GIDSEdge(
        alert_id=alert.id,
        source_id=source,
        target_id=target,
        user_id=user,
        process_name=_process_name(row, alert),
        tactic=KIND_TO_GIDS_TACTIC.get(alert.kind, _title_from_kind(alert.kind)),
        technique=_first_text(row, "Techniques", "SubTechniques", "AlertType", default=alert.kind),
        severity=HIDDEN_SEVERITY_SCORE if hide_severity else alert.severity,
        timestamp=alert.timestamp,
        raw=_stripped_raw(row, source_name=source_name, hide_severity=hide_severity),
    )


def _load_ground_truth(
    data_root: Path,
    scope: str,
    ground_truth: str,
    secrl_root: Path,
) -> dict[str, set[str]]:
    if ground_truth == "incident-graphs":
        return load_incident_graph_alert_ids(secrl_root)
    if ground_truth == "security-incidents":
        return load_incident_alert_ids(data_root, scope=scope)
    raise ValueError("ground_truth must be 'incident-graphs' or 'security-incidents'")


def _entity_value(entity: Entity | None) -> str:
    if entity is None:
        return ""
    return f"{entity.type.value}:{entity.value}"


def _first_entity_value(alert: Alert, entity_type: EntityType) -> str:
    for entity in alert.entities:
        if entity.type == entity_type:
            return entity.value
    return ""


def _process_name(row: dict[str, str], alert: Alert) -> str:
    process = _process_from_entities(row)
    if process:
        return process
    return _first_text(row, "AlertName", "DisplayName", default=alert.action)


def _process_from_entities(row: dict[str, str]) -> str:
    try:
        entities = json.loads(row.get("Entities", "") or "[]")
    except json.JSONDecodeError:
        return ""
    for item in entities:
        if not isinstance(item, dict) or str(item.get("Type", "")).lower() != "process":
            continue
        for key in ("CommandLine", "ImageFile", "Name", "ProcessId"):
            value = str(item.get(key, "")).strip()
            if value:
                return value[:500]
    return ""


def _stripped_raw(row: dict[str, str], *, source_name: str, hide_severity: bool) -> dict[str, str]:
    stripped = set(RAW_LABEL_FIELDS)
    if hide_severity:
        stripped.update({"AlertSeverity", "Severity"})
    raw = {key: value for key, value in row.items() if key not in stripped}
    raw["_eacs_source_scope"] = source_name
    return raw


def _first_text(row: dict[str, str], *keys: str, default: str) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return default


def _title_from_kind(kind: str) -> str:
    return " ".join(part.capitalize() for part in kind.split("_") if part) or "Observed"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
