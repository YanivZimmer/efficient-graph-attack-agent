from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional

from .models import Alert, Entity, EntityType
from .secrl import (
    _cluster_stored_alerts,
    _mean,
    _score_discovered_cluster,
    _verify_incident_groups,
)
from .sketch import GraphSketchingFilter, SketchDecision


TACTIC_KINDS = {
    "initial access": "initial_access",
    "execution": "execution",
    "persistence": "privilege_escalation",
    "privilege escalation": "privilege_escalation",
    "credential access": "credential_access",
    "lateral movement": "lateral_movement",
    "exfiltration": "data_exfiltration",
}

SEVERITY_SCORES = {
    "informational": 2,
    "low": 3,
    "medium": 6,
    "high": 9,
    "critical": 10,
}
LABEL_FIELDS = {"is_incident", "incident_id"}
SEVERITY_FIELD = "severity"
HIDDEN_SEVERITY_SCORE = 5


@dataclass(frozen=True)
class FalconIncidentResult:
    incident: str
    ground_truth_alerts: int
    detected_alerts: int
    alert_recall: float
    any_detected: bool
    all_detected: bool


@dataclass(frozen=True)
class FalconDetectionRow:
    detector: str
    selected_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    incident_recall_any: float
    incident_recall_all: float
    mean_incident_alert_recall: float


@dataclass(frozen=True)
class FalconDiscoveredCluster:
    cluster_id: str
    status: str
    score: float
    alert_count: int
    start_time: str
    end_time: str
    max_severity: int
    attack_graph_pattern: str
    attack_graph_score: float
    attack_graph_edges: list[str]
    known_ground_truth_incidents: list[str]
    decision_reasons: dict[str, int]
    alert_ids: list[str]
    alert_names: list[str]
    rationale: list[str]


@dataclass(frozen=True)
class FalconDiscoverySummary:
    stored_alerts: int
    clusters_considered: int
    clusters_reported: int
    known_overlap_clusters: int
    candidate_new_clusters: int
    candidate_new_alerts: int
    detected_ground_truth_incidents: int
    missed_ground_truth_incidents: int
    incident_recall_any: float
    mean_ground_truth_alert_recall: float


@dataclass(frozen=True)
class FalconEvaluationReport:
    input_path: str
    feature_mode: str
    candidate_alerts: int
    ground_truth_alerts: int
    ground_truth_incidents: int
    alerts_per_second: float
    detection_rows: list[FalconDetectionRow]
    eacs_decision_reasons: dict[str, int]
    eacs_incidents: list[FalconIncidentResult]
    discovery: FalconDiscoverySummary
    discovered_clusters: list[FalconDiscoveredCluster]
    notes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Falcon Graph Alerts Evaluation",
            "",
            f"- Input: `{self.input_path}`",
            f"- Feature mode: `{self.feature_mode}`",
            f"- Candidate alerts: `{self.candidate_alerts}`",
            f"- Ground-truth incident alerts: `{self.ground_truth_alerts}`",
            f"- Ground-truth incidents: `{self.ground_truth_incidents}`",
            f"- Throughput: `{self.alerts_per_second:.1f}` alerts/sec",
            "",
            "## Notes",
            "",
        ]
        lines.extend(f"- {note}" for note in self.notes)
        lines.extend(
            [
                "",
                "## Alert Detection",
                "",
                "| Detector | Selected | TP | FP | FN | Precision | Recall | F1 | Incident Any | Incident All | Mean Incident Alert Recall |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in self.detection_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.detector,
                        str(row.selected_alerts),
                        str(row.true_positives),
                        str(row.false_positives),
                        str(row.false_negatives),
                        f"{row.precision:.3f}",
                        f"{row.recall:.3f}",
                        f"{row.f1:.3f}",
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
                "## E-ACS Decision Reasons",
                "",
                "| Reason | Count |",
                "| --- | ---: |",
            ]
        )
        for reason, count in self.eacs_decision_reasons.items():
            lines.append(f"| {reason} | {count} |")
        lines.extend(
            [
                "",
                "## Incident Discovery",
                "",
                f"- Stored alerts: `{self.discovery.stored_alerts}`",
                f"- Clusters considered: `{self.discovery.clusters_considered}`",
                f"- Clusters reported: `{self.discovery.clusters_reported}`",
                f"- Known-overlap clusters: `{self.discovery.known_overlap_clusters}`",
                f"- Candidate-new clusters: `{self.discovery.candidate_new_clusters}`",
                f"- Candidate-new alerts: `{self.discovery.candidate_new_alerts}`",
                f"- Detected ground-truth incidents: `{self.discovery.detected_ground_truth_incidents}`",
                f"- Missed ground-truth incidents: `{self.discovery.missed_ground_truth_incidents}`",
                f"- Incident recall: `{self.discovery.incident_recall_any:.3f}`",
                f"- Mean ground-truth alert recall: `{self.discovery.mean_ground_truth_alert_recall:.3f}`",
                "",
                "## Top Discovered Clusters",
                "",
                "| Cluster | Status | Alerts | Score | Pattern | Known Incidents | Reasons |",
                "| --- | --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for cluster in self.discovered_clusters[:20]:
            reasons = ", ".join(f"{key}={value}" for key, value in cluster.decision_reasons.items())
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{cluster.cluster_id}`",
                        cluster.status,
                        str(cluster.alert_count),
                        f"{cluster.score:.3f}",
                        cluster.attack_graph_pattern,
                        _escape_table(", ".join(cluster.known_ground_truth_incidents) or "-"),
                        _escape_table(reasons),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## E-ACS Per-Incident Recall",
                "",
                "| Incident | GT Alerts | Detected | Recall |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for incident in self.eacs_incidents:
            lines.append(
                f"| `{incident.incident}` | {incident.ground_truth_alerts} | "
                f"{incident.detected_alerts} | {incident.alert_recall:.3f} |"
            )
        return "\n".join(lines) + "\n"


def alert_from_falcon_row(
    row: dict[str, str],
    *,
    include_mitre_tactics: bool = True,
    hide_severity: bool = False,
) -> Alert:
    source_node = _first_text(row, "source_node", default="falcon-source")
    target_node = _first_text(row, "target_node", default="")
    tactic = _first_text(row, "tactic", default="observed")
    technique = _first_text(row, "technique", default="")
    process = _first_text(row, "process", default="observed")
    kind = TACTIC_KINDS.get(tactic.lower(), _normalize_token(tactic) or "observed") if include_mitre_tactics else "falcon_alert"
    stripped_fields = set(LABEL_FIELDS)
    if not include_mitre_tactics:
        stripped_fields.update({"tactic", "technique"})
    if hide_severity:
        stripped_fields.add(SEVERITY_FIELD)
    raw = {key: value for key, value in row.items() if key not in stripped_fields}
    raw["AlertName"] = " ".join(part for part in ([tactic, technique, process] if include_mitre_tactics else [process]) if part)
    tags = {kind, technique.lower()} if include_mitre_tactics and technique else ({kind} if include_mitre_tactics else set())

    return Alert(
        id=_first_text(row, "alert_id", default=f"falcon:{source_node}:{target_node}:{tactic}"),
        source=Entity(type=EntityType.HOST, value=source_node),
        target=Entity(type=EntityType.HOST, value=target_node) if target_node else None,
        kind=kind,
        action=_normalize_token(process) or _normalize_token(technique) or "observed",
        severity=HIDDEN_SEVERITY_SCORE if hide_severity else _severity_score(_first_text(row, "severity", default="Low")),
        timestamp=_parse_timestamp(_first_text(row, "timestamp", default="")),
        raw=raw,
        tags=tags,
    )


def evaluate_falcon_graph_alerts(
    path: str | Path,
    *,
    include_mitre_tactics: bool = True,
    hide_severity: bool = False,
    min_alerts: int = 2,
    min_score: float = 0.55,
    max_gap_minutes: int = 120,
) -> FalconEvaluationReport:
    input_path = Path(path)
    rows = list(_iter_csv_rows(input_path))
    alerts = [
        alert_from_falcon_row(row, include_mitre_tactics=include_mitre_tactics, hide_severity=hide_severity)
        for row in rows
    ]
    labels = {alert.id: _truthy(row.get("is_incident", "")) for alert, row in zip(alerts, rows)}
    incident_alert_ids = _incident_alert_ids(alerts, rows)
    ground_truth_ids = {alert_id for alert_id, is_incident in labels.items() if is_incident}

    started = perf_counter()
    decisions = _evaluate_eacs(alerts)
    duration = perf_counter() - started
    eacs_ids = {alert_id for alert_id, decision in decisions.items() if decision.store}
    severity_ids = set() if hide_severity else {alert.id for alert in alerts if alert.severity >= 8}
    severity_note = (
        "The High/Critical severity baseline is disabled because severity was hidden from model-visible inputs."
        if hide_severity
        else _severity_baseline_note(alerts, labels)
    )

    detection_rows = [
        _score_detection(
            "E-ACS graph sketch" if include_mitre_tactics else "E-ACS graph sketch without MITRE tactics",
            eacs_ids,
            labels,
            incident_alert_ids,
        ),
        _score_detection("High/Critical severity baseline", severity_ids, labels, incident_alert_ids),
    ]
    discovery, discovered_clusters = _discover_clusters(
        alerts=alerts,
        decisions=decisions,
        incident_alert_ids=incident_alert_ids,
        min_alerts=min_alerts,
        min_score=min_score,
        max_gap_minutes=max_gap_minutes,
    )

    return FalconEvaluationReport(
        input_path=str(input_path),
        feature_mode=_feature_mode(include_mitre_tactics=include_mitre_tactics, hide_severity=hide_severity),
        candidate_alerts=len(alerts),
        ground_truth_alerts=len(ground_truth_ids),
        ground_truth_incidents=len(incident_alert_ids),
        alerts_per_second=len(alerts) / duration if duration else float("inf"),
        detection_rows=detection_rows,
        eacs_decision_reasons=dict(_sorted_counter(Counter(decision.reason for decision in decisions.values()))),
        eacs_incidents=_incident_results(eacs_ids, incident_alert_ids),
        discovery=discovery,
        discovered_clusters=discovered_clusters,
        notes=[
            "`is_incident` and `incident_id` are stripped from `Alert.raw` and used only for post-hoc scoring.",
            (
                "Falcon tactics are mapped to E-ACS attack-stage labels before streaming through the graph sketch."
                if include_mitre_tactics
                else "Falcon `tactic` and `technique` fields are stripped before streaming through the graph sketch."
            ),
            severity_note,
        ],
    )


def _feature_mode(*, include_mitre_tactics: bool, hide_severity: bool) -> str:
    mode = "with_mitre_tactics" if include_mitre_tactics else "without_mitre_tactics"
    return f"{mode}_severity_hidden" if hide_severity else mode


def _evaluate_eacs(alerts: Iterable[Alert]) -> dict[str, SketchDecision]:
    sketch_filter = GraphSketchingFilter()
    decisions: dict[str, SketchDecision] = {}
    for alert in alerts:
        decision = sketch_filter.evaluate(alert)
        decisions[alert.id] = decision
    return decisions


def _score_detection(
    detector: str,
    selected_ids: set[str],
    labels: dict[str, bool],
    incident_alert_ids: dict[str, set[str]],
) -> FalconDetectionRow:
    ground_truth_ids = {alert_id for alert_id, is_incident in labels.items() if is_incident}
    true_positives = len(selected_ids & ground_truth_ids)
    false_positives = len(selected_ids - ground_truth_ids)
    false_negatives = len(ground_truth_ids - selected_ids)
    precision = true_positives / len(selected_ids) if selected_ids else 0.0
    recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    incidents = _incident_results(selected_ids, incident_alert_ids)
    return FalconDetectionRow(
        detector=detector,
        selected_alerts=len(selected_ids),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        incident_recall_any=_mean(1.0 if incident.any_detected else 0.0 for incident in incidents),
        incident_recall_all=_mean(1.0 if incident.all_detected else 0.0 for incident in incidents),
        mean_incident_alert_recall=_mean(incident.alert_recall for incident in incidents),
    )


def _severity_baseline_note(alerts: list[Alert], labels: dict[str, bool]) -> str:
    incident_alerts = [alert for alert in alerts if labels.get(alert.id, False)]
    noise_alerts = [alert for alert in alerts if not labels.get(alert.id, False)]
    incident_high = sum(1 for alert in incident_alerts if alert.severity >= 8)
    noise_high = sum(1 for alert in noise_alerts if alert.severity >= 8)
    incident_rate = incident_high / len(incident_alerts) if incident_alerts else 0.0
    noise_rate = noise_high / len(noise_alerts) if noise_alerts else 0.0
    if incident_rate == 1.0 and noise_rate == 0.0:
        return "The severity baseline is included because this file is severity-separable: all incident alerts are High/Critical and all noise is Medium or lower."
    return (
        "The severity baseline is included as a shortcut check; this file is not severity-separable "
        f"(incident High/Critical rate={incident_rate:.3f}, noise High/Critical rate={noise_rate:.3f})."
    )


def _incident_results(selected_ids: set[str], incident_alert_ids: dict[str, set[str]]) -> list[FalconIncidentResult]:
    results = []
    for incident, alert_ids in sorted(incident_alert_ids.items(), key=_incident_sort_key):
        detected = alert_ids & selected_ids
        results.append(
            FalconIncidentResult(
                incident=incident,
                ground_truth_alerts=len(alert_ids),
                detected_alerts=len(detected),
                alert_recall=len(detected) / len(alert_ids) if alert_ids else 0.0,
                any_detected=bool(detected),
                all_detected=bool(alert_ids) and alert_ids <= selected_ids,
            )
        )
    return results


def _discover_clusters(
    *,
    alerts: list[Alert],
    decisions: dict[str, SketchDecision],
    incident_alert_ids: dict[str, set[str]],
    min_alerts: int,
    min_score: float,
    max_gap_minutes: int,
) -> tuple[FalconDiscoverySummary, list[FalconDiscoveredCluster]]:
    stored_alerts = [alert for alert in alerts if decisions[alert.id].store]
    clusters = _cluster_stored_alerts(stored_alerts, max_gap_minutes=max_gap_minutes)
    group_alert_ids: dict[str, set[str]] = {}
    discovered: list[FalconDiscoveredCluster] = []

    for index, cluster in enumerate(clusters, start=1):
        if len(cluster) < min_alerts:
            continue
        score, rationale, attack_graph_match = _score_discovered_cluster(cluster, decisions)
        if score < min_score:
            continue

        cluster_id = f"falcon-cluster-{index:04d}"
        alert_ids = {alert.id for alert in cluster}
        known_incidents = sorted(
            incident for incident, ground_truth_ids in incident_alert_ids.items() if alert_ids & ground_truth_ids
        )
        group_alert_ids[cluster_id] = alert_ids
        decision_reasons = Counter(decisions[alert.id].reason for alert in cluster)
        timestamps = [alert.timestamp for alert in cluster]
        discovered.append(
            FalconDiscoveredCluster(
                cluster_id=cluster_id,
                status="known_incident_overlap" if known_incidents else "candidate_new_incident",
                score=score,
                alert_count=len(cluster),
                start_time=min(timestamps).isoformat(),
                end_time=max(timestamps).isoformat(),
                max_severity=max(alert.severity for alert in cluster),
                attack_graph_pattern=attack_graph_match.pattern,
                attack_graph_score=attack_graph_match.score,
                attack_graph_edges=list(attack_graph_match.matched_edges),
                known_ground_truth_incidents=known_incidents,
                decision_reasons=dict(_sorted_counter(decision_reasons)),
                alert_ids=sorted(alert_ids),
                alert_names=[str(alert.raw.get("AlertName", alert.action)) for alert in cluster[:5]],
                rationale=rationale,
            )
        )

    matches = _verify_incident_groups(
        incident_alert_ids=incident_alert_ids,
        candidate_ids={alert.id for alert in alerts},
        group_alert_ids=group_alert_ids,
    )
    candidate_new_ids = set().union(
        *(
            set(cluster.alert_ids)
            for cluster in discovered
            if cluster.status == "candidate_new_incident"
        ),
        set(),
    )
    detected_ground_truth_incidents = sum(1 for match in matches if match.any_identified)
    return (
        FalconDiscoverySummary(
            stored_alerts=len(stored_alerts),
            clusters_considered=len(clusters),
            clusters_reported=len(discovered),
            known_overlap_clusters=sum(1 for cluster in discovered if cluster.status == "known_incident_overlap"),
            candidate_new_clusters=sum(1 for cluster in discovered if cluster.status == "candidate_new_incident"),
            candidate_new_alerts=len(candidate_new_ids),
            detected_ground_truth_incidents=detected_ground_truth_incidents,
            missed_ground_truth_incidents=len(matches) - detected_ground_truth_incidents,
            incident_recall_any=_mean(1.0 if match.any_identified else 0.0 for match in matches),
            mean_ground_truth_alert_recall=_mean(match.alert_recall for match in matches),
        ),
        sorted(discovered, key=lambda cluster: (-cluster.score, cluster.start_time, cluster.cluster_id)),
    )


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key): "" if value is None else str(value) for key, value in row.items() if key is not None}


def _incident_alert_ids(alerts: list[Alert], rows: list[dict[str, str]]) -> dict[str, set[str]]:
    incident_alert_ids: dict[str, set[str]] = defaultdict(set)
    for alert, row in zip(alerts, rows):
        incident_id = _first_text(row, "incident_id", default="")
        if incident_id:
            incident_alert_ids[incident_id].add(alert.id)
    return dict(incident_alert_ids)


def _first_text(row: dict[str, str], *names: str, default: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return default


def _severity_score(value: str) -> int:
    return SEVERITY_SCORES.get(value.strip().lower(), 1)


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _normalize_token(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    token = "".join(chars).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _incident_sort_key(value: tuple[str, set[str]]) -> tuple[int, str]:
    incident, _ = value
    suffix = incident.rsplit("-", 1)[-1]
    return (int(suffix), incident) if suffix.isdigit() else (10**9, incident)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate E-ACS on falcon_graph_alerts.csv.")
    parser.add_argument("--input", default="falcon_graph_alerts.csv", help="Path to Falcon graph alerts CSV.")
    parser.add_argument("--output-json", help="Optional JSON report path.")
    parser.add_argument("--output-md", help="Optional Markdown report path.")
    parser.add_argument("--min-alerts", type=int, default=2, help="Minimum alerts in a reported incident cluster.")
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum incident cluster score.")
    parser.add_argument("--max-gap-minutes", type=int, default=120, help="Maximum entity-linked cluster gap.")
    parser.add_argument(
        "--without-mitre-tactics",
        action="store_true",
        help="Strip tactic and technique fields before E-ACS detection.",
    )
    parser.add_argument("--hide-severity", action="store_true", help="Remove severity from model-visible inputs.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_falcon_graph_alerts(
        args.input,
        include_mitre_tactics=not args.without_mitre_tactics,
        hide_severity=args.hide_severity,
        min_alerts=args.min_alerts,
        min_score=args.min_score,
        max_gap_minutes=args.max_gap_minutes,
    )
    if args.output_json:
        _write_json(Path(args.output_json), report.to_json_dict())
    if args.output_md:
        _write_markdown(Path(args.output_md), report.to_markdown())
    if not args.output_json and not args.output_md:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
