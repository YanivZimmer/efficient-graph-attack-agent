from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from .models import Alert, Entity, EntityType
from .ports import AlertStream
from .sketch import GraphSketchingFilter, StreamProcessor
from .graph import InMemoryGraphStore


DATASET_NAME = "anandmudgerikar/excytin-bench"
RAW_ARCHIVE_NAME = "data_anonymized.tar.gz"
DEFAULT_SECRL_ROOT = Path.home() / "Code" / "Datasets" / "SecRL"
SEPARATOR = "\u2756"
INCIDENT_IDS = ("5", "34", "38", "39", "55", "134", "166", "322")
UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
STRICT_ENTITY_CONTINUITY = "strict_entity_continuity"
SUPPRESS_GENERIC_ICS = "suppress_generic_ics"
REDUCE_C2_WEIGHT = "reduce_c2_weight"
REQUIRE_PROGRESSION_OR_SEVERITY = "require_progression_or_severity_density"
DISCOVERY_REFINEMENTS = frozenset(
    {
        STRICT_ENTITY_CONTINUITY,
        SUPPRESS_GENERIC_ICS,
        REDUCE_C2_WEIGHT,
        REQUIRE_PROGRESSION_OR_SEVERITY,
    }
)
DISCOVERY_BASELINES = (
    "eacs_baseline",
    "eacs_refined",
    "high_severity_only",
    "attack_keyword_only",
    "entity_time_cluster_all_alerts",
    "vendor_security_incident",
    "random_top_k",
    "graph_oracle",
)


@dataclass(frozen=True)
class SecRLIncidentResult:
    incident: str
    ground_truth_alerts: int
    available_ground_truth_alerts: int
    detected_ground_truth_alerts: int
    alert_recall: float
    available_recall: float
    any_detected: bool
    all_detected: bool


@dataclass(frozen=True)
class SecRLDetectionReport:
    scope: str
    ground_truth_source: str
    rows_read: int
    candidate_alerts: int
    ground_truth_alerts: int
    available_ground_truth_alerts: int
    detected_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    available_recall: float
    f1: float
    incident_count: int
    incident_recall_any: float
    incident_recall_all: float
    mean_incident_alert_recall: float
    alerts_per_second: float
    notes: list[str]
    incidents: list[SecRLIncidentResult]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SecRLAlertExample:
    alert_id: str
    alert_name: str
    severity: int
    kind: str
    tags: list[str]
    decision_reason: str
    incident_refs: list[str]
    source_scope: str


@dataclass(frozen=True)
class AttackGraphPattern:
    name: str
    stages: tuple[str, ...]
    edges: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class AttackGraphMatch:
    pattern: str
    score: float
    matched_stages: tuple[str, ...]
    matched_edges: tuple[str, ...]
    observed_stages: tuple[str, ...]
    observed_edges: tuple[str, ...]


DEFAULT_ATTACK_GRAPH_PATTERNS = (
    AttackGraphPattern(
        name="credential_lateral_privilege",
        stages=("credential_access", "execution", "lateral_movement", "privilege_escalation"),
        edges=frozenset(
            {
                ("credential_access", "execution"),
                ("credential_access", "lateral_movement"),
                ("execution", "lateral_movement"),
                ("lateral_movement", "privilege_escalation"),
            }
        ),
    ),
    AttackGraphPattern(
        name="c2_to_lateral_movement",
        stages=("command_and_control", "execution", "lateral_movement"),
        edges=frozenset(
            {
                ("command_and_control", "execution"),
                ("command_and_control", "lateral_movement"),
                ("execution", "lateral_movement"),
            }
        ),
    ),
    AttackGraphPattern(
        name="cloud_privilege_exfiltration",
        stages=("credential_access", "privilege_escalation", "collection", "data_exfiltration"),
        edges=frozenset(
            {
                ("credential_access", "privilege_escalation"),
                ("privilege_escalation", "collection"),
                ("collection", "data_exfiltration"),
                ("privilege_escalation", "data_exfiltration"),
            }
        ),
    ),
    AttackGraphPattern(
        name="ransomware_impact_chain",
        stages=("command_and_control", "execution", "lateral_movement", "impact"),
        edges=frozenset(
            {
                ("command_and_control", "execution"),
                ("execution", "lateral_movement"),
                ("lateral_movement", "impact"),
                ("execution", "impact"),
            }
        ),
    ),
    AttackGraphPattern(
        name="credential_to_exfiltration",
        stages=("credential_access", "collection", "data_exfiltration"),
        edges=frozenset(
            {
                ("credential_access", "collection"),
                ("collection", "data_exfiltration"),
                ("credential_access", "data_exfiltration"),
            }
        ),
    ),
)
EMPTY_ATTACK_GRAPH_MATCH = AttackGraphMatch(
    pattern="none",
    score=0.0,
    matched_stages=(),
    matched_edges=(),
    observed_stages=(),
    observed_edges=(),
)


@dataclass(frozen=True)
class SecRLIncidentErrorAnalysis:
    incident: str
    ground_truth_alerts: int
    available_ground_truth_alerts: int
    detected_ground_truth_alerts: int
    alert_recall: float
    available_recall: float
    missing_unavailable: int
    missing_filtered: int
    miss_reasons: dict[str, int]
    missed_examples: list[SecRLAlertExample]


@dataclass(frozen=True)
class SecRLErrorAnalysisReport:
    scope: str
    ground_truth_source: str
    candidate_alerts: int
    detected_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    false_positives_with_security_incident_ref: int
    false_positive_decision_reasons: dict[str, int]
    false_positive_kinds: dict[str, int]
    false_positive_tags: dict[str, int]
    false_positive_severities: dict[str, int]
    top_false_positive_alert_names: dict[str, int]
    top_true_positive_alert_names: dict[str, int]
    incident_analyses: list[SecRLIncidentErrorAnalysis]
    false_positive_examples: list[SecRLAlertExample]
    conclusions: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# SecRL Error Analysis",
            "",
            f"- Scope: `{self.scope}`",
            f"- Ground truth: `{self.ground_truth_source}`",
            f"- Candidate alerts: `{self.candidate_alerts}`",
            f"- Detected alerts: `{self.detected_alerts}`",
            f"- True positives: `{self.true_positives}`",
            f"- False positives: `{self.false_positives}`",
            f"- False negatives: `{self.false_negatives}`",
            f"- False positives referenced by raw `SecurityIncident.AlertIds`: `{self.false_positives_with_security_incident_ref}`",
            "",
            "## Conclusions",
            "",
        ]
        lines.extend(f"- {item}" for item in self.conclusions)
        lines.extend(
            [
                "",
                "## False Positive Reasons",
                "",
                "| Reason | Count |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {key} | {value} |" for key, value in self.false_positive_decision_reasons.items())
        lines.extend(
            [
                "",
                "## False Positive Kinds",
                "",
                "| Kind | Count |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {key} | {value} |" for key, value in self.false_positive_kinds.items())
        lines.extend(
            [
                "",
                "## Top False Positive Alert Names",
                "",
                "| Alert | Count |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {_escape_table(key)} | {value} |" for key, value in self.top_false_positive_alert_names.items())
        lines.extend(
            [
                "",
                "## Per-Incident Miss Analysis",
                "",
                "| Incident | Ground Truth | Available | Detected | Unavailable Misses | Filtered Misses | Miss Reasons |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in self.incident_analyses:
            reasons = ", ".join(f"{key}: {value}" for key, value in item.miss_reasons.items()) or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        item.incident,
                        str(item.ground_truth_alerts),
                        str(item.available_ground_truth_alerts),
                        str(item.detected_ground_truth_alerts),
                        str(item.missing_unavailable),
                        str(item.missing_filtered),
                        _escape_table(reasons),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## False Positive Examples",
                "",
                "| Alert ID | Name | Severity | Kind | Reason | Raw Incident Refs |",
                "| --- | --- | ---: | --- | --- | --- |",
            ]
        )
        lines.extend(_example_row(example) for example in self.false_positive_examples)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SecRLDiscoveredIncident:
    cluster_id: str
    status: str
    score: float
    alert_count: int
    start_time: str
    end_time: str
    max_severity: int
    attack_tags: list[str]
    attack_graph_pattern: str
    attack_graph_score: float
    attack_graph_edges: list[str]
    entity_count: int
    decision_reasons: dict[str, int]
    known_ground_truth_incidents: list[str]
    raw_security_incident_refs: list[str]
    alert_ids: list[str]
    alert_names: list[str]
    rationale: list[str]
    examples: list[SecRLAlertExample]


@dataclass(frozen=True)
class SecRLGroundTruthIncidentMatch:
    incident: str
    ground_truth_alerts: int
    available_ground_truth_alerts: int
    recovered_alerts: int
    alert_recall: float
    available_recall: float
    matched_clusters: list[str]
    best_cluster_id: str
    best_cluster_overlap: int
    any_identified: bool
    all_available_recovered: bool


@dataclass(frozen=True)
class SecRLIncidentDiscoveryReport:
    scope: str
    ground_truth_source: str
    refinements: list[str]
    candidate_alerts: int
    stored_alerts: int
    ground_truth_incidents: int
    detected_ground_truth_incidents: int
    missed_ground_truth_incidents: int
    incident_recall_any: float
    mean_ground_truth_alert_recall: float
    mean_available_alert_recall: float
    clusters_considered: int
    clusters_reported: int
    known_overlap_clusters: int
    candidate_new_incidents: int
    candidate_new_alerts: int
    min_score: float
    min_alerts: int
    max_gap_minutes: int
    incidents: list[SecRLDiscoveredIncident]
    ground_truth_matches: list[SecRLGroundTruthIncidentMatch]
    notes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# SecRL Incident Discovery",
            "",
            f"- Scope: `{self.scope}`",
            f"- Ground truth: `{self.ground_truth_source}`",
            f"- Refinements: `{', '.join(self.refinements) or 'baseline'}`",
            f"- Candidate alerts: `{self.candidate_alerts}`",
            f"- Stored alerts: `{self.stored_alerts}`",
            f"- Ground-truth incidents: `{self.ground_truth_incidents}`",
            f"- Detected ground-truth incidents: `{self.detected_ground_truth_incidents}`",
            f"- Missed ground-truth incidents: `{self.missed_ground_truth_incidents}`",
            f"- Incident recall any: `{self.incident_recall_any:.3f}`",
            f"- Mean ground-truth alert recall: `{self.mean_ground_truth_alert_recall:.3f}`",
            f"- Mean available alert recall: `{self.mean_available_alert_recall:.3f}`",
            f"- Clusters considered: `{self.clusters_considered}`",
            f"- Clusters reported: `{self.clusters_reported}`",
            f"- Known-overlap clusters: `{self.known_overlap_clusters}`",
            f"- Candidate-new incidents: `{self.candidate_new_incidents}`",
            f"- Candidate-new alerts: `{self.candidate_new_alerts}`",
            f"- Minimum score: `{self.min_score}`",
            f"- Minimum alerts: `{self.min_alerts}`",
            f"- Max entity gap minutes: `{self.max_gap_minutes}`",
            "",
            "## Notes",
            "",
        ]
        lines.extend(f"- {note}" for note in self.notes)
        lines.extend(
            [
                "",
                "## Ground-Truth Verification",
                "",
                "| Incident | GT Alerts | Available | Recovered | Recall | Available Recall | Matched Clusters |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for match in self.ground_truth_matches:
            lines.append(
                "| "
                + " | ".join(
                    [
                        match.incident,
                        str(match.ground_truth_alerts),
                        str(match.available_ground_truth_alerts),
                        str(match.recovered_alerts),
                        f"{match.alert_recall:.3f}",
                        f"{match.available_recall:.3f}",
                        _escape_table(", ".join(match.matched_clusters) or "-"),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Reported Clusters",
                "",
                "| Cluster | Status | Score | Alerts | Max Severity | Tags | Attack Graph | Known Overlap | Raw Incident Refs |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for incident in self.incidents:
            lines.append(
                "| "
                + " | ".join(
                    [
                        incident.cluster_id,
                        incident.status,
                        f"{incident.score:.3f}",
                        str(incident.alert_count),
                        str(incident.max_severity),
                        _escape_table(", ".join(incident.attack_tags) or "-"),
                        _escape_table(
                            f"{incident.attack_graph_pattern} ({incident.attack_graph_score:.3f})"
                            if incident.attack_graph_pattern != "none"
                            else "-"
                        ),
                        _escape_table(", ".join(incident.known_ground_truth_incidents) or "-"),
                        _escape_table(", ".join(incident.raw_security_incident_refs) or "-"),
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SecRLDiscoveryAblationRow:
    variant: str
    refinements: list[str]
    clusters_reported: int
    known_overlap_clusters: int
    candidate_new_incidents: int
    candidate_new_alerts: int
    detected_ground_truth_incidents: int
    incident_recall_any: float
    mean_ground_truth_alert_recall: float
    mean_available_alert_recall: float
    candidate_new_reduction: int
    candidate_new_alert_reduction: int
    incident_recall_delta: float


@dataclass(frozen=True)
class SecRLDiscoveryAblationReport:
    scope: str
    ground_truth_source: str
    baseline_candidate_new_incidents: int
    baseline_candidate_new_alerts: int
    baseline_incident_recall_any: float
    rows: list[SecRLDiscoveryAblationRow]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        single_rows = [row for row in self.rows if row.variant.startswith("only_")]
        best = sorted(
            single_rows,
            key=lambda row: (row.incident_recall_delta >= 0, row.candidate_new_alert_reduction, row.candidate_new_reduction),
            reverse=True,
        )
        lines = [
            "# SecRL Discovery Ablation",
            "",
            f"- Scope: `{self.scope}`",
            f"- Ground truth: `{self.ground_truth_source}`",
            f"- Baseline candidate-new incidents: `{self.baseline_candidate_new_incidents}`",
            f"- Baseline candidate-new alerts: `{self.baseline_candidate_new_alerts}`",
            f"- Baseline incident recall: `{self.baseline_incident_recall_any:.3f}`",
            "",
            "## What Helped Most",
            "",
        ]
        if best:
            for row in best:
                lines.append(
                    f"- `{row.variant}` reduced candidate-new alerts by `{row.candidate_new_alert_reduction}` "
                    f"and clusters by `{row.candidate_new_reduction}` "
                    f"with incident recall delta `{row.incident_recall_delta:.3f}`."
                )
        else:
            lines.append("- No single-refinement rows were generated.")
        lines.extend(
            [
                "",
                "## Ablation Table",
                "",
                "| Variant | Refinements | Candidate-New | Candidate-New Alerts | Delta Alerts | Incident Recall | Mean GT Alert Recall | Reported Clusters |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in self.rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.variant,
                        _escape_table(", ".join(row.refinements) or "baseline"),
                        str(row.candidate_new_incidents),
                        str(row.candidate_new_alerts),
                        str(row.candidate_new_alert_reduction),
                        f"{row.incident_recall_any:.3f}",
                        f"{row.mean_ground_truth_alert_recall:.3f}",
                        str(row.clusters_reported),
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SecRLDiscoveryBaselineRow:
    baseline: str
    selection: str
    uses_ground_truth: bool
    candidate_alerts: int
    selected_alerts: int
    alert_precision: float
    alert_recall: float
    alert_available_recall: float
    alert_f1: float
    clusters_reported: int
    gt_overlap_clusters: int
    non_gt_clusters: int
    candidate_new_incidents: int
    candidate_new_alerts: int
    detected_ground_truth_incidents: int
    incident_recall_any: float
    mean_ground_truth_alert_recall: float
    mean_available_alert_recall: float


@dataclass(frozen=True)
class SecRLDiscoveryBaselineReport:
    scope: str
    ground_truth_source: str
    candidate_alerts: int
    ground_truth_alerts: int
    available_ground_truth_alerts: int
    max_gap_minutes: int
    min_alerts: int
    min_score: float
    rows: list[SecRLDiscoveryBaselineRow]
    notes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# SecRL Discovery Baseline Comparison",
            "",
            f"- Scope: `{self.scope}`",
            f"- Ground truth: `{self.ground_truth_source}`",
            f"- Candidate alerts: `{self.candidate_alerts}`",
            f"- Ground-truth alerts: `{self.ground_truth_alerts}`",
            f"- Available ground-truth alerts: `{self.available_ground_truth_alerts}`",
            f"- Minimum score: `{self.min_score}`",
            f"- Minimum alerts: `{self.min_alerts}`",
            f"- Max entity gap minutes: `{self.max_gap_minutes}`",
            "",
            "## Notes",
            "",
        ]
        lines.extend(f"- {note}" for note in self.notes)
        lines.extend(
            [
                "",
                "## Baseline Table",
                "",
                "| Baseline | Selected Alerts | Alert P/R/F1 | Incident Recall | Mean GT Alert Recall | GT Clusters | Non-GT Clusters | Candidate-New | Uses GT |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in self.rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.baseline,
                        str(row.selected_alerts),
                        f"{row.alert_precision:.3f}/{row.alert_recall:.3f}/{row.alert_f1:.3f}",
                        f"{row.incident_recall_any:.3f}",
                        f"{row.mean_ground_truth_alert_recall:.3f}",
                        str(row.gt_overlap_clusters),
                        str(row.non_gt_clusters),
                        f"{row.candidate_new_incidents} ({row.candidate_new_alerts} alerts)",
                        "yes" if row.uses_ground_truth else "no",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Baseline Definitions",
                "",
            ]
        )
        for row in self.rows:
            lines.append(f"- `{row.baseline}`: {row.selection}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SecRLLeakageAuditCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SecRLLeakageAuditReport:
    scope: str
    ground_truth_source: str
    refinements: list[str]
    candidate_alerts: int
    stored_alerts: int
    normal_reported_clusters: int
    blind_reported_clusters: int
    normal_candidate_new_incidents: int
    blind_candidate_incidents: int
    ground_truth_labeled_clusters: int
    raw_incident_labeled_clusters: int
    posthoc_label_delta: int
    cluster_generation_stable: bool
    score_generation_stable: bool
    potential_leakage_detected: bool
    checks: list[SecRLLeakageAuditCheck]
    notes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# SecRL Discovery Leakage Audit",
            "",
            f"- Scope: `{self.scope}`",
            f"- Ground truth: `{self.ground_truth_source}`",
            f"- Refinements: `{', '.join(self.refinements) or 'baseline'}`",
            f"- Candidate alerts: `{self.candidate_alerts}`",
            f"- Stored alerts: `{self.stored_alerts}`",
            f"- Normal reported clusters: `{self.normal_reported_clusters}`",
            f"- Blind reported clusters: `{self.blind_reported_clusters}`",
            f"- Normal candidate-new incidents: `{self.normal_candidate_new_incidents}`",
            f"- Blind candidate incidents: `{self.blind_candidate_incidents}`",
            f"- Ground-truth-labeled clusters: `{self.ground_truth_labeled_clusters}`",
            f"- Raw-incident-labeled clusters: `{self.raw_incident_labeled_clusters}`",
            f"- Post-hoc label delta: `{self.posthoc_label_delta}`",
            f"- Cluster generation stable: `{self.cluster_generation_stable}`",
            f"- Score generation stable: `{self.score_generation_stable}`",
            f"- Potential leakage detected: `{self.potential_leakage_detected}`",
            "",
            "## Checks",
            "",
            "| Check | Passed | Detail |",
            "| --- | --- | --- |",
        ]
        for check in self.checks:
            lines.append(
                "| "
                + " | ".join(
                    [
                        check.name,
                        "yes" if check.passed else "no",
                        _escape_table(check.detail),
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"


class SecRLAlertStream(AlertStream):
    def __init__(self, alerts: Iterable[Alert]) -> None:
        self.alerts = list(alerts)

    async def __aiter__(self):
        for alert in self.alerts:
            yield alert


def download_secrl_raw_logs(
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    extract: bool = True,
    force: bool = False,
) -> Path:
    database_dir = secrl_root / "secgym" / "database"
    archive_path = database_dir / RAW_ARCHIVE_NAME
    data_dir = database_dir / "data_anonymized"

    database_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists() or force:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub to download SecRL raw logs") from exc

        downloaded = hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=RAW_ARCHIVE_NAME,
            local_dir=str(database_dir),
            force_download=force,
        )
        archive_path = Path(downloaded)

    if extract and (force or not data_dir.exists()):
        _safe_extract_tar(archive_path, database_dir)

    return data_dir if data_dir.exists() else archive_path


async def evaluate_secrl_alert_detection(
    data_root: Path,
    scope: str = "incidents",
    limit: Optional[int] = None,
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
) -> SecRLDetectionReport:
    started = perf_counter()
    if ground_truth == "incident-graphs":
        incident_alert_ids = load_incident_graph_alert_ids(secrl_root)
    elif ground_truth == "security-incidents":
        incident_alert_ids = load_incident_alert_ids(data_root, scope=scope)
    else:
        raise ValueError("ground_truth must be 'incident-graphs' or 'security-incidents'")
    alerts = list(iter_security_alerts(data_root, scope=scope, limit=limit))

    graph_store = InMemoryGraphStore()
    stats = await StreamProcessor(GraphSketchingFilter(), graph_store).process(SecRLAlertStream(alerts))
    detected_ids = graph_store.alert_ids
    candidate_ids = {alert.id for alert in alerts}
    positive_ids = set().union(*incident_alert_ids.values()) if incident_alert_ids else set()
    available_positive_ids = positive_ids & candidate_ids

    true_positives = len(detected_ids & positive_ids)
    false_positives = len(detected_ids - positive_ids)
    false_negatives = len(positive_ids - detected_ids)
    precision = true_positives / len(detected_ids) if detected_ids else 0.0
    recall = true_positives / len(positive_ids) if positive_ids else 0.0
    available_recall = true_positives / len(available_positive_ids) if available_positive_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    incident_results: list[SecRLIncidentResult] = []
    for incident, ids in incident_alert_ids.items():
        if not ids:
            continue
        overlap = ids & detected_ids
        available = ids & candidate_ids
        incident_results.append(
            SecRLIncidentResult(
                incident=incident,
                ground_truth_alerts=len(ids),
                available_ground_truth_alerts=len(available),
                detected_ground_truth_alerts=len(overlap),
                alert_recall=len(overlap) / len(ids),
                available_recall=len(overlap) / len(available) if available else 0.0,
                any_detected=bool(overlap),
                all_detected=ids <= detected_ids,
            )
        )

    incident_count = len(incident_results)
    incident_any = sum(1 for item in incident_results if item.any_detected)
    incident_all = sum(1 for item in incident_results if item.all_detected)
    duration = perf_counter() - started
    return SecRLDetectionReport(
        scope=scope,
        ground_truth_source=ground_truth,
        rows_read=stats.processed,
        candidate_alerts=len(alerts),
        ground_truth_alerts=len(positive_ids),
        available_ground_truth_alerts=len(available_positive_ids),
        detected_alerts=len(detected_ids),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        available_recall=available_recall,
        f1=f1,
        incident_count=incident_count,
        incident_recall_any=incident_any / incident_count if incident_count else 0.0,
        incident_recall_all=incident_all / incident_count if incident_count else 0.0,
        mean_incident_alert_recall=sum(item.alert_recall for item in incident_results) / incident_count if incident_count else 0.0,
        alerts_per_second=stats.processed / duration if duration else float("inf"),
        notes=[
            "Ground truth positives are SecurityIncident.AlertIds matched against SecurityAlert.SystemAlertId.",
            "When ground_truth_source is incident-graphs, positives come from SecRL qagen/graph_files/incident_*.graphml.",
            "available_recall uses only ground-truth alert IDs present in the evaluated SecurityAlert rows.",
            "Detection means the E-ACS sketch filter stored the SecRL alert in the graph.",
            "Scope 'incidents' reads the 8 incident folders; scope 'full' reads data_anonymized/alphineskihouse.",
        ],
        incidents=incident_results,
    )


def analyze_secrl_errors(
    data_root: Path,
    scope: str = "full",
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    limit: Optional[int] = None,
    max_examples: int = 12,
) -> SecRLErrorAnalysisReport:
    incident_alert_ids = _load_ground_truth_ids(data_root, scope, ground_truth, secrl_root)
    candidate_alerts = list(iter_security_alerts(data_root, scope=scope, limit=limit))
    candidate_by_id = {alert.id: alert for alert in candidate_alerts}
    decisions = _evaluate_alert_decisions(candidate_alerts)
    detected_ids = {alert_id for alert_id, decision in decisions.items() if decision.store}
    candidate_ids = set(candidate_by_id)
    positive_ids = set().union(*incident_alert_ids.values()) if incident_alert_ids else set()

    raw_incident_refs = _load_security_incident_refs(data_root, scope)
    true_positive_ids = detected_ids & positive_ids
    false_positive_ids = detected_ids - positive_ids
    false_negative_ids = positive_ids - detected_ids

    false_positive_alerts = [candidate_by_id[id_] for id_ in sorted(false_positive_ids) if id_ in candidate_by_id]
    true_positive_alerts = [candidate_by_id[id_] for id_ in sorted(true_positive_ids) if id_ in candidate_by_id]

    incident_analyses = []
    for incident, ids in incident_alert_ids.items():
        available = ids & candidate_ids
        detected = ids & detected_ids
        missed_available = sorted((ids - detected) & candidate_ids)
        miss_reasons = Counter(decisions[id_].reason for id_ in missed_available if id_ in decisions)
        incident_analyses.append(
            SecRLIncidentErrorAnalysis(
                incident=incident,
                ground_truth_alerts=len(ids),
                available_ground_truth_alerts=len(available),
                detected_ground_truth_alerts=len(detected),
                alert_recall=len(detected) / len(ids),
                available_recall=len(detected & available) / len(available) if available else 0.0,
                missing_unavailable=len(ids - candidate_ids),
                missing_filtered=len(missed_available),
                miss_reasons=_sorted_counter(miss_reasons),
                missed_examples=[
                    _alert_example(candidate_by_id[id_], decisions[id_].reason, raw_incident_refs, scope)
                    for id_ in missed_available[:max_examples]
                    if id_ in decisions
                ],
            )
        )

    fp_reasons = Counter(decisions[alert.id].reason for alert in false_positive_alerts)
    fp_kinds = Counter(alert.kind for alert in false_positive_alerts)
    fp_severities = Counter(str(alert.severity) for alert in false_positive_alerts)
    fp_tags: Counter[str] = Counter()
    for alert in false_positive_alerts:
        fp_tags.update(alert.tags or {"<none>"})

    report = SecRLErrorAnalysisReport(
        scope=scope,
        ground_truth_source=ground_truth,
        candidate_alerts=len(candidate_alerts),
        detected_alerts=len(detected_ids),
        true_positives=len(true_positive_ids),
        false_positives=len(false_positive_ids),
        false_negatives=len(false_negative_ids),
        false_positives_with_security_incident_ref=sum(1 for id_ in false_positive_ids if id_ in raw_incident_refs),
        false_positive_decision_reasons=_sorted_counter(fp_reasons),
        false_positive_kinds=_sorted_counter(fp_kinds),
        false_positive_tags=_sorted_counter(fp_tags),
        false_positive_severities=_sorted_counter(fp_severities),
        top_false_positive_alert_names=_top_alert_names(false_positive_alerts),
        top_true_positive_alert_names=_top_alert_names(true_positive_alerts),
        incident_analyses=incident_analyses,
        false_positive_examples=[
            _alert_example(alert, decisions[alert.id].reason, raw_incident_refs, scope)
            for alert in false_positive_alerts[:max_examples]
        ],
        conclusions=_build_error_conclusions(
            scope=scope,
            false_positive_ids=false_positive_ids,
            raw_incident_refs=raw_incident_refs,
            incident_analyses=incident_analyses,
            fp_reasons=fp_reasons,
            fp_tags=fp_tags,
        ),
    )
    return report


def discover_secrl_incidents(
    data_root: Path,
    scope: str = "full",
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    limit: Optional[int] = None,
    max_gap_minutes: int = 120,
    min_alerts: int = 2,
    min_score: float = 0.55,
    max_examples: int = 8,
    refinements: Optional[Iterable[str]] = None,
    use_ground_truth_labels: bool = True,
    use_security_incident_labels: bool = True,
) -> SecRLIncidentDiscoveryReport:
    if max_gap_minutes <= 0:
        raise ValueError("max_gap_minutes must be positive")
    if min_alerts <= 0:
        raise ValueError("min_alerts must be positive")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    active_refinements = _normalize_refinements(refinements)

    incident_alert_ids = _load_ground_truth_ids(data_root, scope, ground_truth, secrl_root)
    label_incident_alert_ids = incident_alert_ids if use_ground_truth_labels else {}
    candidate_alerts = list(iter_security_alerts(data_root, scope=scope, limit=limit))
    candidate_by_id = {alert.id: alert for alert in candidate_alerts}
    decisions = _evaluate_alert_decisions(candidate_alerts)
    stored_alerts = [
        candidate_by_id[alert_id]
        for alert_id, decision in decisions.items()
        if decision.store and alert_id in candidate_by_id
    ]
    raw_incident_refs = _load_security_incident_refs(data_root, scope) if use_security_incident_labels else {}
    clusters = _cluster_stored_alerts(stored_alerts, max_gap_minutes=max_gap_minutes)

    incidents: list[SecRLDiscoveredIncident] = []
    for idx, cluster in enumerate(clusters, start=1):
        if len(cluster) < min_alerts:
            continue
        if not _passes_discovery_refinements(cluster, decisions, active_refinements):
            continue
        score, rationale, attack_graph_match = _score_discovered_cluster(cluster, decisions, active_refinements)
        if score < min_score:
            continue

        alert_ids = {alert.id for alert in cluster}
        known_ground_truth = sorted(
            incident for incident, ids in label_incident_alert_ids.items() if alert_ids & ids
        )
        raw_refs = sorted({ref for alert_id in alert_ids for ref in raw_incident_refs.get(alert_id, set())})
        status = "known_incident_overlap" if known_ground_truth or raw_refs else "candidate_new_incident"
        start_time, end_time = _cluster_time_range(cluster)
        sorted_cluster = sorted(cluster, key=lambda alert: (alert.timestamp, alert.id))

        incidents.append(
            SecRLDiscoveredIncident(
                cluster_id=f"cluster_{idx:04d}",
                status=status,
                score=score,
                alert_count=len(cluster),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                max_severity=max(alert.severity for alert in cluster),
                attack_tags=sorted(_cluster_attack_tags(cluster)),
                attack_graph_pattern=attack_graph_match.pattern,
                attack_graph_score=attack_graph_match.score,
                attack_graph_edges=list(attack_graph_match.matched_edges),
                entity_count=len(_cluster_entities(cluster)),
                decision_reasons=_sorted_counter(Counter(decisions[alert.id].reason for alert in cluster)),
                known_ground_truth_incidents=known_ground_truth,
                raw_security_incident_refs=raw_refs,
                alert_ids=[alert.id for alert in sorted_cluster],
                alert_names=[_alert_name(alert) for alert in sorted_cluster],
                rationale=rationale,
                examples=[
                    _alert_example(alert, decisions[alert.id].reason, raw_incident_refs, scope)
                    for alert in sorted_cluster[:max_examples]
                ],
            )
        )

    incidents = sorted(
        incidents,
        key=lambda item: (item.status != "candidate_new_incident", -item.score, item.cluster_id),
    )
    candidate_ids = {alert.id for alert in candidate_alerts}
    ground_truth_matches = _verify_discovered_incidents(
        incident_alert_ids=incident_alert_ids,
        candidate_ids=candidate_ids,
        discovered_incidents=incidents,
    )
    candidate_new = sum(1 for item in incidents if item.status == "candidate_new_incident")
    candidate_new_alerts = sum(item.alert_count for item in incidents if item.status == "candidate_new_incident")
    known_overlap = sum(1 for item in incidents if item.status == "known_incident_overlap")
    detected_ground_truth = sum(1 for item in ground_truth_matches if item.any_identified)
    ground_truth_count = len(ground_truth_matches)
    return SecRLIncidentDiscoveryReport(
        scope=scope,
        ground_truth_source=ground_truth,
        refinements=sorted(active_refinements),
        candidate_alerts=len(candidate_alerts),
        stored_alerts=len(stored_alerts),
        ground_truth_incidents=ground_truth_count,
        detected_ground_truth_incidents=detected_ground_truth,
        missed_ground_truth_incidents=ground_truth_count - detected_ground_truth,
        incident_recall_any=detected_ground_truth / ground_truth_count if ground_truth_count else 0.0,
        mean_ground_truth_alert_recall=_mean(item.alert_recall for item in ground_truth_matches),
        mean_available_alert_recall=_mean(item.available_recall for item in ground_truth_matches),
        clusters_considered=len(clusters),
        clusters_reported=len(incidents),
        known_overlap_clusters=known_overlap,
        candidate_new_incidents=candidate_new,
        candidate_new_alerts=candidate_new_alerts,
        min_score=min_score,
        min_alerts=min_alerts,
        max_gap_minutes=max_gap_minutes,
        incidents=incidents,
        ground_truth_matches=ground_truth_matches,
        notes=[
            "Clusters are built only from alerts admitted by the E-ACS sketch filter.",
            "Alerts are connected when they share a non-generic entity within max_gap_minutes.",
            "known_incident_overlap means a cluster overlaps benchmark ground truth or raw SecurityIncident.AlertIds.",
            "candidate_new_incident means the cluster is suspicious but has no known alert-ID overlap; it is not counted as a false positive in this report.",
            "Ground-truth verification uses only reported clusters after min-alert and min-score thresholds.",
            f"Post-hoc status labels use benchmark ground truth: {'yes' if use_ground_truth_labels else 'no'}.",
            f"Post-hoc status labels use raw SecurityIncident refs: {'yes' if use_security_incident_labels else 'no'}.",
        ],
    )


def run_secrl_discovery_ablation(
    data_root: Path,
    scope: str = "full",
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    limit: Optional[int] = None,
    max_gap_minutes: int = 120,
    min_alerts: int = 2,
    min_score: float = 0.55,
) -> SecRLDiscoveryAblationReport:
    variants = _discovery_ablation_variants()
    reports = {
        variant: discover_secrl_incidents(
            data_root,
            scope=scope,
            ground_truth=ground_truth,
            secrl_root=secrl_root,
            limit=limit,
            max_gap_minutes=max_gap_minutes,
            min_alerts=min_alerts,
            min_score=min_score,
            refinements=refinements,
        )
        for variant, refinements in variants.items()
    }
    baseline = reports["baseline"]
    rows = [
        _ablation_row(
            variant=variant,
            report=report,
            baseline=baseline,
        )
        for variant, report in reports.items()
    ]
    return SecRLDiscoveryAblationReport(
        scope=scope,
        ground_truth_source=ground_truth,
        baseline_candidate_new_incidents=baseline.candidate_new_incidents,
        baseline_candidate_new_alerts=baseline.candidate_new_alerts,
        baseline_incident_recall_any=baseline.incident_recall_any,
        rows=rows,
    )


def compare_secrl_discovery_baselines(
    data_root: Path,
    scope: str = "full",
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    limit: Optional[int] = None,
    max_gap_minutes: int = 120,
    min_alerts: int = 2,
    min_score: float = 0.55,
    random_seed: int = 7,
) -> SecRLDiscoveryBaselineReport:
    if max_gap_minutes <= 0:
        raise ValueError("max_gap_minutes must be positive")
    if min_alerts <= 0:
        raise ValueError("min_alerts must be positive")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")

    incident_alert_ids = _load_ground_truth_ids(data_root, scope, ground_truth, secrl_root)
    candidate_alerts = list(iter_security_alerts(data_root, scope=scope, limit=limit))
    candidate_by_id = {alert.id: alert for alert in candidate_alerts}
    candidate_ids = set(candidate_by_id)
    positive_ids = set().union(*incident_alert_ids.values()) if incident_alert_ids else set()
    available_positive_ids = positive_ids & candidate_ids
    decisions = _evaluate_alert_decisions(candidate_alerts)
    raw_incident_refs = _load_security_incident_refs(data_root, scope)
    eacs_selected_ids = {alert_id for alert_id, decision in decisions.items() if decision.store}

    eacs_baseline = discover_secrl_incidents(
        data_root,
        scope=scope,
        ground_truth=ground_truth,
        secrl_root=secrl_root,
        limit=limit,
        max_gap_minutes=max_gap_minutes,
        min_alerts=min_alerts,
        min_score=min_score,
        refinements=frozenset(),
    )
    eacs_refined = discover_secrl_incidents(
        data_root,
        scope=scope,
        ground_truth=ground_truth,
        secrl_root=secrl_root,
        limit=limit,
        max_gap_minutes=max_gap_minutes,
        min_alerts=min_alerts,
        min_score=min_score,
        refinements=DISCOVERY_REFINEMENTS,
    )
    reference_k = max(1, eacs_refined.stored_alerts or eacs_baseline.stored_alerts or len(eacs_selected_ids))

    rows = [
        _baseline_row_from_discovery_report(
            "eacs_baseline",
            "E-ACS sketch-selected alerts clustered by shared entity and time, using the baseline cluster scorer.",
            eacs_baseline,
            eacs_selected_ids,
            positive_ids,
            available_positive_ids,
        ),
        _baseline_row_from_discovery_report(
            "eacs_refined",
            "E-ACS sketch-selected alerts with all discovery precision refinements enabled.",
            eacs_refined,
            eacs_selected_ids,
            positive_ids,
            available_positive_ids,
        ),
    ]

    baseline_groups = [
        (
            "high_severity_only",
            "Alerts with normalized severity >= 8, then entity/time clustering and the common cluster scorer.",
            _groups_from_clustered_alerts(
                [alert for alert in candidate_alerts if alert.severity >= 8],
                max_gap_minutes,
            ),
            False,
            True,
        ),
        (
            "attack_keyword_only",
            "Alerts with inferred MITRE-like attack tags, then entity/time clustering and the common cluster scorer.",
            _groups_from_clustered_alerts(
                [alert for alert in candidate_alerts if _cluster_attack_tags([alert])],
                max_gap_minutes,
            ),
            False,
            True,
        ),
        (
            "entity_time_cluster_all_alerts",
            "All raw SecurityAlert rows with entity/time clustering and the common cluster scorer.",
            _groups_from_clustered_alerts(candidate_alerts, max_gap_minutes),
            False,
            True,
        ),
        (
            "vendor_security_incident",
            "Raw SecurityIncident.AlertIds groups from the SecRL data. This uses vendor incident rows, not graph labels.",
            _groups_from_alert_id_map(load_incident_alert_ids(data_root, scope=scope), candidate_by_id),
            False,
            False,
        ),
        (
            "random_top_k",
            f"A deterministic random sample of {reference_k} raw SecurityAlert rows, then entity/time clustering and the common cluster scorer.",
            _groups_from_clustered_alerts(
                _random_alert_sample(candidate_alerts, reference_k, random_seed),
                max_gap_minutes,
            ),
            False,
            True,
        ),
        (
            "graph_oracle",
            "Evaluator-only upper bound that groups alerts directly from the selected ground-truth incident IDs.",
            _groups_from_alert_id_map(incident_alert_ids, candidate_by_id),
            True,
            False,
        ),
    ]
    for baseline, selection, groups, uses_ground_truth, score_groups in baseline_groups:
        rows.append(
            _baseline_row_from_groups(
                baseline=baseline,
                selection=selection,
                uses_ground_truth=uses_ground_truth,
                groups=groups,
                candidate_alert_count=len(candidate_alerts),
                candidate_ids=candidate_ids,
                positive_ids=positive_ids,
                available_positive_ids=available_positive_ids,
                incident_alert_ids=incident_alert_ids,
                raw_incident_refs=raw_incident_refs,
                decisions=decisions,
                min_alerts=1 if baseline in {"vendor_security_incident", "graph_oracle"} else min_alerts,
                min_score=min_score,
                score_groups=score_groups,
            )
        )

    return SecRLDiscoveryBaselineReport(
        scope=scope,
        ground_truth_source=ground_truth,
        candidate_alerts=len(candidate_alerts),
        ground_truth_alerts=len(positive_ids),
        available_ground_truth_alerts=len(available_positive_ids),
        max_gap_minutes=max_gap_minutes,
        min_alerts=min_alerts,
        min_score=min_score,
        rows=rows,
        notes=[
            "Normal baselines do not read benchmark incident graph labels during selection or clustering.",
            "`graph_oracle` intentionally uses ground-truth labels and is only an upper bound.",
            "`vendor_security_incident` uses raw SecRL SecurityIncident.AlertIds rows, which may include neighboring Sentinel incidents outside the eight graph-labeled benchmark incidents.",
            "For heuristic baselines, the same entity/time clustering and cluster score threshold are used so the comparison isolates alert-selection quality.",
        ],
    )


def audit_secrl_discovery_leakage(
    data_root: Path,
    scope: str = "full",
    ground_truth: str = "incident-graphs",
    secrl_root: Path = DEFAULT_SECRL_ROOT,
    limit: Optional[int] = None,
    max_gap_minutes: int = 120,
    min_alerts: int = 2,
    min_score: float = 0.55,
    refinements: Optional[Iterable[str]] = None,
) -> SecRLLeakageAuditReport:
    active_refinements = _normalize_refinements(refinements)
    normal = discover_secrl_incidents(
        data_root,
        scope=scope,
        ground_truth=ground_truth,
        secrl_root=secrl_root,
        limit=limit,
        max_gap_minutes=max_gap_minutes,
        min_alerts=min_alerts,
        min_score=min_score,
        refinements=active_refinements,
        use_ground_truth_labels=True,
        use_security_incident_labels=True,
    )
    blind = discover_secrl_incidents(
        data_root,
        scope=scope,
        ground_truth=ground_truth,
        secrl_root=secrl_root,
        limit=limit,
        max_gap_minutes=max_gap_minutes,
        min_alerts=min_alerts,
        min_score=min_score,
        refinements=active_refinements,
        use_ground_truth_labels=False,
        use_security_incident_labels=False,
    )

    normal_signature = _discovery_generation_signature(normal)
    blind_signature = _discovery_generation_signature(blind)
    cluster_generation_stable = normal_signature == blind_signature
    score_generation_stable = _discovery_score_signature(normal) == _discovery_score_signature(blind)
    ground_truth_labeled_clusters = sum(1 for incident in normal.incidents if incident.known_ground_truth_incidents)
    raw_labeled_clusters = sum(1 for incident in normal.incidents if incident.raw_security_incident_refs)
    posthoc_label_delta = blind.candidate_new_incidents - normal.candidate_new_incidents

    checks = [
        SecRLLeakageAuditCheck(
            name="cluster_generation_without_ground_truth_labels",
            passed=cluster_generation_stable,
            detail=(
                "Normal and label-disabled runs produced identical cluster IDs, alert sets, topology patterns, and scores."
                if cluster_generation_stable
                else "Normal and label-disabled runs produced different cluster signatures."
            ),
        ),
        SecRLLeakageAuditCheck(
            name="score_generation_without_ground_truth_labels",
            passed=score_generation_stable,
            detail=(
                "Scores and attack-graph matches were unchanged after disabling post-hoc labels."
                if score_generation_stable
                else "Scores or attack-graph matches changed after disabling post-hoc labels."
            ),
        ),
        SecRLLeakageAuditCheck(
            name="attack_patterns_not_loaded_from_eval_graphs",
            passed=True,
            detail="Discovery uses DEFAULT_ATTACK_GRAPH_PATTERNS; SecRL GraphML labels are not passed into the attack-graph matcher.",
        ),
        SecRLLeakageAuditCheck(
            name="posthoc_labels_change_only_status",
            passed=posthoc_label_delta >= 0,
            detail=(
                f"Post-hoc labels moved {posthoc_label_delta} blind candidate cluster(s) into known-overlap status."
            ),
        ),
    ]
    potential_leakage = any(not check.passed for check in checks[:2])
    return SecRLLeakageAuditReport(
        scope=scope,
        ground_truth_source=ground_truth,
        refinements=sorted(active_refinements),
        candidate_alerts=normal.candidate_alerts,
        stored_alerts=normal.stored_alerts,
        normal_reported_clusters=normal.clusters_reported,
        blind_reported_clusters=blind.clusters_reported,
        normal_candidate_new_incidents=normal.candidate_new_incidents,
        blind_candidate_incidents=blind.candidate_new_incidents,
        ground_truth_labeled_clusters=ground_truth_labeled_clusters,
        raw_incident_labeled_clusters=raw_labeled_clusters,
        posthoc_label_delta=posthoc_label_delta,
        cluster_generation_stable=cluster_generation_stable,
        score_generation_stable=score_generation_stable,
        potential_leakage_detected=potential_leakage,
        checks=checks,
        notes=[
            "The audit runs discovery twice: once with normal post-hoc labels and once with benchmark/raw incident status labels disabled.",
            "A stable generation signature means labels did not affect candidate generation, cluster membership, scoring, or attack-graph matching.",
            "This does not prove the code can never leak; it is an executable regression check over the configured evaluation path.",
            "`SecurityIncident.AlertIds` and benchmark graph labels still affect reported status and candidate-new counts in normal reports.",
        ],
    )


def _discovery_ablation_variants() -> dict[str, frozenset[str]]:
    all_refinements = frozenset(DISCOVERY_REFINEMENTS)
    variants: dict[str, frozenset[str]] = {"baseline": frozenset()}
    for refinement in sorted(DISCOVERY_REFINEMENTS):
        variants[f"only_{refinement}"] = frozenset({refinement})
    variants["all_refinements"] = all_refinements
    for refinement in sorted(DISCOVERY_REFINEMENTS):
        variants[f"all_except_{refinement}"] = all_refinements - {refinement}
    return variants


def _ablation_row(
    variant: str,
    report: SecRLIncidentDiscoveryReport,
    baseline: SecRLIncidentDiscoveryReport,
) -> SecRLDiscoveryAblationRow:
    return SecRLDiscoveryAblationRow(
        variant=variant,
        refinements=report.refinements,
        clusters_reported=report.clusters_reported,
        known_overlap_clusters=report.known_overlap_clusters,
        candidate_new_incidents=report.candidate_new_incidents,
        candidate_new_alerts=report.candidate_new_alerts,
        detected_ground_truth_incidents=report.detected_ground_truth_incidents,
        incident_recall_any=report.incident_recall_any,
        mean_ground_truth_alert_recall=report.mean_ground_truth_alert_recall,
        mean_available_alert_recall=report.mean_available_alert_recall,
        candidate_new_reduction=baseline.candidate_new_incidents - report.candidate_new_incidents,
        candidate_new_alert_reduction=baseline.candidate_new_alerts - report.candidate_new_alerts,
        incident_recall_delta=report.incident_recall_any - baseline.incident_recall_any,
    )


def _baseline_row_from_discovery_report(
    baseline: str,
    selection: str,
    report: SecRLIncidentDiscoveryReport,
    selected_ids: set[str],
    positive_ids: set[str],
    available_positive_ids: set[str],
) -> SecRLDiscoveryBaselineRow:
    alert_precision, alert_recall, alert_available_recall, alert_f1 = _alert_selection_metrics(
        selected_ids,
        positive_ids,
        available_positive_ids,
    )
    gt_overlap_clusters = sum(1 for incident in report.incidents if incident.known_ground_truth_incidents)
    return SecRLDiscoveryBaselineRow(
        baseline=baseline,
        selection=selection,
        uses_ground_truth=False,
        candidate_alerts=report.candidate_alerts,
        selected_alerts=report.stored_alerts,
        alert_precision=alert_precision,
        alert_recall=alert_recall,
        alert_available_recall=alert_available_recall,
        alert_f1=alert_f1,
        clusters_reported=report.clusters_reported,
        gt_overlap_clusters=gt_overlap_clusters,
        non_gt_clusters=report.clusters_reported - gt_overlap_clusters,
        candidate_new_incidents=report.candidate_new_incidents,
        candidate_new_alerts=report.candidate_new_alerts,
        detected_ground_truth_incidents=report.detected_ground_truth_incidents,
        incident_recall_any=report.incident_recall_any,
        mean_ground_truth_alert_recall=report.mean_ground_truth_alert_recall,
        mean_available_alert_recall=report.mean_available_alert_recall,
    )


def _baseline_row_from_groups(
    baseline: str,
    selection: str,
    uses_ground_truth: bool,
    groups: dict[str, list[Alert]],
    candidate_alert_count: int,
    candidate_ids: set[str],
    positive_ids: set[str],
    available_positive_ids: set[str],
    incident_alert_ids: dict[str, set[str]],
    raw_incident_refs: dict[str, set[str]],
    decisions: dict[str, Any],
    min_alerts: int,
    min_score: float,
    score_groups: bool,
) -> SecRLDiscoveryBaselineRow:
    reported_groups: dict[str, set[str]] = {}
    selected_ids: set[str] = set()
    candidate_new_alerts = 0
    candidate_new_incidents = 0
    gt_overlap_clusters = 0

    for group_id, group in groups.items():
        group = [alert for alert in group if alert.id in candidate_ids]
        if not group:
            continue
        selected_ids.update(alert.id for alert in group)
        if len(group) < min_alerts:
            continue
        if score_groups:
            score, _, _ = _score_discovered_cluster(group, decisions)
            if score < min_score:
                continue
        alert_ids = {alert.id for alert in group}
        known_ground_truth = [
            incident
            for incident, ids in incident_alert_ids.items()
            if alert_ids & ids
        ]
        raw_refs = {ref for alert_id in alert_ids for ref in raw_incident_refs.get(alert_id, set())}
        if known_ground_truth:
            gt_overlap_clusters += 1
        if not known_ground_truth and not raw_refs:
            candidate_new_incidents += 1
            candidate_new_alerts += len(alert_ids)
        reported_groups[group_id] = alert_ids

    matches = _verify_incident_groups(
        incident_alert_ids=incident_alert_ids,
        candidate_ids=candidate_ids,
        group_alert_ids=reported_groups,
    )
    detected_ground_truth = sum(1 for item in matches if item.any_identified)
    ground_truth_count = len(matches)
    alert_precision, alert_recall, alert_available_recall, alert_f1 = _alert_selection_metrics(
        selected_ids,
        positive_ids,
        available_positive_ids,
    )
    return SecRLDiscoveryBaselineRow(
        baseline=baseline,
        selection=selection,
        uses_ground_truth=uses_ground_truth,
        candidate_alerts=candidate_alert_count,
        selected_alerts=len(selected_ids),
        alert_precision=alert_precision,
        alert_recall=alert_recall,
        alert_available_recall=alert_available_recall,
        alert_f1=alert_f1,
        clusters_reported=len(reported_groups),
        gt_overlap_clusters=gt_overlap_clusters,
        non_gt_clusters=len(reported_groups) - gt_overlap_clusters,
        candidate_new_incidents=candidate_new_incidents,
        candidate_new_alerts=candidate_new_alerts,
        detected_ground_truth_incidents=detected_ground_truth,
        incident_recall_any=detected_ground_truth / ground_truth_count if ground_truth_count else 0.0,
        mean_ground_truth_alert_recall=_mean(item.alert_recall for item in matches),
        mean_available_alert_recall=_mean(item.available_recall for item in matches),
    )


def _alert_selection_metrics(
    selected_ids: set[str],
    positive_ids: set[str],
    available_positive_ids: set[str],
) -> tuple[float, float, float, float]:
    true_positives = len(selected_ids & positive_ids)
    precision = true_positives / len(selected_ids) if selected_ids else 0.0
    recall = true_positives / len(positive_ids) if positive_ids else 0.0
    available_recall = len(selected_ids & available_positive_ids) / len(available_positive_ids) if available_positive_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, available_recall, f1


def _groups_from_clustered_alerts(alerts: list[Alert], max_gap_minutes: int) -> dict[str, list[Alert]]:
    return {
        f"cluster_{idx:04d}": cluster
        for idx, cluster in enumerate(_cluster_stored_alerts(alerts, max_gap_minutes=max_gap_minutes), start=1)
    }


def _groups_from_alert_id_map(alert_ids_by_group: dict[str, set[str]], candidate_by_id: dict[str, Alert]) -> dict[str, list[Alert]]:
    groups: dict[str, list[Alert]] = {}
    for group_id, alert_ids in sorted(alert_ids_by_group.items()):
        alerts = [candidate_by_id[alert_id] for alert_id in sorted(alert_ids) if alert_id in candidate_by_id]
        if alerts:
            groups[group_id] = alerts
    return groups


def _random_alert_sample(alerts: list[Alert], sample_size: int, seed: int) -> list[Alert]:
    if sample_size >= len(alerts):
        return list(alerts)
    rng = random.Random(seed)
    return rng.sample(alerts, sample_size)


def _discovery_generation_signature(report: SecRLIncidentDiscoveryReport) -> dict[str, tuple[Any, ...]]:
    return {
        incident.cluster_id: (
            tuple(incident.alert_ids),
            round(incident.score, 6),
            incident.attack_graph_pattern,
            round(incident.attack_graph_score, 6),
            tuple(incident.attack_graph_edges),
        )
        for incident in report.incidents
    }


def _discovery_score_signature(report: SecRLIncidentDiscoveryReport) -> dict[str, tuple[Any, ...]]:
    return {
        incident.cluster_id: (
            round(incident.score, 6),
            incident.attack_graph_pattern,
            round(incident.attack_graph_score, 6),
            tuple(incident.attack_graph_edges),
        )
        for incident in report.incidents
    }


def load_incident_alert_ids(data_root: Path, scope: str = "incidents") -> dict[str, set[str]]:
    incident_ids: dict[str, set[str]] = {}
    for folder in _scope_folders(data_root, scope):
        table = folder / "SecurityIncident.csv"
        if not table.exists():
            continue
        for row in _iter_csv_rows(table):
            incident_key = _first_text(row, "IncidentNumber", "IncidentName", default=folder.name)
            alert_ids = _parse_alert_ids(row.get("AlertIds", ""))
            if alert_ids:
                incident_ids[f"{folder.name}:{incident_key}"] = alert_ids
    return incident_ids


def _load_ground_truth_ids(
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


def load_incident_graph_alert_ids(secrl_root: Path = DEFAULT_SECRL_ROOT) -> dict[str, set[str]]:
    graph_dir = secrl_root / "secgym" / "qagen" / "graph_files"
    if not graph_dir.exists():
        raise FileNotFoundError(f"SecRL graph_files directory not found: {graph_dir}")

    incident_ids: dict[str, set[str]] = {}
    for incident_id in INCIDENT_IDS:
        graph_file = graph_dir / f"incident_{incident_id}.graphml"
        if not graph_file.exists():
            continue
        ids = _alert_ids_from_graphml(graph_file)
        if ids:
            incident_ids[f"incident_{incident_id}"] = ids
    return incident_ids


def iter_security_alerts(data_root: Path, scope: str = "incidents", limit: Optional[int] = None) -> Iterator[Alert]:
    count = 0
    for folder in _scope_folders(data_root, scope):
        for row in _iter_security_alert_rows(folder):
            yield alert_from_security_alert_row(row, folder.name)
            count += 1
            if limit is not None and count >= limit:
                return


def _evaluate_alert_decisions(alerts: Iterable[Alert]):
    sketch_filter = GraphSketchingFilter()
    decisions = {}
    for alert in alerts:
        decision = sketch_filter.evaluate(alert)
        previous = decisions.get(alert.id)
        if previous is None or (decision.store and not previous.store):
            decisions[alert.id] = decision
    return decisions


def alert_from_security_alert_row(row: dict[str, str], source_name: str) -> Alert:
    system_alert_id = _first_text(row, "SystemAlertId", "AlertId", "VendorOriginalId", default="")
    if not system_alert_id:
        system_alert_id = f"{source_name}:{_first_text(row, 'TimeGenerated', 'StartTime', default='unknown')}:{_first_text(row, 'AlertName', default='alert')}"

    entities = _entities_from_security_alert(row)
    source = entities[0] if entities else Entity(type=EntityType.SERVICE, value=source_name)
    target = entities[1] if len(entities) > 1 else None
    alert_name = _first_text(row, "AlertName", "DisplayName", default="security_alert")
    raw = dict(row)
    raw["_eacs_source_scope"] = source_name

    return Alert(
        id=system_alert_id,
        source=source,
        target=target,
        kind=_infer_kind(row),
        action=_normalize_token(alert_name) or "observed",
        severity=_severity_score(_first_text(row, "AlertSeverity", "Severity", default="Low")),
        timestamp=_alert_timestamp(row),
        raw=raw,
        tags=_attack_tags(row),
    )


def _scope_folders(data_root: Path, scope: str) -> list[Path]:
    if scope == "full":
        return [data_root / "alphineskihouse"]
    if scope == "incidents":
        return [data_root / "incidents" / f"incident_{incident_id}" for incident_id in INCIDENT_IDS]
    if scope.startswith("incident_"):
        return [data_root / "incidents" / scope]
    raise ValueError("scope must be 'incidents', 'full', or an incident_<id> folder name")


def _iter_security_alert_rows(folder: Path) -> Iterator[dict[str, str]]:
    table_file = folder / "SecurityAlert.csv"
    table_dir = folder / "SecurityAlert"
    if table_file.exists():
        yield from _iter_csv_rows(table_file)
    elif table_dir.exists():
        for csv_file in sorted(table_dir.glob("SecurityAlert_*.csv")):
            yield from _iter_csv_rows(csv_file)


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=SEPARATOR, quotechar='"')
        for row in reader:
            yield {str(key): "" if value is None else str(value) for key, value in row.items() if key is not None}


def _load_security_incident_refs(data_root: Path, scope: str) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for folder in _scope_folders(data_root, scope):
        table = folder / "SecurityIncident.csv"
        if not table.exists():
            continue
        for row in _iter_csv_rows(table):
            incident_number = _first_text(row, "IncidentNumber", "IncidentName", default=folder.name)
            incident_label = f"{folder.name}:{incident_number}"
            for alert_id in _parse_alert_ids(row.get("AlertIds", "")):
                refs[alert_id].add(incident_label)
    return refs


def _cluster_stored_alerts(alerts: list[Alert], max_gap_minutes: int) -> list[list[Alert]]:
    if not alerts:
        return []

    parent = {alert.id: alert.id for alert in alerts}

    def find(alert_id: str) -> str:
        while parent[alert_id] != alert_id:
            parent[alert_id] = parent[parent[alert_id]]
            alert_id = parent[alert_id]
        return alert_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    entity_alerts: dict[str, list[Alert]] = defaultdict(list)
    for alert in alerts:
        for entity_id in _cluster_entity_ids(alert):
            entity_alerts[entity_id].append(alert)

    max_gap_seconds = max_gap_minutes * 60
    for entity_group in entity_alerts.values():
        ordered = sorted(entity_group, key=lambda alert: (alert.timestamp, alert.id))
        for idx, alert in enumerate(ordered):
            previous_idx = idx - 1
            while previous_idx >= 0:
                previous = ordered[previous_idx]
                gap = (alert.timestamp - previous.timestamp).total_seconds()
                if gap > max_gap_seconds:
                    break
                union(alert.id, previous.id)
                previous_idx -= 1

    clusters_by_root: dict[str, list[Alert]] = defaultdict(list)
    for alert in alerts:
        clusters_by_root[find(alert.id)].append(alert)

    return sorted(
        (sorted(cluster, key=lambda alert: (alert.timestamp, alert.id)) for cluster in clusters_by_root.values()),
        key=lambda cluster: (cluster[0].timestamp, cluster[0].id),
    )


def _verify_discovered_incidents(
    incident_alert_ids: dict[str, set[str]],
    candidate_ids: set[str],
    discovered_incidents: list[SecRLDiscoveredIncident],
) -> list[SecRLGroundTruthIncidentMatch]:
    return _verify_incident_groups(
        incident_alert_ids=incident_alert_ids,
        candidate_ids=candidate_ids,
        group_alert_ids={
            incident.cluster_id: set(incident.alert_ids)
            for incident in discovered_incidents
        },
    )


def _verify_incident_groups(
    incident_alert_ids: dict[str, set[str]],
    candidate_ids: set[str],
    group_alert_ids: dict[str, set[str]],
) -> list[SecRLGroundTruthIncidentMatch]:
    cluster_alert_ids = dict(group_alert_ids)
    matches: list[SecRLGroundTruthIncidentMatch] = []
    for incident, ground_truth_ids in sorted(incident_alert_ids.items()):
        available = ground_truth_ids & candidate_ids
        overlaps = {
            cluster_id: ground_truth_ids & alert_ids
            for cluster_id, alert_ids in cluster_alert_ids.items()
        }
        overlaps = {cluster_id: ids for cluster_id, ids in overlaps.items() if ids}
        recovered = set().union(*overlaps.values()) if overlaps else set()
        best_cluster_id = ""
        best_cluster_overlap = 0
        if overlaps:
            best_cluster_id, best_overlap_ids = max(
                overlaps.items(),
                key=lambda item: (len(item[1]), item[0]),
            )
            best_cluster_overlap = len(best_overlap_ids)

        matches.append(
            SecRLGroundTruthIncidentMatch(
                incident=incident,
                ground_truth_alerts=len(ground_truth_ids),
                available_ground_truth_alerts=len(available),
                recovered_alerts=len(recovered),
                alert_recall=len(recovered) / len(ground_truth_ids) if ground_truth_ids else 0.0,
                available_recall=len(recovered & available) / len(available) if available else 0.0,
                matched_clusters=sorted(overlaps),
                best_cluster_id=best_cluster_id,
                best_cluster_overlap=best_cluster_overlap,
                any_identified=bool(overlaps),
                all_available_recovered=bool(available) and available <= recovered,
            )
        )
    return matches


def _normalize_refinements(refinements: Optional[Iterable[str]]) -> frozenset[str]:
    if refinements is None:
        return frozenset()
    normalized = frozenset(refinements)
    unknown = normalized - DISCOVERY_REFINEMENTS
    if unknown:
        raise ValueError(f"unknown discovery refinements: {', '.join(sorted(unknown))}")
    return normalized


def _passes_discovery_refinements(
    cluster: list[Alert],
    decisions: dict[str, Any],
    refinements: frozenset[str],
) -> bool:
    if STRICT_ENTITY_CONTINUITY in refinements and not _has_strong_entity_continuity(cluster):
        return False
    if SUPPRESS_GENERIC_ICS in refinements and _is_generic_ics_cluster(cluster):
        return False
    if REQUIRE_PROGRESSION_OR_SEVERITY in refinements and not _has_progression_or_severity_density(cluster):
        return False
    return True


def _has_strong_entity_continuity(cluster: list[Alert]) -> bool:
    counts = _cluster_entity_counts(cluster)
    if not counts:
        return False
    shared_entities = [count for count in counts.values() if count > 1]
    if not shared_entities:
        return False
    max_count = max(shared_entities)
    anchor_ratio = max_count / len(cluster)
    if len(cluster) <= 3:
        return max_count >= 2
    if len(cluster) <= 10:
        return max_count >= 3
    return anchor_ratio >= 0.20 and max_count >= 3


def _is_generic_ics_cluster(cluster: list[Alert]) -> bool:
    names = [_alert_name(alert).lower() for alert in cluster]
    generic_count = sum(1 for name in names if _is_generic_ics_alert_name(name))
    if not names:
        return False
    return generic_count / len(names) >= 0.50


def _is_generic_ics_alert_name(name: str) -> bool:
    generic_patterns = (
        "new activity detected - cip class",
        "new activity detected - cip class service",
        "new port discovery",
        "ethernet/ip cip service request failed",
        "ethernet/ip encapsulation protocol command",
        "modbus exception",
        "unauthorized mitsubishi melsec command",
        "omron fins unauthorized command",
        "bacnet operation failed",
        "new asset detected",
    )
    return any(pattern in name for pattern in generic_patterns)


def _has_progression_or_severity_density(cluster: list[Alert]) -> bool:
    meaningful_tags = _cluster_attack_tags(cluster) - {"command_and_control"}
    high_severity = sum(1 for alert in cluster if alert.severity >= 8)
    high_severity_density = high_severity / len(cluster)
    if len(meaningful_tags) >= 2:
        return True
    if len(cluster) <= 3:
        return high_severity >= 1
    return high_severity_density >= 0.25


def _score_discovered_cluster(
    cluster: list[Alert],
    decisions: dict[str, Any],
    refinements: frozenset[str] = frozenset(),
) -> tuple[float, list[str], AttackGraphMatch]:
    tags = _cluster_attack_tags(cluster)
    score_tags = tags
    if REDUCE_C2_WEIGHT in refinements:
        score_tags = tags - {"command_and_control"}
    attack_graph_match = _best_attack_graph_match(cluster)
    severities = [alert.severity for alert in cluster]
    reason_counts = Counter(decisions[alert.id].reason for alert in cluster)
    low_frequency = sum(1 for alert in cluster if decisions[alert.id].estimated_frequency <= 3)
    shared_entities = _shared_cluster_entities(cluster)

    severity_score = max(severities) / 10
    tactic_score = min(len(score_tags) / 3, 1.0)
    if REDUCE_C2_WEIGHT in refinements and tags == {"command_and_control"}:
        tactic_score = 0.0
    rarity_score = low_frequency / len(cluster)
    size_score = min(len(cluster) / 5, 1.0)
    chain_score = 0.0
    if len(cluster) >= 2:
        chain_score += 0.30
    if shared_entities:
        chain_score += 0.25
    if len(score_tags) >= 2:
        chain_score += 0.25
    if len({alert.kind for alert in cluster}) >= 2:
        chain_score += 0.20
    chain_score = min(chain_score, 1.0)

    heuristic_score = (
        severity_score * 0.30
        + tactic_score * 0.25
        + chain_score * 0.20
        + rarity_score * 0.15
        + size_score * 0.10
    )
    score = heuristic_score + attack_graph_match.score * 0.10
    rationale = [
        f"max_severity={max(severities)}",
        f"attack_tags={','.join(sorted(tags)) or '<none>'}",
        f"shared_entities={len(shared_entities)}",
        f"attack_graph={attack_graph_match.pattern}:{attack_graph_match.score:.2f}",
        f"decision_reasons={dict(_sorted_counter(reason_counts))}",
    ]
    if attack_graph_match.matched_edges:
        rationale.append(f"attack_graph_edges={','.join(attack_graph_match.matched_edges)}")
    if REDUCE_C2_WEIGHT in refinements:
        rationale.append(f"score_tags={','.join(sorted(score_tags)) or '<none>'}")
    if reason_counts.get("high_severity"):
        rationale.append("contains high-severity alert(s)")
    if reason_counts.get("attack_topology"):
        rationale.append("contains attack-topology alert(s)")
    if reason_counts.get("rare_relationship"):
        rationale.append("contains rare relationship alert(s)")
    return min(score, 1.0), rationale, attack_graph_match


def _cluster_attack_tags(cluster: Iterable[Alert]) -> set[str]:
    tags: set[str] = set()
    for alert in cluster:
        tags.update(tag for tag in alert.tags if tag in GraphSketchingFilter.ATTACK_KINDS)
        if alert.kind in GraphSketchingFilter.ATTACK_KINDS:
            tags.add(alert.kind)
    return tags


def _best_attack_graph_match(
    cluster: list[Alert],
    patterns: tuple[AttackGraphPattern, ...] = DEFAULT_ATTACK_GRAPH_PATTERNS,
) -> AttackGraphMatch:
    observed_stages = _cluster_attack_tags(cluster)
    observed_edges = _cluster_attack_edges(cluster)
    stage_sequence = _cluster_stage_sequence(cluster)
    if not observed_stages:
        return EMPTY_ATTACK_GRAPH_MATCH

    matches = [
        _score_attack_graph_pattern(pattern, observed_stages, observed_edges, stage_sequence)
        for pattern in patterns
    ]
    return max(matches, key=lambda match: (match.score, len(match.matched_edges), len(match.matched_stages)))


def _score_attack_graph_pattern(
    pattern: AttackGraphPattern,
    observed_stages: set[str],
    observed_edges: set[tuple[str, str]],
    stage_sequence: list[str],
) -> AttackGraphMatch:
    pattern_stages = set(pattern.stages)
    matched_stages = tuple(sorted(observed_stages & pattern_stages))
    matched_edges = tuple(sorted(_format_attack_edge(edge) for edge in observed_edges & pattern.edges))
    stage_score = len(matched_stages) / len(pattern_stages) if pattern_stages else 0.0
    edge_score = len(matched_edges) / len(pattern.edges) if pattern.edges else 0.0
    order_score = _ordered_stage_score(stage_sequence, pattern.stages)
    score = stage_score * 0.35 + edge_score * 0.45 + order_score * 0.20
    if pattern.edges and not matched_edges:
        score *= 0.65
    return AttackGraphMatch(
        pattern=pattern.name if score > 0 else "none",
        score=round(min(score, 1.0), 6),
        matched_stages=matched_stages,
        matched_edges=matched_edges,
        observed_stages=tuple(sorted(observed_stages)),
        observed_edges=tuple(sorted(_format_attack_edge(edge) for edge in observed_edges)),
    )


def _cluster_attack_edges(cluster: list[Alert]) -> set[tuple[str, str]]:
    ordered = sorted(cluster, key=lambda alert: (alert.timestamp, alert.id))
    edges: set[tuple[str, str]] = set()
    alert_tags = {alert.id: _cluster_attack_tags([alert]) for alert in ordered}
    alert_entities = {alert.id: set(_cluster_entity_ids(alert)) for alert in ordered}

    for idx, source in enumerate(ordered):
        for target in ordered[idx + 1 :]:
            if alert_entities[source.id] and alert_entities[target.id] and not (alert_entities[source.id] & alert_entities[target.id]):
                continue
            edges.update(_tag_edges(alert_tags[source.id], alert_tags[target.id]))

    sequence = _cluster_stage_sequence(cluster)
    edges.update((left, right) for left, right in zip(sequence, sequence[1:]) if left != right)
    return edges


def _cluster_stage_sequence(cluster: list[Alert]) -> list[str]:
    ordered = sorted(cluster, key=lambda alert: (alert.timestamp, alert.id))
    sequence: list[str] = []
    for alert in ordered:
        for tag in sorted(_cluster_attack_tags([alert]), key=_stage_sort_key):
            if not sequence or sequence[-1] != tag:
                sequence.append(tag)
    return sequence


def _tag_edges(left_tags: set[str], right_tags: set[str]) -> set[tuple[str, str]]:
    return {
        (left, right)
        for left in left_tags
        for right in right_tags
        if left != right
    }


def _ordered_stage_score(sequence: list[str], pattern_stages: tuple[str, ...]) -> float:
    if not sequence or not pattern_stages:
        return 0.0
    previous = [0] * (len(pattern_stages) + 1)
    for observed in sequence:
        current = previous[:]
        for idx, expected in enumerate(pattern_stages, start=1):
            if observed == expected:
                current[idx] = max(current[idx], previous[idx - 1] + 1)
            else:
                current[idx] = max(current[idx], current[idx - 1], previous[idx])
        previous = current
    return previous[-1] / len(pattern_stages)


def _stage_sort_key(stage: str) -> tuple[int, str]:
    order = {
        "initial_access": 10,
        "credential_access": 20,
        "execution": 30,
        "command_and_control": 35,
        "lateral_movement": 40,
        "privilege_escalation": 50,
        "defense_evasion": 60,
        "discovery": 70,
        "collection": 80,
        "data_exfiltration": 90,
        "impact": 100,
    }
    return order.get(stage, 1000), stage


def _format_attack_edge(edge: tuple[str, str]) -> str:
    return f"{edge[0]}->{edge[1]}"


def _cluster_entities(cluster: Iterable[Alert]) -> set[str]:
    entities: set[str] = set()
    for alert in cluster:
        entities.update(_cluster_entity_ids(alert))
    return entities


def _shared_cluster_entities(cluster: Iterable[Alert]) -> set[str]:
    counts = _cluster_entity_counts(cluster)
    return {entity_id for entity_id, count in counts.items() if count > 1}


def _cluster_entity_counts(cluster: Iterable[Alert]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for alert in cluster:
        counts.update(set(_cluster_entity_ids(alert)))
    return counts


def _cluster_entity_ids(alert: Alert) -> list[str]:
    source_scope = str(alert.raw.get("_eacs_source_scope", "")).strip().lower()
    generic_ids = {f"service:{source_scope}"} if source_scope else set()
    return [entity.id for entity in alert.entities if entity.id not in generic_ids]


def _cluster_time_range(cluster: list[Alert]) -> tuple[datetime, datetime]:
    timestamps = [alert.timestamp for alert in cluster]
    return min(timestamps), max(timestamps)


def _alert_name(alert: Alert) -> str:
    return str(alert.raw.get("AlertName") or alert.raw.get("DisplayName") or alert.action)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _entities_from_security_alert(row: dict[str, str]) -> list[Entity]:
    entities_payload = row.get("Entities", "")
    try:
        raw_entities = json.loads(entities_payload) if entities_payload else []
    except json.JSONDecodeError:
        raw_entities = []

    entities: list[Entity] = []
    seen: set[str] = set()
    for item in raw_entities:
        if not isinstance(item, dict) or "$ref" in item:
            continue
        entity = _entity_from_dict(item)
        if entity and entity.id not in seen:
            seen.add(entity.id)
            entities.append(entity)

    for field, entity_type in (
        ("CompromisedEntity", EntityType.HOST),
        ("SourceComputerId", EntityType.HOST),
        ("ResourceId", EntityType.SERVICE),
    ):
        value = row.get(field, "").strip()
        if value:
            entity = Entity(type=entity_type, value=value)
            if entity.id not in seen:
                seen.add(entity.id)
                entities.append(entity)

    return entities


def _entity_from_dict(item: dict[str, Any]) -> Optional[Entity]:
    raw_type = str(item.get("Type", "")).lower()
    candidates: list[tuple[EntityType, str]] = []

    if raw_type == "account":
        candidates = [(EntityType.USER, str(item.get(key, ""))) for key in ("UserPrincipalName", "Name", "Sid", "AadUserId")]
    elif raw_type == "host":
        candidates = [(EntityType.HOST, str(item.get(key, ""))) for key in ("HostName", "FQDN", "AadDeviceId", "MdatpDeviceId")]
    elif raw_type == "ip":
        candidates = [(EntityType.IP, str(item.get("Address", "")))]
    elif raw_type in {"mailbox", "mailboxconfiguration"}:
        candidates = [(EntityType.USER, str(item.get("MailboxPrimaryAddress", "")))]
    elif raw_type == "file":
        candidates = [(EntityType.FILE, str(item.get("Name", "")))]
    elif raw_type == "url":
        candidates = [(EntityType.SERVICE, str(item.get("Url", "")))]
    elif raw_type in {"cloud-application", "oauth-application", "service-principal"}:
        candidates = [(EntityType.SERVICE, str(item.get(key, ""))) for key in ("Name", "AppId", "OAuthAppId", "ServicePrincipalObjectId")]
    elif raw_type == "process":
        candidates = [(EntityType.SERVICE, str(item.get(key, ""))) for key in ("CommandLine", "ProcessId")]

    for entity_type, value in candidates:
        value = value.strip()
        if value and value.lower() not in {"none", "nan", "system", "localsystem"}:
            return Entity(type=entity_type, value=value[:500])
    return None


def _parse_alert_ids(value: str) -> set[str]:
    if not value:
        return set()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return {str(item) for item in parsed if item}
    except json.JSONDecodeError:
        pass
    return set(UUID_PATTERN.findall(value))


def _infer_kind(row: dict[str, str]) -> str:
    text = _row_text(row)
    tags = _attack_tags(row)
    if "initial_access" in tags:
        return "initial_access"
    if "lateral_movement" in tags:
        return "lateral_movement"
    if "credential_access" in tags:
        return "credential_access"
    if "execution" in tags:
        return "execution"
    if "data_exfiltration" in tags:
        return "data_exfiltration"
    if "command_and_control" in tags:
        return "command_and_control"
    if "privilege_escalation" in tags:
        return "privilege_escalation"
    if "defense_evasion" in tags:
        return "defense_evasion"
    if "discovery" in tags:
        return "discovery"
    if "collection" in tags:
        return "collection"
    if "impact" in tags:
        return "impact"
    return _normalize_token(_first_text(row, "AlertName", "DisplayName", default=text[:50])) or "security_alert"


def _attack_tags(row: dict[str, str]) -> set[str]:
    text = _row_text(row)
    tags: set[str] = set()
    if any(term in text for term in ("initial access", "drive-by", "malvertising", "fakeupdates", "phish")):
        tags.add("initial_access")
    if any(term in text for term in ("lateral", "remote activity", "impacket", "smb", "wmi")):
        tags.add("lateral_movement")
    if any(term in text for term in ("credential", "password", "lsass", "dcsync", "phish", "signin", "sign-in")):
        tags.add("credential_access")
    if any(term in text for term in ("execution", "powershell", "encodedcommand", "command execution", "process creation", "remote command", "script")):
        tags.add("execution")
    if any(term in text for term in ("collection", "email collection", "inbox", "archive mailbox", "mailbox")):
        tags.add("collection")
    if any(term in text for term in ("exfil", "exfiltration", "data theft", "downloaded data")):
        tags.add("data_exfiltration")
    if any(term in text for term in ("c2", "command and control", "command-and-control", "commandandcontrol", "cobalt strike", "beacon", "domain fronting", "callback")):
        tags.add("command_and_control")
    if any(term in text for term in ("privilege", "persistence", "scheduler", "service creation", "oauth")):
        tags.add("privilege_escalation")
    if any(term in text for term in ("defense evasion", "bypass", "disabled security", "tamper", "obfuscat")):
        tags.add("defense_evasion")
    if any(term in text for term in ("discovery", "enumerat", "reconnaissance", "network scan", "port scan")):
        tags.add("discovery")
    if any(term in text for term in ("impact", "ransomware", "shadow copies", "file backups were deleted", "wipe", "encrypted")):
        tags.add("impact")
    return tags


def _severity_score(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "high":
        return 9
    if normalized == "medium":
        return 6
    if normalized == "low":
        return 4
    if normalized in {"informational", "info"}:
        return 2
    try:
        numeric = int(float(normalized))
    except ValueError:
        return 1
    return max(0, min(10, numeric))


def _alert_timestamp(row: dict[str, str]) -> datetime:
    for key in ("StartTime", "TimeGenerated", "EndTime"):
        value = row.get(key, "").strip()
        if not value or value.lower() in {"nan", "none"}:
            continue
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            try:
                parsed = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _first_text(row: dict[str, str], *keys: str, default: str) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return default


def _row_text(row: dict[str, str]) -> str:
    fields = ("AlertName", "DisplayName", "Description", "Tactics", "Techniques", "Entities")
    return " ".join(row.get(field, "") for field in fields).lower()


def _normalize_token(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return value.strip("_")


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if os.path.commonpath([str(destination), str(target)]) != str(destination):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(destination)


def _alert_ids_from_graphml(graph_file: Path) -> set[str]:
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(graph_file).getroot()
    keys = {node.attrib["id"]: node.attrib.get("attr.name", "") for node in root.findall("g:key", namespace)}
    alert_ids: set[str] = set()

    for node in root.findall(".//g:node", namespace):
        data = {keys[item.attrib["key"]]: item.text or "" for item in node.findall("g:data", namespace)}
        if data.get("type") != "alert" or not data.get("entry"):
            continue
        try:
            entry = json.loads(data["entry"])
        except json.JSONDecodeError:
            continue
        system_alert_id = str(entry.get("SystemAlertId", "")).strip()
        if system_alert_id:
            alert_ids.add(system_alert_id)
    return alert_ids


def _alert_example(
    alert: Alert,
    decision_reason: str,
    raw_incident_refs: dict[str, set[str]],
    source_scope: str,
) -> SecRLAlertExample:
    return SecRLAlertExample(
        alert_id=alert.id,
        alert_name=str(alert.raw.get("AlertName") or alert.raw.get("DisplayName") or alert.action),
        severity=alert.severity,
        kind=alert.kind,
        tags=sorted(alert.tags),
        decision_reason=decision_reason,
        incident_refs=sorted(raw_incident_refs.get(alert.id, set())),
        source_scope=source_scope,
    )


def _top_alert_names(alerts: Iterable[Alert], limit: int = 12) -> dict[str, int]:
    counter = Counter(str(alert.raw.get("AlertName") or alert.raw.get("DisplayName") or alert.action) for alert in alerts)
    return dict(counter.most_common(limit))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _build_error_conclusions(
    scope: str,
    false_positive_ids: set[str],
    raw_incident_refs: dict[str, set[str]],
    incident_analyses: list[SecRLIncidentErrorAnalysis],
    fp_reasons: Counter[str],
    fp_tags: Counter[str],
) -> list[str]:
    conclusions = []
    referenced = sum(1 for id_ in false_positive_ids if id_ in raw_incident_refs)
    if false_positive_ids:
        conclusions.append(
            f"{referenced}/{len(false_positive_ids)} false positives are referenced by raw SecurityIncident rows. "
            "These are usually neighboring SecRL/Sentinel incidents rather than ordinary benign alerts."
        )
    if fp_reasons:
        reason, count = fp_reasons.most_common(1)[0]
        conclusions.append(f"The dominant false-positive gate was '{reason}' ({count} alerts), so precision is mostly limited by broad sketch-filter heuristics.")
    if fp_tags:
        tag, count = fp_tags.most_common(1)[0]
        conclusions.append(f"The most common false-positive attack tag was '{tag}' ({count} alerts), showing that keyword topology matching is too permissive.")
    unavailable = sum(item.missing_unavailable for item in incident_analyses)
    filtered = sum(item.missing_filtered for item in incident_analyses)
    if unavailable:
        conclusions.append(f"{unavailable} false negatives were not present in the evaluated '{scope}' SecurityAlert rows, so they are data-scope misses rather than filter misses.")
    if filtered:
        conclusions.append(f"{filtered} available ground-truth alerts were filtered out; inspect per-incident miss reasons for the needed recall fix.")
    strong = [item.incident for item in incident_analyses if item.alert_recall >= 0.99]
    weak = [item.incident for item in incident_analyses if item.alert_recall < 0.8]
    if strong:
        conclusions.append(f"High-recall incidents were {', '.join(strong)} because their alerts had high severity or explicit attack-topology labels.")
    if weak:
        conclusions.append(f"Weaker incidents were {', '.join(weak)}; in incident-window scope, most misses are alerts absent from that incident folder.")
    return conclusions


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _example_row(example: SecRLAlertExample) -> str:
    refs = ", ".join(example.incident_refs) or "-"
    return (
        f"| `{example.alert_id}` | {_escape_table(example.alert_name)} | {example.severity} | "
        f"{example.kind} | {example.decision_reason} | {_escape_table(refs)} |"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SecRL raw logs and evaluate E-ACS alert detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download data_anonymized.tar.gz into the SecRL database folder.")
    download.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    download.add_argument("--no-extract", action="store_true")
    download.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate-alerts", help="Evaluate E-ACS detection on SecRL SecurityAlert CSVs.")
    evaluate.add_argument("--data-root", type=Path, default=DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized")
    evaluate.add_argument("--scope", default="incidents")
    evaluate.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    evaluate.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--output", type=Path, default=None)

    analyze = subparsers.add_parser("analyze-errors", help="Explain SecRL false positives and false negatives.")
    analyze.add_argument("--data-root", type=Path, default=DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized")
    analyze.add_argument("--scope", default="full")
    analyze.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    analyze.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    analyze.add_argument("--limit", type=int, default=None)
    analyze.add_argument("--max-examples", type=int, default=12)
    analyze.add_argument("--output-json", type=Path, default=None)
    analyze.add_argument("--output-md", type=Path, default=None)

    discover = subparsers.add_parser("discover-incidents", help="Cluster stored alerts into known-overlap and candidate-new incidents.")
    discover.add_argument("--data-root", type=Path, default=DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized")
    discover.add_argument("--scope", default="full")
    discover.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    discover.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    discover.add_argument("--limit", type=int, default=None)
    discover.add_argument("--max-gap-minutes", type=int, default=120)
    discover.add_argument("--min-alerts", type=int, default=2)
    discover.add_argument("--min-score", type=float, default=0.55)
    discover.add_argument("--max-examples", type=int, default=8)
    discover.add_argument("--profile", default="baseline", choices=["baseline", "refined"])
    discover.add_argument("--refinement", action="append", default=[], choices=sorted(DISCOVERY_REFINEMENTS))
    discover.add_argument("--output-json", type=Path, default=None)
    discover.add_argument("--output-md", type=Path, default=None)

    ablate = subparsers.add_parser("ablate-discovery", help="Run discovery refinement ablations on SecRL raw logs.")
    ablate.add_argument("--data-root", type=Path, default=DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized")
    ablate.add_argument("--scope", default="full")
    ablate.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    ablate.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    ablate.add_argument("--limit", type=int, default=None)
    ablate.add_argument("--max-gap-minutes", type=int, default=120)
    ablate.add_argument("--min-alerts", type=int, default=2)
    ablate.add_argument("--min-score", type=float, default=0.55)
    ablate.add_argument("--output-json", type=Path, default=None)
    ablate.add_argument("--output-md", type=Path, default=None)

    compare = subparsers.add_parser("compare-baselines", help="Compare E-ACS discovery against simple raw-log baselines.")
    compare.add_argument("--data-root", type=Path, default=DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized")
    compare.add_argument("--scope", default="full")
    compare.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    compare.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    compare.add_argument("--limit", type=int, default=None)
    compare.add_argument("--max-gap-minutes", type=int, default=120)
    compare.add_argument("--min-alerts", type=int, default=2)
    compare.add_argument("--min-score", type=float, default=0.55)
    compare.add_argument("--random-seed", type=int, default=7)
    compare.add_argument("--output-json", type=Path, default=None)
    compare.add_argument("--output-md", type=Path, default=None)

    audit = subparsers.add_parser("audit-leakage", help="Audit whether discovery generation changes when post-hoc labels are disabled.")
    audit.add_argument("--data-root", type=Path, default=DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized")
    audit.add_argument("--scope", default="full")
    audit.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    audit.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument("--max-gap-minutes", type=int, default=120)
    audit.add_argument("--min-alerts", type=int, default=2)
    audit.add_argument("--min-score", type=float, default=0.55)
    audit.add_argument("--profile", default="baseline", choices=["baseline", "refined"])
    audit.add_argument("--refinement", action="append", default=[], choices=sorted(DISCOVERY_REFINEMENTS))
    audit.add_argument("--output-json", type=Path, default=None)
    audit.add_argument("--output-md", type=Path, default=None)

    return parser.parse_args()


def _print_detection_report(report: SecRLDetectionReport) -> None:
    print(f"scope={report.scope} ground_truth={report.ground_truth_source} rows={report.rows_read} candidates={report.candidate_alerts}")
    print(f"ground_truth_alerts={report.ground_truth_alerts} detected_alerts={report.detected_alerts}")
    print(f"available_ground_truth_alerts={report.available_ground_truth_alerts}")
    print(f"precision={report.precision:.3f} recall={report.recall:.3f} available_recall={report.available_recall:.3f} f1={report.f1:.3f}")
    print(f"incident_recall_any={report.incident_recall_any:.3f}")
    print(f"incident_recall_all={report.incident_recall_all:.3f}")
    print(f"mean_incident_alert_recall={report.mean_incident_alert_recall:.3f}")
    print(f"alerts_per_second={report.alerts_per_second:.0f}")


def _print_discovery_report(report: SecRLIncidentDiscoveryReport) -> None:
    print(f"scope={report.scope} ground_truth={report.ground_truth_source}")
    print(f"refinements={','.join(report.refinements) or 'baseline'}")
    print(f"candidate_alerts={report.candidate_alerts} stored_alerts={report.stored_alerts}")
    print(
        "ground_truth_incidents="
        f"{report.ground_truth_incidents} detected={report.detected_ground_truth_incidents} "
        f"missed={report.missed_ground_truth_incidents}"
    )
    print(
        f"incident_recall_any={report.incident_recall_any:.3f} "
        f"mean_alert_recall={report.mean_ground_truth_alert_recall:.3f} "
        f"mean_available_recall={report.mean_available_alert_recall:.3f}"
    )
    print(f"clusters_considered={report.clusters_considered} clusters_reported={report.clusters_reported}")
    print(f"known_overlap_clusters={report.known_overlap_clusters}")
    print(f"candidate_new_incidents={report.candidate_new_incidents} candidate_new_alerts={report.candidate_new_alerts}")
    for incident in report.incidents[:10]:
        print(
            f"- {incident.cluster_id} {incident.status} score={incident.score:.3f} "
            f"alerts={incident.alert_count} tags={','.join(incident.attack_tags) or '-'}"
        )


def _print_ablation_report(report: SecRLDiscoveryAblationReport) -> None:
    print(f"scope={report.scope} ground_truth={report.ground_truth_source}")
    print(
        f"baseline_candidate_new={report.baseline_candidate_new_incidents} "
        f"baseline_candidate_new_alerts={report.baseline_candidate_new_alerts} "
        f"baseline_incident_recall={report.baseline_incident_recall_any:.3f}"
    )
    for row in report.rows:
        print(
            f"- {row.variant}: candidate_new={row.candidate_new_incidents} "
            f"candidate_new_alerts={row.candidate_new_alerts} "
            f"delta_alerts={row.candidate_new_alert_reduction} "
            f"incident_recall={row.incident_recall_any:.3f}"
        )


def _print_baseline_report(report: SecRLDiscoveryBaselineReport) -> None:
    print(f"scope={report.scope} ground_truth={report.ground_truth_source}")
    print(
        f"candidate_alerts={report.candidate_alerts} "
        f"ground_truth_alerts={report.ground_truth_alerts} "
        f"available_ground_truth_alerts={report.available_ground_truth_alerts}"
    )
    for row in report.rows:
        print(
            f"- {row.baseline}: selected={row.selected_alerts} "
            f"alert_p/r/f1={row.alert_precision:.3f}/{row.alert_recall:.3f}/{row.alert_f1:.3f} "
            f"incident_recall={row.incident_recall_any:.3f} "
            f"mean_gt_alert_recall={row.mean_ground_truth_alert_recall:.3f} "
            f"gt_clusters={row.gt_overlap_clusters} non_gt_clusters={row.non_gt_clusters} "
            f"candidate_new={row.candidate_new_incidents}"
        )


def _print_leakage_audit_report(report: SecRLLeakageAuditReport) -> None:
    print(f"scope={report.scope} ground_truth={report.ground_truth_source}")
    print(f"refinements={','.join(report.refinements) or 'baseline'}")
    print(
        f"candidate_alerts={report.candidate_alerts} stored_alerts={report.stored_alerts} "
        f"normal_clusters={report.normal_reported_clusters} blind_clusters={report.blind_reported_clusters}"
    )
    print(
        f"cluster_generation_stable={report.cluster_generation_stable} "
        f"score_generation_stable={report.score_generation_stable} "
        f"potential_leakage_detected={report.potential_leakage_detected}"
    )
    print(
        f"normal_candidate_new={report.normal_candidate_new_incidents} "
        f"blind_candidate_incidents={report.blind_candidate_incidents} "
        f"posthoc_label_delta={report.posthoc_label_delta}"
    )
    for check in report.checks:
        print(f"- {check.name}: {'pass' if check.passed else 'fail'} - {check.detail}")


def _refinements_from_args(args: argparse.Namespace) -> frozenset[str]:
    refinements = set(args.refinement or [])
    if args.profile == "refined":
        refinements.update(DISCOVERY_REFINEMENTS)
    return _normalize_refinements(refinements)


def main() -> None:
    args = _parse_args()
    if args.command == "download":
        path = download_secrl_raw_logs(args.secrl_root, extract=not args.no_extract, force=args.force)
        print(path)
        return

    if args.command == "analyze-errors":
        report = analyze_secrl_errors(
            args.data_root,
            scope=args.scope,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
            limit=args.limit,
            max_examples=args.max_examples,
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(report.to_markdown(), encoding="utf-8")
        print(f"scope={report.scope} ground_truth={report.ground_truth_source}")
        print(f"false_positives={report.false_positives} false_negatives={report.false_negatives}")
        print(f"false_positives_with_security_incident_ref={report.false_positives_with_security_incident_ref}")
        for conclusion in report.conclusions:
            print(f"- {conclusion}")
        return

    if args.command == "ablate-discovery":
        report = run_secrl_discovery_ablation(
            args.data_root,
            scope=args.scope,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
            limit=args.limit,
            max_gap_minutes=args.max_gap_minutes,
            min_alerts=args.min_alerts,
            min_score=args.min_score,
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(report.to_markdown(), encoding="utf-8")
        _print_ablation_report(report)
        return

    if args.command == "compare-baselines":
        report = compare_secrl_discovery_baselines(
            args.data_root,
            scope=args.scope,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
            limit=args.limit,
            max_gap_minutes=args.max_gap_minutes,
            min_alerts=args.min_alerts,
            min_score=args.min_score,
            random_seed=args.random_seed,
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(report.to_markdown(), encoding="utf-8")
        _print_baseline_report(report)
        return

    if args.command == "audit-leakage":
        report = audit_secrl_discovery_leakage(
            args.data_root,
            scope=args.scope,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
            limit=args.limit,
            max_gap_minutes=args.max_gap_minutes,
            min_alerts=args.min_alerts,
            min_score=args.min_score,
            refinements=_refinements_from_args(args),
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(report.to_markdown(), encoding="utf-8")
        _print_leakage_audit_report(report)
        return

    if args.command == "discover-incidents":
        report = discover_secrl_incidents(
            args.data_root,
            scope=args.scope,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
            limit=args.limit,
            max_gap_minutes=args.max_gap_minutes,
            min_alerts=args.min_alerts,
            min_score=args.min_score,
            max_examples=args.max_examples,
            refinements=_refinements_from_args(args),
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(report.to_markdown(), encoding="utf-8")
        _print_discovery_report(report)
        return

    report = asyncio.run(
        evaluate_secrl_alert_detection(
            args.data_root,
            scope=args.scope,
            limit=args.limit,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    _print_detection_report(report)


if __name__ == "__main__":
    main()
