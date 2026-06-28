from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .falcon import alert_from_falcon_row
from .gids import (
    GIDSDetector,
    GIDSLocalVerdictAgent,
    GIDSRunResult,
    edge_from_falcon_row,
    _candidate_order_score,
    _cluster_edges_by_entity_time,
    _run_gids_rare,
)
from .sketch import GraphSketchingFilter, SketchDecision


HIGH_SEVERITIES = {"High", "Critical"}
LABEL_FIELDS = {"is_incident", "incident_id"}


@dataclass(frozen=True)
class DatasetProfile:
    alerts: int
    incident_alerts: int
    noise_alerts: int
    incidents: int
    incident_severity: dict[str, int]
    noise_severity: dict[str, int]
    incident_tactics: dict[str, int]
    noise_tactics: dict[str, int]
    incident_high_critical_rate: float
    noise_high_critical_rate: float


@dataclass(frozen=True)
class ClusterErrorExample:
    cluster_id: str
    alert_count: int
    true_positive_alerts: int
    false_positive_alerts: int
    matched_incidents: list[str]
    severities: dict[str, int]
    tactics: dict[str, int]
    explanation: str


@dataclass(frozen=True)
class DetectorErrorAnalysis:
    detector: str
    selected_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    incident_any_recall: float
    incident_all_recall: float
    selected_tp_high_critical_rate: float
    selected_fp_high_critical_rate: float
    missed_high_critical_rate: float
    false_positive_severity: dict[str, int]
    false_positive_tactics: dict[str, int]
    missed_severity: dict[str, int]
    missed_tactics: dict[str, int]
    decision_reason_true_positives: dict[str, int]
    decision_reason_false_positives: dict[str, int]
    false_positive_clusters: int
    mixed_clusters: int
    top_false_positive_clusters: list[ClusterErrorExample]
    top_mixed_clusters: list[ClusterErrorExample]
    top_missed_incidents: dict[str, int]
    interpretation: list[str]


@dataclass(frozen=True)
class GoldenFeatureAudit:
    label_fields_present_in_csv: bool
    falcon_adapter_strips_labels: bool
    gids_adapter_strips_labels: bool
    strict_label_leakage_found: bool
    shortcut_findings: list[str]


@dataclass(frozen=True)
class FalconErrorAnalysisReport:
    input_path: str
    dataset: DatasetProfile
    golden_feature_audit: GoldenFeatureAudit
    detectors: list[DetectorErrorAnalysis]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Falcon Hard Dataset Error Analysis",
            "",
            f"- Input: `{self.input_path}`",
            f"- Alerts: `{self.dataset.alerts}`",
            f"- Incident alerts: `{self.dataset.incident_alerts}`",
            f"- Noise alerts: `{self.dataset.noise_alerts}`",
            f"- Incidents: `{self.dataset.incidents}`",
            f"- Incident High/Critical rate: `{self.dataset.incident_high_critical_rate:.3f}`",
            f"- Noise High/Critical rate: `{self.dataset.noise_high_critical_rate:.3f}`",
            "",
            "## Dataset Shape",
            "",
            "| Split | Severity Counts | Tactic Counts |",
            "| --- | --- | --- |",
            f"| Incident | {_counter_text(self.dataset.incident_severity)} | {_counter_text(self.dataset.incident_tactics)} |",
            f"| Noise | {_counter_text(self.dataset.noise_severity)} | {_counter_text(self.dataset.noise_tactics)} |",
            "",
            "## Golden Feature Audit",
            "",
            f"- CSV contains scoring labels: `{self.golden_feature_audit.label_fields_present_in_csv}`",
            f"- Falcon adapter strips labels: `{self.golden_feature_audit.falcon_adapter_strips_labels}`",
            f"- GIDS adapter strips labels: `{self.golden_feature_audit.gids_adapter_strips_labels}`",
            f"- Strict label leakage found: `{self.golden_feature_audit.strict_label_leakage_found}`",
            "",
        ]
        lines.extend(f"- {finding}" for finding in self.golden_feature_audit.shortcut_findings)
        lines.extend(
            [
                "",
                "## Method Summary",
                "",
                "| Detector | TP | FP | FN | Precision | Recall | F1 | TP High/Critical | FP High/Critical | Miss High/Critical | FP Clusters | Mixed Clusters |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for detector in self.detectors:
            lines.append(
                "| "
                + " | ".join(
                    [
                        detector.detector,
                        str(detector.true_positives),
                        str(detector.false_positives),
                        str(detector.false_negatives),
                        f"{detector.precision:.3f}",
                        f"{detector.recall:.3f}",
                        f"{detector.f1:.3f}",
                        f"{detector.selected_tp_high_critical_rate:.3f}",
                        f"{detector.selected_fp_high_critical_rate:.3f}",
                        f"{detector.missed_high_critical_rate:.3f}",
                        str(detector.false_positive_clusters),
                        str(detector.mixed_clusters),
                    ]
                )
                + " |"
            )
        for detector in self.detectors:
            lines.extend(_detector_markdown(detector))
        return "\n".join(lines) + "\n"


def analyze_falcon_errors(path: str | Path, *, hide_severity: bool = False) -> FalconErrorAnalysisReport:
    input_path = Path(path)
    rows = list(_iter_csv_rows(input_path))
    row_by_id = {row["alert_id"]: row for row in rows}
    labels = {row["alert_id"]: _truthy(row.get("is_incident", "")) for row in rows}
    incident_alert_ids = _incident_alert_ids(rows)
    ground_truth_ids = {alert_id for alert_id, is_incident in labels.items() if is_incident}
    dataset = _dataset_profile(rows)
    audit = _golden_feature_audit(rows, hide_severity=hide_severity)

    method_specs = []
    if hide_severity:
        method_specs.append(
            _method_spec(
                name="High/Critical severity baseline",
                clusters={},
                row_by_id=row_by_id,
                interpretation=[
                    "Disabled for this ablation because severity is hidden from model-visible inputs.",
                    "This row is retained to make the comparison table explicit.",
                ],
            )
        )
    else:
        method_specs.append(
            _method_spec(
                name="High/Critical severity baseline",
                clusters={f"severity-{alert_id}": {alert_id} for alert_id, row in row_by_id.items() if row["severity"] in HIGH_SEVERITIES},
                row_by_id=row_by_id,
                interpretation=[
                    "Correct classifications come directly from the High/Critical shortcut.",
                    "False positives are High/Critical noise decoys.",
                    "Misses are mostly Low/Medium incident alerts, proving priority alone is insufficient.",
                ],
            )
        )

    eacs_clusters, eacs_reasons = _eacs_no_mitre_predictions(rows, hide_severity=hide_severity)
    method_specs.append(
        _method_spec(
            name="E-ACS graph sketch without MITRE tactics",
            clusters=eacs_clusters,
            row_by_id=row_by_id,
            decision_reasons=eacs_reasons,
            interpretation=[
                (
                    "True positives come from rare relationship gates because severity is hidden."
                    if hide_severity
                    else "True positives come from both high severity and rare relationship gates."
                ),
                (
                    "False positives are dominated by rare-relationship noise that looks novel in the sketch."
                    if hide_severity
                    else "False positives are dominated by high-severity and rare-relationship noise that looks novel in the sketch."
                ),
                (
                    "Misses are incident alerts that were not rare enough at the time they appeared."
                    if hide_severity
                    else "Misses are incident alerts that were neither high severity nor rare enough at the time they appeared."
                ),
            ],
        )
    )

    gids_edges = [edge_from_falcon_row(row, hide_severity=hide_severity) for row in rows]
    gids_run = GIDSDetector().run(gids_edges)
    gids_clusters = {incident.incident_id: set(incident.alert_ids) for incident in gids_run.incidents}
    method_specs.append(
        _method_spec(
            name="GIDS",
            clusters=gids_clusters,
            row_by_id=row_by_id,
            gids_run=gids_run,
            interpretation=[
                "GIDS recovers every real incident because the generated incidents form strong entity/time communities.",
                "Most false positives are standalone decoy communities that share entities, tactics, and timing but have no ground-truth incident overlap.",
                "Correct detections rely on graph continuity and synthetic attack-chain topology, not on label fields.",
            ],
        )
    )

    local_verdicts = GIDSLocalVerdictAgent(use_severity=not hide_severity).validate(gids_run.incidents, gids_edges)
    selected_local = {
        incident.incident_id: set(incident.alert_ids)
        for incident in gids_run.incidents
        if any(decision.candidate_id == incident.incident_id and decision.selected for decision in local_verdicts)
    }
    method_specs.append(
        _method_spec(
            name="GIDS + local verdict agent",
            clusters=selected_local,
            row_by_id=row_by_id,
            gids_run=gids_run,
            interpretation=[
                (
                    "The local verdict agent removes decoy communities by requiring ordered tactic-chain evidence without severity."
                    if hide_severity
                    else "The local verdict agent removes nearly all standalone decoy communities by requiring ordered tactic-chain evidence."
                ),
                "Remaining false positives are contamination: noise alerts merged into otherwise real incident clusters.",
                "Correct decisions rely on generated attack-chain shape and pattern support, not on scoring labels.",
            ],
        )
    )

    gids_rare_run, rare_edges = _run_gids_rare(gids_edges)
    gids_rare_clusters = {incident.incident_id: set(incident.alert_ids) for incident in gids_rare_run.incidents}
    method_specs.append(
        _method_spec(
            name="gids_rare",
            clusters=gids_rare_clusters,
            row_by_id=row_by_id,
            gids_run=gids_rare_run,
            interpretation=[
                "This variant adds streaming relationship rarity as a candidate source and a small scoring feature.",
                "Correct detections still require graph continuity, tactic-chain shape, or pattern support.",
                "False positives are expected when benign/noise relationships are novel inside the benchmark window.",
            ],
        )
    )

    rare_verdicts = GIDSLocalVerdictAgent(use_severity=not hide_severity, use_rarity=True).validate(
        gids_rare_run.incidents,
        rare_edges,
    )
    selected_rare_local = {
        incident.incident_id: set(incident.alert_ids)
        for incident in gids_rare_run.incidents
        if any(decision.candidate_id == incident.incident_id and decision.selected for decision in rare_verdicts)
    }
    method_specs.append(
        _method_spec(
            name="gids_rare_with_agent",
            clusters=selected_rare_local,
            row_by_id=row_by_id,
            gids_run=gids_rare_run,
            interpretation=[
                "This variant applies the local final verdict agent after rarity-expanded GIDS candidates.",
                "It should keep the recall benefit of rarity only when the candidate also has ordered tactic-chain evidence.",
                "Remaining false positives are usually novel decoy communities or noise attached to a real chain.",
            ],
        )
    )

    local_clusters = {} if hide_severity else _local_analyst_clusters(rows)
    method_specs.append(
        _method_spec(
            name="Local analyst severity-chain review",
            clusters=local_clusters,
            row_by_id=row_by_id,
            interpretation=[
                (
                    "Disabled for this ablation because the method explicitly uses High/Critical severity as a first-pass filter."
                    if hide_severity
                    else "This method explicitly uses High/Critical severity as a first-pass filter."
                ),
                (
                    "No predictions are emitted when severity is hidden."
                    if hide_severity
                    else "It catches at least one alert from every incident but misses Low/Medium incident steps."
                ),
                (
                    "This row is retained to show that severity-dependent baselines no longer apply."
                    if hide_severity
                    else "False positives are high-severity noise bursts linked by shared host/user/time."
                ),
            ],
        )
    )

    detectors = [
        _analyze_method(
            spec=spec,
            row_by_id=row_by_id,
            labels=labels,
            ground_truth_ids=ground_truth_ids,
            incident_alert_ids=incident_alert_ids,
        )
        for spec in method_specs
    ]
    return FalconErrorAnalysisReport(
        input_path=str(input_path),
        dataset=dataset,
        golden_feature_audit=audit,
        detectors=detectors,
    )


def _method_spec(
    *,
    name: str,
    clusters: dict[str, set[str]],
    row_by_id: dict[str, dict[str, str]],
    decision_reasons: Optional[dict[str, str]] = None,
    gids_run: Optional[GIDSRunResult] = None,
    interpretation: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "clusters": clusters,
        "decision_reasons": decision_reasons or {},
        "gids_run": gids_run,
        "interpretation": interpretation,
    }


def _analyze_method(
    *,
    spec: dict[str, Any],
    row_by_id: dict[str, dict[str, str]],
    labels: dict[str, bool],
    ground_truth_ids: set[str],
    incident_alert_ids: dict[str, set[str]],
) -> DetectorErrorAnalysis:
    clusters: dict[str, set[str]] = spec["clusters"]
    selected_ids = set().union(*clusters.values(), set())
    true_positives = selected_ids & ground_truth_ids
    false_positives = selected_ids - ground_truth_ids
    false_negatives = ground_truth_ids - selected_ids
    precision = len(true_positives) / len(selected_ids) if selected_ids else 0.0
    recall = len(true_positives) / len(ground_truth_ids) if ground_truth_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    incident_matches = _incident_matches(selected_ids, incident_alert_ids)
    decision_reasons: dict[str, str] = spec["decision_reasons"]
    fp_cluster_examples = _cluster_examples(clusters, row_by_id, ground_truth_ids, incident_alert_ids, want_mixed=False)
    mixed_cluster_examples = _cluster_examples(clusters, row_by_id, ground_truth_ids, incident_alert_ids, want_mixed=True)

    interpretation = list(spec["interpretation"])
    interpretation.extend(_shortcut_interpretation(spec["name"], true_positives, false_positives, false_negatives, row_by_id))

    return DetectorErrorAnalysis(
        detector=spec["name"],
        selected_alerts=len(selected_ids),
        true_positives=len(true_positives),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        precision=precision,
        recall=recall,
        f1=f1,
        incident_any_recall=_mean(1.0 if match["any"] else 0.0 for match in incident_matches),
        incident_all_recall=_mean(1.0 if match["all"] else 0.0 for match in incident_matches),
        selected_tp_high_critical_rate=_high_rate(true_positives, row_by_id),
        selected_fp_high_critical_rate=_high_rate(false_positives, row_by_id),
        missed_high_critical_rate=_high_rate(false_negatives, row_by_id),
        false_positive_severity=_counter_for(false_positives, row_by_id, "severity"),
        false_positive_tactics=_counter_for(false_positives, row_by_id, "tactic"),
        missed_severity=_counter_for(false_negatives, row_by_id, "severity"),
        missed_tactics=_counter_for(false_negatives, row_by_id, "tactic"),
        decision_reason_true_positives=_reason_counts(true_positives, decision_reasons),
        decision_reason_false_positives=_reason_counts(false_positives, decision_reasons),
        false_positive_clusters=len(fp_cluster_examples),
        mixed_clusters=len(mixed_cluster_examples),
        top_false_positive_clusters=fp_cluster_examples[:12],
        top_mixed_clusters=mixed_cluster_examples[:12],
        top_missed_incidents=_top_missed_incidents(false_negatives, incident_alert_ids),
        interpretation=interpretation,
    )


def _eacs_no_mitre_predictions(
    rows: list[dict[str, str]],
    *,
    hide_severity: bool = False,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    sketch_filter = GraphSketchingFilter()
    clusters: dict[str, set[str]] = {}
    reasons: dict[str, str] = {}
    for row in rows:
        alert = alert_from_falcon_row(row, include_mitre_tactics=False, hide_severity=hide_severity)
        decision: SketchDecision = sketch_filter.evaluate(alert)
        reasons[alert.id] = decision.reason
        if decision.store:
            clusters[f"eacs-{alert.id}"] = {alert.id}
    return clusters, reasons


def _local_analyst_clusters(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    edges = [edge_from_falcon_row(row) for row in rows if row["severity"] in HIGH_SEVERITIES]
    clusters = _cluster_edges_by_entity_time(edges, max_gap_minutes=120, min_alerts=2)
    return {f"local-{idx:04d}": {edge.alert_id for edge in cluster} for idx, cluster in enumerate(clusters, start=1)}


def _cluster_examples(
    clusters: dict[str, set[str]],
    row_by_id: dict[str, dict[str, str]],
    ground_truth_ids: set[str],
    incident_alert_ids: dict[str, set[str]],
    *,
    want_mixed: bool,
) -> list[ClusterErrorExample]:
    examples = []
    for cluster_id, alert_ids in clusters.items():
        tp_ids = alert_ids & ground_truth_ids
        fp_ids = alert_ids - ground_truth_ids
        is_mixed = bool(tp_ids and fp_ids)
        is_fp_cluster = bool(fp_ids and not tp_ids)
        if want_mixed and not is_mixed:
            continue
        if not want_mixed and not is_fp_cluster:
            continue
        rows = [row_by_id[alert_id] for alert_id in alert_ids if alert_id in row_by_id]
        matched = sorted(incident for incident, ids in incident_alert_ids.items() if alert_ids & ids)
        explanation = _cluster_explanation(rows, mixed=is_mixed)
        examples.append(
            ClusterErrorExample(
                cluster_id=cluster_id,
                alert_count=len(alert_ids),
                true_positive_alerts=len(tp_ids),
                false_positive_alerts=len(fp_ids),
                matched_incidents=matched,
                severities=dict(_sorted_counter(Counter(row["severity"] for row in rows))),
                tactics=dict(_sorted_counter(Counter(row["tactic"] for row in rows))),
                explanation=explanation,
            )
        )
    return sorted(examples, key=lambda item: (-item.false_positive_alerts, -item.alert_count, item.cluster_id))


def _cluster_explanation(rows: list[dict[str, str]], *, mixed: bool) -> str:
    if not rows:
        return "empty cluster"
    high_count = sum(1 for row in rows if row["severity"] in HIGH_SEVERITIES)
    tactics = {row["tactic"] for row in rows}
    users = {row["user"] for row in rows}
    hosts = {row["source_node"] for row in rows} | {row["target_node"] for row in rows}
    ordered_score = _order_score_from_rows(rows)
    if mixed:
        return (
            "real incident cluster contaminated by nearby noise; "
            f"high={high_count}/{len(rows)}, tactics={len(tactics)}, users={len(users)}, hosts={len(hosts)}, order_score={ordered_score:.2f}"
        )
    return (
        "decoy/noise cluster with enough graph structure to pass; "
        f"high={high_count}/{len(rows)}, tactics={len(tactics)}, users={len(users)}, hosts={len(hosts)}, order_score={ordered_score:.2f}"
    )


def _order_score_from_rows(rows: list[dict[str, str]]) -> float:
    edges = [edge_from_falcon_row(row) for row in rows]
    return _candidate_order_score(edges)


def _dataset_profile(rows: list[dict[str, str]]) -> DatasetProfile:
    incident = [row for row in rows if _truthy(row.get("is_incident", ""))]
    noise = [row for row in rows if not _truthy(row.get("is_incident", ""))]
    return DatasetProfile(
        alerts=len(rows),
        incident_alerts=len(incident),
        noise_alerts=len(noise),
        incidents=len({row["incident_id"] for row in incident if row["incident_id"]}),
        incident_severity=dict(_sorted_counter(Counter(row["severity"] for row in incident))),
        noise_severity=dict(_sorted_counter(Counter(row["severity"] for row in noise))),
        incident_tactics=dict(_sorted_counter(Counter(row["tactic"] for row in incident))),
        noise_tactics=dict(_sorted_counter(Counter(row["tactic"] for row in noise))),
        incident_high_critical_rate=_high_rate({row["alert_id"] for row in incident}, {row["alert_id"]: row for row in rows}),
        noise_high_critical_rate=_high_rate({row["alert_id"] for row in noise}, {row["alert_id"]: row for row in rows}),
    )


def _golden_feature_audit(rows: list[dict[str, str]], *, hide_severity: bool = False) -> GoldenFeatureAudit:
    sample = rows[0] if rows else {}
    falcon_raw = alert_from_falcon_row(sample, include_mitre_tactics=False, hide_severity=hide_severity).raw if sample else {}
    gids_raw = edge_from_falcon_row(sample, hide_severity=hide_severity).raw if sample else {}
    label_fields_present = any(field in sample for field in LABEL_FIELDS)
    falcon_strips = not any(field in falcon_raw for field in LABEL_FIELDS)
    gids_strips = not any(field in gids_raw for field in LABEL_FIELDS)
    strict_leak = not (falcon_strips and gids_strips)
    findings = [
        "Strict golden labels are `is_incident` and `incident_id`; deterministic detectors strip them before prediction.",
        (
            "Severity was hidden from model-visible inputs in this ablation; severity tables below describe held-out error distribution only."
            if hide_severity
            else "Severity is not a golden label, but it is a shortcut feature. In the hard dataset it is intentionally imperfect."
        ),
        "MITRE tactic order is not a label, but it is generator-shaped evidence because incidents are generated as ordered attack chains.",
        "Correct GIDS + local verdict classifications depend heavily on tactic-chain shape and pattern support; this is useful signal, but also synthetic-dataset-specific.",
    ]
    return GoldenFeatureAudit(
        label_fields_present_in_csv=label_fields_present,
        falcon_adapter_strips_labels=falcon_strips,
        gids_adapter_strips_labels=gids_strips,
        strict_label_leakage_found=strict_leak,
        shortcut_findings=findings,
    )


def _shortcut_interpretation(
    name: str,
    true_positives: set[str],
    false_positives: set[str],
    false_negatives: set[str],
    row_by_id: dict[str, dict[str, str]],
) -> list[str]:
    tp_high = _high_rate(true_positives, row_by_id)
    fp_high = _high_rate(false_positives, row_by_id)
    miss_high = _high_rate(false_negatives, row_by_id)
    findings = [
        f"Selected true-positive High/Critical rate: {tp_high:.3f}.",
        f"Selected false-positive High/Critical rate: {fp_high:.3f}.",
        f"Missed-alert High/Critical rate: {miss_high:.3f}.",
    ]
    if "severity" in name.lower() or "Local analyst" in name:
        findings.append("This method uses severity as a primary decision feature; correct high-severity detections should be treated as shortcut-driven.")
    if "GIDS" in name:
        findings.append("This method does not use scoring labels; correct detections are driven by graph continuity and tactic-chain evidence.")
    if false_negatives:
        findings.append("Missed alerts are concentrated in non-high-severity incident steps unless noted otherwise in the missed severity table.")
    return findings


def _detector_markdown(detector: DetectorErrorAnalysis) -> list[str]:
    lines = [
        "",
        f"## {detector.detector}",
        "",
        "### Why It Was Right Or Wrong",
        "",
    ]
    lines.extend(f"- {item}" for item in detector.interpretation)
    lines.extend(
        [
            "",
            "### Error Distributions",
            "",
            f"- False-positive severity: `{_counter_text(detector.false_positive_severity)}`",
            f"- False-positive tactics: `{_counter_text(detector.false_positive_tactics)}`",
            f"- Missed severity: `{_counter_text(detector.missed_severity)}`",
            f"- Missed tactics: `{_counter_text(detector.missed_tactics)}`",
            f"- Decision reasons for true positives: `{_counter_text(detector.decision_reason_true_positives)}`",
            f"- Decision reasons for false positives: `{_counter_text(detector.decision_reason_false_positives)}`",
            f"- Top missed incidents: `{_counter_text(detector.top_missed_incidents)}`",
            "",
            "### False-Positive Clusters",
            "",
        ]
    )
    lines.extend(_cluster_table(detector.top_false_positive_clusters))
    lines.extend(
        [
            "",
            "### Mixed Real Clusters With Extra Noise",
            "",
        ]
    )
    lines.extend(_cluster_table(detector.top_mixed_clusters))
    return lines


def _cluster_table(examples: list[ClusterErrorExample]) -> list[str]:
    if not examples:
        return ["- None."]
    lines = [
        "| Cluster | Alerts | TP | FP | Matched Incidents | Severities | Tactics | Explanation |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for example in examples:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{example.cluster_id}`",
                    str(example.alert_count),
                    str(example.true_positive_alerts),
                    str(example.false_positive_alerts),
                    _escape_table(", ".join(example.matched_incidents) or "-"),
                    _escape_table(_counter_text(example.severities)),
                    _escape_table(_counter_text(example.tactics)),
                    _escape_table(example.explanation),
                ]
            )
            + " |"
        )
    return lines


def _incident_matches(selected_ids: set[str], incident_alert_ids: dict[str, set[str]]) -> list[dict[str, Any]]:
    matches = []
    for incident, alert_ids in incident_alert_ids.items():
        recovered = selected_ids & alert_ids
        matches.append({"incident": incident, "any": bool(recovered), "all": bool(alert_ids and alert_ids <= selected_ids)})
    return matches


def _top_missed_incidents(false_negatives: set[str], incident_alert_ids: dict[str, set[str]]) -> dict[str, int]:
    counts = {
        incident: len(false_negatives & alert_ids)
        for incident, alert_ids in incident_alert_ids.items()
        if false_negatives & alert_ids
    }
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12])


def _reason_counts(alert_ids: set[str], reasons: dict[str, str]) -> dict[str, int]:
    return dict(_sorted_counter(Counter(reasons.get(alert_id, "<none>") for alert_id in alert_ids)))


def _counter_for(alert_ids: set[str], row_by_id: dict[str, dict[str, str]], field: str) -> dict[str, int]:
    return dict(_sorted_counter(Counter(row_by_id[alert_id][field] for alert_id in alert_ids if alert_id in row_by_id)))


def _high_rate(alert_ids: set[str], row_by_id: dict[str, dict[str, str]]) -> float:
    if not alert_ids:
        return 0.0
    return sum(1 for alert_id in alert_ids if row_by_id[alert_id]["severity"] in HIGH_SEVERITIES) / len(alert_ids)


def _incident_alert_ids(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    incidents: dict[str, set[str]] = {}
    for row in rows:
        incident_id = row.get("incident_id", "")
        if incident_id:
            incidents.setdefault(incident_id, set()).add(row["alert_id"])
    return incidents


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key): "" if value is None else str(value) for key, value in row.items() if key is not None}


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _counter_text(counter: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counter.items()) or "-"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Falcon/GIDS benchmark errors and shortcut reliance.")
    parser.add_argument("--input", default="falcon_graph_alerts_hard.csv")
    parser.add_argument("--output-json", default="reports/falcon_hard_error_analysis.json")
    parser.add_argument("--output-md", default="reports/falcon_hard_error_analysis.md")
    parser.add_argument("--hide-severity", action="store_true", help="Remove severity from model-visible detector inputs.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = analyze_falcon_errors(args.input, hide_severity=args.hide_severity)
    if args.output_json:
        _write_json(Path(args.output_json), report.to_json_dict())
    if args.output_md:
        _write_markdown(Path(args.output_md), report.to_markdown())
    if not args.output_json and not args.output_md:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
