from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

from .sketch import CountMinSketch


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
ProgressCallback = Optional[Callable[[str], None]]
LOCAL_ATTACK_CHAINS = (
    ("Initial Access", "Execution", "Credential Access", "Lateral Movement", "Privilege Escalation", "Exfiltration"),
    ("Initial Access", "Execution", "Persistence", "Credential Access", "Lateral Movement"),
    ("Credential Access", "Execution", "Lateral Movement", "Exfiltration"),
    ("Execution", "Privilege Escalation", "Credential Access", "Lateral Movement", "Exfiltration"),
)


@dataclass(frozen=True)
class GIDSNode:
    id: str
    label: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class GIDSEdge:
    alert_id: str
    source_id: str
    target_id: str
    user_id: str
    process_name: str
    tactic: str
    technique: str
    severity: int
    timestamp: datetime
    raw: dict[str, Any]
    relationship_seen: int = 0
    rare_relationship: bool = False

    @property
    def host_ids(self) -> set[str]:
        return {self.source_id, self.target_id} - {""}

    @property
    def correlation_entities(self) -> set[str]:
        entities = {f"host:{host}" for host in self.host_ids}
        if self.user_id:
            entities.add(f"user:{self.user_id}")
        return entities


@dataclass(frozen=True)
class GIDSSubgraphMatch:
    pattern: str
    alert_ids: list[str]
    rationale: str


@dataclass(frozen=True)
class GIDSIncident:
    incident_id: str
    alert_ids: list[str]
    start_time: str
    end_time: str
    alert_count: int
    host_count: int
    user_count: int
    max_severity: int
    tactics: list[str]
    pattern_matches: list[str]
    structural_score: float
    confidence: float
    narrative: str


@dataclass(frozen=True)
class GIDSRunResult:
    candidate_alerts: int
    pattern_matches: int
    communities_considered: int
    incidents: list[GIDSIncident]
    elapsed_seconds: float
    rare_relationships: int = 0


@dataclass(frozen=True)
class DetectorEvaluation:
    detector: str
    status: str
    selected_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    clusters_reported: int
    known_overlap_clusters: int
    candidate_new_clusters: int
    incident_recall_any: float
    incident_recall_all: float
    mean_incident_alert_recall: float
    notes: list[str]


@dataclass(frozen=True)
class LLMRationalizerDecision:
    candidate_id: str
    verdict: str
    confidence: float
    selected: bool
    rationale: str
    remediation_steps: list[str]


@dataclass(frozen=True)
class GIDSComparisonReport:
    input_path: str
    candidate_alerts: int
    ground_truth_alerts: int
    ground_truth_incidents: int
    gids_run: GIDSRunResult
    detectors: list[DetectorEvaluation]
    rationalizer_decisions: list[LLMRationalizerDecision]
    local_verdict_decisions: list[LLMRationalizerDecision]
    notes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# GIDS vs Plain Gemini Evaluation",
            "",
            f"- Input: `{self.input_path}`",
            f"- Candidate alerts: `{self.candidate_alerts}`",
            f"- Ground-truth incident alerts: `{self.ground_truth_alerts}`",
            f"- Ground-truth incidents: `{self.ground_truth_incidents}`",
            f"- GIDS pattern matches: `{self.gids_run.pattern_matches}`",
            f"- GIDS communities considered: `{self.gids_run.communities_considered}`",
            f"- GIDS runtime: `{self.gids_run.elapsed_seconds:.4f}` seconds",
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
        if self.rationalizer_decisions:
            lines.extend(
                [
                    "",
                    "## GIDS Gemini Rationalizer",
                    "",
                    "| Candidate | Selected | Verdict | Confidence | Rationale | Remediation |",
                    "| --- | --- | --- | ---: | --- | --- |",
                ]
            )
            for decision in self.rationalizer_decisions[:30]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{decision.candidate_id}`",
                            "yes" if decision.selected else "no",
                            _escape_table(decision.verdict),
                            f"{decision.confidence:.3f}",
                            _escape_table(decision.rationale),
                            _escape_table("; ".join(decision.remediation_steps) or "-"),
                        ]
                    )
                    + " |"
                )
        if self.local_verdict_decisions:
            lines.extend(
                [
                    "",
                    "## GIDS Local Verdict Agent",
                    "",
                    "| Candidate | Selected | Verdict | Confidence | Rationale | Actions |",
                    "| --- | --- | --- | ---: | --- | --- |",
                ]
            )
            for decision in self.local_verdict_decisions[:40]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{decision.candidate_id}`",
                            "yes" if decision.selected else "no",
                            _escape_table(decision.verdict),
                            f"{decision.confidence:.3f}",
                            _escape_table(decision.rationale),
                            _escape_table("; ".join(decision.remediation_steps) or "-"),
                        ]
                    )
                    + " |"
                )
        lines.extend(
            [
                "",
                "## GIDS Incidents",
                "",
                "| Incident | Alerts | Hosts | Max Severity | Score | Confidence | Patterns | Tactics |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for incident in self.gids_run.incidents[:30]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{incident.incident_id}`",
                        str(incident.alert_count),
                        str(incident.host_count),
                        str(incident.max_severity),
                        f"{incident.structural_score:.3f}",
                        f"{incident.confidence:.3f}",
                        _escape_table(", ".join(incident.pattern_matches) or "-"),
                        _escape_table(", ".join(incident.tactics) or "-"),
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"


class GIDSGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, GIDSNode] = {}
        self.edges: list[GIDSEdge] = []

    def ingest(self, edge: GIDSEdge) -> None:
        self.edges.append(edge)
        self.nodes.setdefault(
            f"host:{edge.source_id}",
            GIDSNode(id=f"host:{edge.source_id}", label="Host", properties={"aid": edge.source_id}),
        )
        self.nodes.setdefault(
            f"host:{edge.target_id}",
            GIDSNode(id=f"host:{edge.target_id}", label="Host", properties={"aid": edge.target_id}),
        )
        if edge.user_id:
            self.nodes.setdefault(
                f"user:{edge.user_id}",
                GIDSNode(id=f"user:{edge.user_id}", label="User", properties={"sid": edge.user_id}),
            )
        if edge.process_name:
            self.nodes.setdefault(
                f"process:{edge.process_name}",
                GIDSNode(id=f"process:{edge.process_name}", label="Process", properties={"name": edge.process_name}),
            )


class SubgraphMatchingEngine:
    """Deterministic GIDS layer 1 pattern matcher."""

    PATTERNS = {
        ("Credential Access", "Execution"): "credential_execution",
        ("Credential Access", "Lateral Movement"): "credential_lateral_movement",
        ("Initial Access", "Lateral Movement"): "initial_lateral_movement",
        ("Privilege Escalation", "Exfiltration"): "privilege_exfiltration",
        ("Lateral Movement", "Exfiltration"): "lateral_exfiltration",
    }

    def __init__(self, max_gap_minutes: int = 180) -> None:
        self.max_gap_seconds = max_gap_minutes * 60

    def match(self, edges: list[GIDSEdge]) -> list[GIDSSubgraphMatch]:
        ordered = sorted(edges, key=lambda edge: (edge.timestamp, edge.alert_id))
        matches: list[GIDSSubgraphMatch] = []
        seen: set[tuple[str, str, str]] = set()

        for idx, later in enumerate(ordered):
            for earlier in reversed(ordered[:idx]):
                gap = (later.timestamp - earlier.timestamp).total_seconds()
                if gap < 0:
                    continue
                if gap > self.max_gap_seconds:
                    break
                pattern = self.PATTERNS.get((earlier.tactic, later.tactic))
                if not pattern or not (earlier.correlation_entities & later.correlation_entities):
                    continue
                key = (pattern, earlier.alert_id, later.alert_id)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    GIDSSubgraphMatch(
                        pattern=pattern,
                        alert_ids=[earlier.alert_id, later.alert_id],
                        rationale=f"{earlier.tactic} -> {later.tactic} within {int(gap // 60)}m on a shared entity",
                    )
                )
        return matches


class GIDSRelationshipNovelty:
    """Streaming rare-relationship annotator for GIDS edges."""

    def __init__(
        self,
        sketch: Optional[CountMinSketch] = None,
        max_seen: int = 3,
        max_gap_minutes: int = 180,
        min_alerts: int = 4,
    ) -> None:
        self.sketch = sketch or CountMinSketch()
        self.max_seen = max_seen
        self.max_gap_minutes = max_gap_minutes
        self.min_alerts = min_alerts

    def seed_baseline(self, relationship_key: str, count: int) -> None:
        self.sketch.add(relationship_key, count)

    def annotate(self, edges: list[GIDSEdge]) -> list[GIDSEdge]:
        annotated = []
        for edge in sorted(edges, key=lambda item: (item.timestamp, item.alert_id)):
            key = self.relationship_key(edge)
            seen = self.sketch.estimate(key)
            self.sketch.add(key)
            annotated.append(replace(edge, relationship_seen=seen, rare_relationship=seen <= self.max_seen))
        return annotated

    def components(self, edges: list[GIDSEdge]) -> list[list[GIDSEdge]]:
        rare_edges = [edge for edge in edges if edge.rare_relationship]
        return _cluster_edges_by_entity_time(
            rare_edges,
            max_gap_minutes=self.max_gap_minutes,
            min_alerts=self.min_alerts,
        )

    def relationship_key(self, edge: GIDSEdge) -> str:
        behavior = edge.technique or edge.tactic or edge.process_name or "observed"
        return "|".join(
            [
                _lower_token(edge.source_id),
                _lower_token(edge.target_id),
                _lower_token(edge.tactic),
                _lower_token(behavior),
            ]
        )


class CommunityDetectionCorrelation:
    """Lightweight GIDS layer 2 entity/time community detector."""

    def __init__(self, max_gap_minutes: int = 120, min_weight: float = 0.70, min_alerts: int = 4) -> None:
        self.max_gap_seconds = max_gap_minutes * 60
        self.min_weight = min_weight
        self.min_alerts = min_alerts

    def communities(self, edges: list[GIDSEdge]) -> list[list[GIDSEdge]]:
        if not edges:
            return []
        parent = {edge.alert_id: edge.alert_id for edge in edges}
        edge_by_id = {edge.alert_id: edge for edge in edges}

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

        by_entity: dict[str, list[GIDSEdge]] = defaultdict(list)
        for edge in edges:
            for entity in edge.correlation_entities:
                by_entity[entity].append(edge)

        for entity_edges in by_entity.values():
            ordered = sorted(entity_edges, key=lambda edge: (edge.timestamp, edge.alert_id))
            for idx, edge in enumerate(ordered):
                previous_idx = idx - 1
                while previous_idx >= 0:
                    previous = ordered[previous_idx]
                    gap = (edge.timestamp - previous.timestamp).total_seconds()
                    if gap > self.max_gap_seconds:
                        break
                    weight = _edge_weight(max(edge.severity, previous.severity), gap)
                    if weight >= self.min_weight:
                        union(edge.alert_id, previous.alert_id)
                    previous_idx -= 1

        components: dict[str, list[GIDSEdge]] = defaultdict(list)
        for alert_id, edge in edge_by_id.items():
            components[find(alert_id)].append(edge)

        promoted = []
        for component in components.values():
            hosts = set().union(*(edge.host_ids for edge in component))
            if len(component) >= self.min_alerts and len(hosts) > 1:
                promoted.append(sorted(component, key=lambda edge: (edge.timestamp, edge.alert_id)))
        return sorted(promoted, key=lambda component: (component[0].timestamp, component[0].alert_id))


class GNNLLMHybridReasoner:
    """Lightweight GIDS layer 3 structural scorer with optional Gemini narratives."""

    def __init__(
        self,
        threshold: float = 0.55,
        gemini: Optional["GeminiTextClient"] = None,
        *,
        use_rarity: bool = False,
    ) -> None:
        self.threshold = threshold
        self.gemini = gemini
        self.use_rarity = use_rarity

    def classify(self, cluster: list[GIDSEdge], matches: list[GIDSSubgraphMatch], incident_id: str) -> Optional[GIDSIncident]:
        score = self._structural_score(cluster, matches)
        if score < self.threshold:
            return None
        timestamps = [edge.timestamp for edge in cluster]
        tactics = sorted({edge.tactic for edge in cluster if edge.tactic})
        pattern_names = sorted({match.pattern for match in matches})
        hosts = set().union(*(edge.host_ids for edge in cluster))
        users = {edge.user_id for edge in cluster if edge.user_id}
        narrative = self._narrative(cluster, score, pattern_names)
        return GIDSIncident(
            incident_id=incident_id,
            alert_ids=sorted(edge.alert_id for edge in cluster),
            start_time=min(timestamps).isoformat(),
            end_time=max(timestamps).isoformat(),
            alert_count=len(cluster),
            host_count=len(hosts),
            user_count=len(users),
            max_severity=max(edge.severity for edge in cluster),
            tactics=tactics,
            pattern_matches=pattern_names,
            structural_score=score,
            confidence=score,
            narrative=narrative,
        )

    def _structural_score(self, cluster: list[GIDSEdge], matches: list[GIDSSubgraphMatch]) -> float:
        severities = [edge.severity for edge in cluster]
        timestamps = [edge.timestamp for edge in cluster]
        tactics = {edge.tactic for edge in cluster if edge.tactic}
        hosts = set().union(*(edge.host_ids for edge in cluster))
        duration = max((max(timestamps) - min(timestamps)).total_seconds(), 1)
        severity_score = max(severities) / 10
        high_density = sum(1 for severity in severities if severity >= 8) / len(severities)
        size_score = min(len(cluster) / 8, 1.0)
        host_score = min(len(hosts) / 3, 1.0)
        tactic_score = min(len(tactics) / 4, 1.0)
        temporal_score = max(0.0, 1.0 - min(duration, 6 * 3600) / (6 * 3600))
        pattern_score = min(len(matches) / 2, 1.0)
        rarity_score = sum(1 for edge in cluster if edge.rare_relationship) / len(cluster)
        if self.use_rarity:
            return round(
                min(
                    severity_score * 0.22
                    + high_density * 0.18
                    + size_score * 0.14
                    + host_score * 0.10
                    + tactic_score * 0.10
                    + temporal_score * 0.09
                    + pattern_score * 0.10
                    + rarity_score * 0.07,
                    1.0,
                ),
                6,
            )
        return round(
            min(
                severity_score * 0.25
                + high_density * 0.20
                + size_score * 0.15
                + host_score * 0.10
                + tactic_score * 0.10
                + temporal_score * 0.10
                + pattern_score * 0.10,
                1.0,
            ),
            6,
        )

    def _narrative(self, cluster: list[GIDSEdge], score: float, patterns: list[str]) -> str:
        first = min(cluster, key=lambda edge: (edge.timestamp, edge.alert_id))
        last = max(cluster, key=lambda edge: (edge.timestamp, edge.alert_id))
        pattern_text = ", ".join(patterns) if patterns else "entity/time correlation"
        return (
            f"GIDS classified this subgraph as an incident with score {score:.2f}. "
            f"The activity starts with {first.tactic or 'an alert'} on {first.source_id} "
            f"and ends with {last.tactic or 'an alert'} on {last.target_id}; "
            f"the strongest structural evidence is {pattern_text}."
        )


class GIDSDetector:
    def __init__(
        self,
        sme: Optional[SubgraphMatchingEngine] = None,
        cdc: Optional[CommunityDetectionCorrelation] = None,
        reasoner: Optional[GNNLLMHybridReasoner] = None,
        novelty: Optional[GIDSRelationshipNovelty] = None,
    ) -> None:
        self.sme = sme or SubgraphMatchingEngine()
        self.cdc = cdc or CommunityDetectionCorrelation()
        self.reasoner = reasoner or GNNLLMHybridReasoner()
        self.novelty = novelty

    def run(self, edges: list[GIDSEdge]) -> GIDSRunResult:
        started = perf_counter()
        if self.novelty is not None:
            edges = self.novelty.annotate(edges)
        graph = GIDSGraph()
        for edge in sorted(edges, key=lambda item: (item.timestamp, item.alert_id)):
            graph.ingest(edge)
        pattern_matches = self.sme.match(graph.edges)
        matches_by_alert: dict[str, list[GIDSSubgraphMatch]] = defaultdict(list)
        for match in pattern_matches:
            for alert_id in match.alert_ids:
                matches_by_alert[alert_id].append(match)

        incidents: list[GIDSIncident] = []
        communities = self.cdc.communities(graph.edges)
        if self.novelty is not None:
            rare_components = self.novelty.components(graph.edges)
        elif any(edge.rare_relationship for edge in graph.edges):
            rare_components = _rare_relationship_components(graph.edges)
        else:
            rare_components = []
        candidate_clusters = _deduplicate_clusters(communities + _sme_components(graph.edges, pattern_matches) + rare_components)
        for index, cluster in enumerate(candidate_clusters, start=1):
            cluster_ids = {edge.alert_id for edge in cluster}
            cluster_matches_by_key = {
                (match.pattern, tuple(match.alert_ids)): match
                for alert_id in cluster_ids
                for match in matches_by_alert.get(alert_id, [])
            }
            cluster_matches = sorted(cluster_matches_by_key.values(), key=lambda match: (match.pattern, match.alert_ids))
            incident = self.reasoner.classify(cluster, cluster_matches, f"GIDS-{index:04d}")
            if incident:
                incidents.append(incident)

        return GIDSRunResult(
            candidate_alerts=len(edges),
            pattern_matches=len(pattern_matches),
            communities_considered=len(communities),
            incidents=incidents,
            elapsed_seconds=perf_counter() - started,
            rare_relationships=sum(1 for edge in graph.edges if edge.rare_relationship),
        )


def _run_gids_rare(edges: list[GIDSEdge]) -> tuple[GIDSRunResult, list[GIDSEdge]]:
    novelty = GIDSRelationshipNovelty()
    annotated_edges = novelty.annotate(edges)
    detector = GIDSDetector(reasoner=GNNLLMHybridReasoner(use_rarity=True))
    return detector.run(annotated_edges), annotated_edges


def _sme_components(edges: list[GIDSEdge], matches: list[GIDSSubgraphMatch]) -> list[list[GIDSEdge]]:
    if not matches:
        return []
    edge_by_id = {edge.alert_id: edge for edge in edges}
    parent: dict[str, str] = {}

    def find(alert_id: str) -> str:
        parent.setdefault(alert_id, alert_id)
        while parent[alert_id] != alert_id:
            parent[alert_id] = parent[parent[alert_id]]
            alert_id = parent[alert_id]
        return alert_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for match in matches:
        ids = [alert_id for alert_id in match.alert_ids if alert_id in edge_by_id]
        if not ids:
            continue
        find(ids[0])
        for alert_id in ids[1:]:
            union(ids[0], alert_id)

    components: dict[str, list[GIDSEdge]] = defaultdict(list)
    for alert_id in parent:
        components[find(alert_id)].append(edge_by_id[alert_id])
    return [sorted(component, key=lambda edge: (edge.timestamp, edge.alert_id)) for component in components.values()]


def _rare_relationship_components(
    edges: list[GIDSEdge],
    *,
    max_gap_minutes: int = 180,
    min_alerts: int = 4,
) -> list[list[GIDSEdge]]:
    return _cluster_edges_by_entity_time(
        [edge for edge in edges if edge.rare_relationship],
        max_gap_minutes=max_gap_minutes,
        min_alerts=min_alerts,
    )


def _deduplicate_clusters(clusters: list[list[GIDSEdge]]) -> list[list[GIDSEdge]]:
    unique: list[tuple[frozenset[str], list[GIDSEdge]]] = []
    for cluster in clusters:
        alert_ids = frozenset(edge.alert_id for edge in cluster)
        if not alert_ids:
            continue
        if any(alert_ids <= existing_ids for existing_ids, _ in unique):
            continue
        unique = [(existing_ids, existing) for existing_ids, existing in unique if not existing_ids < alert_ids]
        unique.append((alert_ids, cluster))
    return [cluster for _, cluster in sorted(unique, key=lambda item: (item[1][0].timestamp, item[1][0].alert_id))]


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    use_vertexai: bool
    project: str
    location: str
    api_key: str
    temperature: float
    max_output_tokens: int

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "GeminiSettings":
        load_env_file(env_path)
        return cls(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            use_vertexai=_truthy(os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true")),
            project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            api_key=os.getenv("GEMINI_API_KEY", ""),
            temperature=float(os.getenv("GIDS_LLM_TEMPERATURE", "0")),
            max_output_tokens=int(os.getenv("GIDS_LLM_MAX_OUTPUT_TOKENS", "4096")),
        )

    @property
    def is_configured(self) -> bool:
        if self.use_vertexai:
            return bool(self.project and self.location)
        return bool(self.api_key)


class GeminiTextClient:
    def __init__(self, settings: GeminiSettings) -> None:
        if not settings.is_configured:
            raise RuntimeError("Gemini settings are incomplete. Fill .env before running the Gemini baseline.")
        self.settings = settings

    def generate_text(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install the optional `gemini` dependency to run Gemini calls.") from exc

        if self.settings.use_vertexai:
            client = genai.Client(
                vertexai=True,
                project=self.settings.project,
                location=self.settings.location,
            )
        else:
            client = genai.Client(api_key=self.settings.api_key)

        response = client.models.generate_content(
            model=self.settings.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        return _gemini_response_text(response)


class PlainGeminiIncidentAgent:
    def __init__(self, client: GeminiTextClient, max_alerts: int = 2_000, *, include_severity: bool = True) -> None:
        self.client = client
        self.max_alerts = max_alerts
        self.include_severity = include_severity

    def predict_incidents(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        prompt = self.build_prompt(rows[: self.max_alerts])
        response = self.client.generate_text(prompt)
        payload = _extract_json_payload(response)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        incidents = payload.get("incidents", []) if isinstance(payload, dict) else []
        return incidents if isinstance(incidents, list) else []

    def build_prompt(self, rows: list[dict[str, str]]) -> str:
        sanitized = [_strip_model_fields(row, include_severity=self.include_severity) for row in rows]
        if self.include_severity:
            grouping_guidance = (
                "severity progression, process names, and coherent MITRE tactic sequences. High/Critical alerts are strong evidence, "
                "but require entity/time continuity before grouping. Medium/Low alerts may be supporting context only when they bridge "
                "the same entities in the same time window."
            )
            inference_guidance = "alert content, entities, time, severity, and process fields"
            high_signal_guidance = "Do not return an empty incident list when there are multi-alert High/Critical chains with shared entities.\n"
        else:
            grouping_guidance = (
                "process names, coherent MITRE tactic sequences, time proximity, and shared source/target hosts or users. "
                "Require entity/time continuity before grouping."
            )
            inference_guidance = "alert content, entities, time, tactic sequence, and process fields"
            high_signal_guidance = "Do not return an empty incident list when there are multi-alert tactic chains with shared entities.\n"
        return (
            "You are an L3 incident responder. You receive raw CrowdStrike Falcon alert rows.\n"
            "First perform a private analysis pass over the alert rows before producing JSON.\n"
            "During that analysis, group alerts into candidate incidents using time proximity, shared source/target hosts, shared users, "
            f"{grouping_guidance}\n"
            f"Do not assume labels exist. Infer incident clusters only from {inference_guidance}.\n"
            f"{high_signal_guidance}"
            "Return strict JSON with this shape: "
            '{"incidents":[{"cluster_id":"LLM-1","alert_ids":["..."],"confidence":0.0,"rationale":"..."}]}.\n'
            "Use only alert_id values from the input. Do not include explanatory prose outside JSON.\n\n"
            f"ALERT_ROWS_JSON:\n{json.dumps(sanitized, separators=(',', ':'))}"
        )


class IsolatedGeminiAlertClassifier:
    """LLM baseline that classifies each alert independently, without incident context."""

    def __init__(
        self,
        client: GeminiTextClient,
        *,
        batch_size: int = 100,
        max_alerts: int = 2_000,
        include_severity: bool = True,
    ) -> None:
        self.client = client
        self.batch_size = max(1, batch_size)
        self.max_alerts = max_alerts
        self.include_severity = include_severity

    def predict_alerts(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        capped = rows[: self.max_alerts]
        for offset in range(0, len(capped), self.batch_size):
            prompt = self.build_prompt(capped[offset : offset + self.batch_size])
            response = self.client.generate_text(prompt)
            payload = _extract_json_payload(response)
            if isinstance(payload, list):
                raw_alerts = payload
            elif isinstance(payload, dict):
                raw_alerts = _first_list(payload, "alerts", "predictions", "results")
            else:
                raw_alerts = []
            if isinstance(raw_alerts, list):
                predictions.extend(item for item in raw_alerts if isinstance(item, dict))
        return predictions

    def build_prompt(self, rows: list[dict[str, str]]) -> str:
        sanitized = [_strip_model_fields(row, include_severity=self.include_severity) for row in rows]
        evidence_fields = (
            "alert content, entities, time, severity, tactic, technique, and process fields"
            if self.include_severity
            else "alert content, entities, time, tactic, technique, and process fields"
        )
        return (
            "You are an L3 security analyst. Classify each CrowdStrike Falcon alert independently.\n"
            "Do not group alerts. Do not infer an incident from neighboring rows, repeated users, repeated hosts, or global ordering.\n"
            f"For each isolated alert, decide whether the single alert is enough to mark it as an incident candidate using only {evidence_fields}.\n"
            "Return strict JSON only with this shape: "
            '{"alerts":[{"alert_id":"...","incident_candidate":true,"confidence":0.0}]}.\n'
            "Return exactly one object per input alert_id. Use only alert_id values from the input. No prose outside JSON.\n\n"
            f"ALERT_ROWS_JSON:\n{json.dumps(sanitized, separators=(',', ':'))}"
        )


class GIDSGeminiRationalizerAgent:
    def __init__(self, client: GeminiTextClient, *, include_severity: bool = True, batch_size: int = 10) -> None:
        self.client = client
        self.include_severity = include_severity
        self.batch_size = max(1, batch_size)

    def rationalize(self, incidents: list[GIDSIncident], edges: list[GIDSEdge]) -> list[LLMRationalizerDecision]:
        if not incidents:
            return []
        decisions: list[LLMRationalizerDecision] = []
        for offset in range(0, len(incidents), self.batch_size):
            prompt = self.build_prompt(incidents[offset : offset + self.batch_size], edges)
            response = self.client.generate_text(prompt)
            payload = _extract_json_payload(response)
            if isinstance(payload, list):
                raw_decisions = payload
            elif isinstance(payload, dict):
                raw_decisions = _first_list(payload, "candidates", "incidents", "results", "analyses")
            else:
                raw_decisions = []
            if isinstance(raw_decisions, list):
                decisions.extend(_decision_from_payload(item) for item in raw_decisions if isinstance(item, dict))
        return decisions

    def build_prompt(self, incidents: list[GIDSIncident], edges: list[GIDSEdge]) -> str:
        edge_by_id = {edge.alert_id: edge for edge in edges}
        candidates = []
        for incident in incidents:
            candidate_edges = [edge_by_id[alert_id] for alert_id in incident.alert_ids if alert_id in edge_by_id]
            candidate = {
                "candidate_id": incident.incident_id,
                "GNN_SCORE": incident.structural_score,
                "node_count": incident.host_count + incident.user_count,
                "alert_count": incident.alert_count,
                "host_count": incident.host_count,
                "user_count": incident.user_count,
                "start_time": incident.start_time,
                "end_time": incident.end_time,
                "pattern_matches": incident.pattern_matches,
                "tactics": incident.tactics,
                "alerts": [
                    {
                        "alert_id": edge.alert_id,
                        "timestamp": edge.timestamp.isoformat(),
                        "source_node": edge.source_id,
                        "target_node": edge.target_id,
                        "user": edge.user_id,
                        "tactic": edge.tactic,
                        "technique": edge.technique,
                        "process": edge.process_name,
                    }
                    for edge in candidate_edges
                ],
            }
            if self.include_severity:
                candidate["max_severity"] = incident.max_severity
                for payload, edge in zip(candidate["alerts"], candidate_edges):
                    payload["severity"] = edge.severity
            candidates.append(candidate)
        data_sentence = (
            "DATA: JSON candidate incident subgraphs generated by GIDS. Scoring labels were removed before this prompt.\n"
            if self.include_severity
            else "DATA: JSON candidate incident subgraphs generated by GIDS. Scoring labels were removed before this prompt.\n"
        )
        return (
            "SYSTEM: You are an L3 Incident Responder.\n"
            f"{data_sentence}"
            "TASK: For every candidate, analyze the relationship between its nodes and alerts.\n"
            "1. Determine if this candidate is a true_positive or false_positive.\n"
            "2. Write a concise narrative of the attacker journey or benign explanation.\n"
            "3. List the first three remediation steps if true_positive; otherwise list validation steps.\n"
            "Use GNN_SCORE as supporting evidence, not as ground truth. Preserve every candidate_id.\n"
            f"You must return exactly {len(candidates)} candidate decision objects, one per input candidate_id.\n"
            "Return strict JSON only with this shape: "
            '{"candidates":[{"candidate_id":"GIDS-0001","verdict":"true_positive","confidence":0.0,'
            '"rationale":"...","remediation_steps":["...","...","..."]}]}.\n\n'
            f"JSON_SUBGRAPH_METADATA:\n{json.dumps(candidates, separators=(',', ':'))}"
        )


class GIDSLocalVerdictAgent:
    """No-label local final verdict agent for GIDS candidate incidents."""

    def __init__(
        self,
        *,
        use_severity: bool = True,
        use_rarity: bool = False,
        analyst_threshold: float = 0.68,
    ) -> None:
        self.use_severity = use_severity
        self.use_rarity = use_rarity
        self.analyst_threshold = analyst_threshold

    def validate(self, incidents: list[GIDSIncident], edges: list[GIDSEdge]) -> list[LLMRationalizerDecision]:
        edge_by_id = {edge.alert_id: edge for edge in edges}
        return [
            self._validate_one(incident, [edge_by_id[alert_id] for alert_id in incident.alert_ids if alert_id in edge_by_id])
            for incident in incidents
        ]

    def _validate_one(self, incident: GIDSIncident, edges: list[GIDSEdge]) -> LLMRationalizerDecision:
        alert_count = len(edges)
        tactics = {edge.tactic for edge in edges if edge.tactic}
        high_density = (
            sum(1 for edge in edges if edge.severity >= 8) / alert_count
            if self.use_severity and alert_count
            else 0.0
        )
        rare_density = (
            sum(1 for edge in edges if edge.rare_relationship) / alert_count
            if self.use_rarity and alert_count
            else 0.0
        )
        order_score = _candidate_order_score(edges)
        pattern_count = len(incident.pattern_matches)
        tactic_depth = min(len(tactics) / 6, 1.0)
        volume_score = min(alert_count / 10, 1.0)
        pattern_score = min(pattern_count / 4, 1.0)
        severity_signal = high_density if self.use_severity else 0.0
        rarity_signal = rare_density if self.use_rarity else 0.0
        evidence_confidence = _bounded_float(
            incident.structural_score * 0.26
            + volume_score * 0.16
            + tactic_depth * 0.16
            + pattern_score * 0.16
            + order_score * 0.16
            + severity_signal * 0.05
            + rarity_signal * 0.05
        )
        selected = evidence_confidence >= self.analyst_threshold
        verdict = "true_positive" if selected else "false_positive"
        confidence = evidence_confidence if selected else _bounded_float(1.0 - evidence_confidence)
        rationale = (
            "L3 analyst-style reasoning over candidate evidence: "
            f"alerts={alert_count}, tactics={len(tactics)}, patterns={pattern_count}, "
            f"order_score={order_score:.2f}, structural_score={incident.structural_score:.2f}, "
            f"high_density={high_density:.2f}, rare_density={rare_density:.2f}, "
            f"confidence={evidence_confidence:.2f}"
        )
        if selected:
            actions = [
                "scope shared hosts and users in the candidate window",
                "collect process lineage for the ordered tactic chain",
                "contain affected hosts before credential reuse",
            ]
        else:
            actions = [
                "validate whether alerts share a single root cause",
                "check for benign administrative burst activity",
                "require additional ordered attack evidence before escalation",
            ]
        return LLMRationalizerDecision(
            candidate_id=incident.incident_id,
            verdict=verdict,
            confidence=confidence,
            selected=selected,
            rationale=rationale,
            remediation_steps=actions,
        )


def evaluate_gids_vs_plain_gemini(
    path: str | Path,
    *,
    run_gemini: bool = False,
    run_isolated_gemini: bool = False,
    isolated_gemini_max_alerts: int = 2_000,
    isolated_gemini_batch_size: int = 100,
    env_path: str | Path = ".env",
    prompt_dir: str | Path | None = None,
    hide_severity: bool = False,
    progress: ProgressCallback = None,
) -> GIDSComparisonReport:
    input_path = Path(path)
    _emit_progress(progress, f"loading alerts from {input_path}")
    rows = list(_iter_csv_rows(input_path))
    edges = [edge_from_falcon_row(row, hide_severity=hide_severity) for row in rows]
    labels = {edge.alert_id: _truthy(row.get("is_incident", "")) for edge, row in zip(edges, rows)}
    ground_truth = _incident_alert_ids(edges, rows)
    ground_truth_ids = {alert_id for alert_id, is_incident in labels.items() if is_incident}

    _emit_progress(progress, f"running GIDS structural detector on {len(edges)} alerts")
    gids_run = GIDSDetector().run(edges)
    _emit_progress(progress, f"GIDS produced {len(gids_run.incidents)} candidate incidents")
    if prompt_dir is not None:
        _emit_progress(progress, f"writing prompt artifacts to {prompt_dir}")
        _write_prompt_artifacts(Path(prompt_dir), rows=rows, edges=edges, gids_run=gids_run, include_severity=not hide_severity)
    gids_eval = _evaluate_detector(
        detector="GIDS",
        status="ok",
        clusters={incident.incident_id: set(incident.alert_ids) for incident in gids_run.incidents},
        labels=labels,
        ground_truth=ground_truth,
        notes=["GIDS uses directional graph edges, SME patterns, CDC entity/time communities, and a structural GHR score."],
    )
    _emit_progress(progress, "running GIDS + local verdict agent evaluation")
    gids_local_eval, local_verdict_decisions = _gids_local_verdict_evaluation(
        gids_run,
        edges,
        labels,
        ground_truth,
        use_severity=not hide_severity,
    )
    _emit_progress(progress, f"GIDS + local verdict agent status: {gids_local_eval.status}")
    _emit_progress(progress, "running gids_rare relationship novelty evaluation")
    gids_rare_run, rare_edges = _run_gids_rare(edges)
    gids_rare_eval = _evaluate_detector(
        detector="gids_rare",
        status="ok",
        clusters={incident.incident_id: set(incident.alert_ids) for incident in gids_rare_run.incidents},
        labels=labels,
        ground_truth=ground_truth,
        notes=[
            (
                "GIDS variant that adds Count-Min-Sketch relationship rarity as a candidate source and a small structural-score feature."
            )
        ],
    )
    _emit_progress(
        progress,
        f"gids_rare produced {len(gids_rare_run.incidents)} candidate incidents from {gids_rare_run.rare_relationships} rare edges",
    )
    _emit_progress(progress, "running gids_rare_with_agent evaluation")
    gids_rare_agent_eval, _ = _gids_local_verdict_evaluation(
        gids_rare_run,
        rare_edges,
        labels,
        ground_truth,
        use_severity=not hide_severity,
        use_rarity=True,
        detector_name="gids_rare_with_agent",
    )
    _emit_progress(progress, f"gids_rare_with_agent status: {gids_rare_agent_eval.status}")
    _emit_progress(progress, "running GIDS + Gemini rationalizer evaluation")
    gids_llm_eval, rationalizer_decisions = _gids_gemini_rationalizer_evaluation(
        gids_run,
        edges,
        labels,
        ground_truth,
        run_gemini=run_gemini,
        env_path=env_path,
        include_severity=not hide_severity,
    )
    _emit_progress(progress, f"GIDS + Gemini rationalizer status: {gids_llm_eval.status}")
    _emit_progress(progress, "running plain Gemini raw-alert evaluation")
    gemini_eval = _plain_gemini_evaluation(
        rows,
        labels,
        ground_truth,
        run_gemini=run_gemini,
        env_path=env_path,
        include_severity=not hide_severity,
    )
    _emit_progress(progress, f"plain Gemini status: {gemini_eval.status}")
    _emit_progress(progress, "running isolated-alert Gemini evaluation")
    isolated_gemini_eval = _isolated_gemini_evaluation(
        rows,
        labels,
        ground_truth,
        run_gemini=run_gemini or run_isolated_gemini,
        env_path=env_path,
        include_severity=not hide_severity,
        max_alerts=isolated_gemini_max_alerts,
        batch_size=isolated_gemini_batch_size,
    )
    _emit_progress(progress, f"isolated-alert Gemini status: {isolated_gemini_eval.status}")
    _emit_progress(progress, "running local analyst severity-chain review")
    local_analyst_eval = _local_analyst_evaluation(edges, labels, ground_truth, hide_severity=hide_severity)
    _emit_progress(progress, "finished detector comparison")

    return GIDSComparisonReport(
        input_path=str(input_path),
        candidate_alerts=len(rows),
        ground_truth_alerts=len(ground_truth_ids),
        ground_truth_incidents=len(ground_truth),
        gids_run=gids_run,
        detectors=[
            gids_eval,
            gids_local_eval,
            gids_rare_eval,
            gids_rare_agent_eval,
            gids_llm_eval,
            gemini_eval,
            isolated_gemini_eval,
            local_analyst_eval,
        ],
        rationalizer_decisions=rationalizer_decisions,
        local_verdict_decisions=local_verdict_decisions,
        notes=[
            "`is_incident` and `incident_id` are used only for scoring; they are stripped before GIDS and Gemini prompts.",
            (
                f"`severity` was hidden from prompts and replaced with neutral score {HIDDEN_SEVERITY_SCORE} before graph scoring."
                if hide_severity
                else "`severity` was available as a model feature."
            ),
            "Gemini modes run only when `--run-gemini` is used and valid `.env` settings are provided.",
            "The local analyst row is a transparent no-label heuristic over severity plus entity/time continuity.",
            "The GIDS + local verdict agent row applies a no-label final verdict to each GIDS suspected incident before scoring.",
            "`gids_rare` adds relationship rarity only as supporting graph evidence; ground-truth labels are still scoring-only.",
            "The isolated-alert Gemini row classifies each alert independently and scores selected alert IDs as one-alert clusters.",
        ],
    )


def edge_from_falcon_row(row: dict[str, str], *, hide_severity: bool = False) -> GIDSEdge:
    raw = _strip_model_fields(row, include_severity=not hide_severity)
    return GIDSEdge(
        alert_id=_first_text(row, "alert_id", default=""),
        source_id=_first_text(row, "source_node", default="unknown-source"),
        target_id=_first_text(row, "target_node", default="unknown-target"),
        user_id=_first_text(row, "user", default=""),
        process_name=_first_text(row, "process", default=""),
        tactic=_first_text(row, "tactic", default=""),
        technique=_first_text(row, "technique", default=""),
        severity=HIDDEN_SEVERITY_SCORE if hide_severity else _severity_score(_first_text(row, "severity", default="Low")),
        timestamp=_parse_timestamp(_first_text(row, "timestamp", default="")),
        raw=raw,
    )


def _gids_local_verdict_evaluation(
    gids_run: GIDSRunResult,
    edges: list[GIDSEdge],
    labels: dict[str, bool],
    ground_truth: dict[str, set[str]],
    *,
    use_severity: bool = True,
    use_rarity: bool = False,
    detector_name: str = "GIDS + local verdict agent",
) -> tuple[DetectorEvaluation, list[LLMRationalizerDecision]]:
    decisions = GIDSLocalVerdictAgent(use_severity=use_severity, use_rarity=use_rarity).validate(gids_run.incidents, edges)
    incidents_by_id = {incident.incident_id: incident for incident in gids_run.incidents}
    selected_clusters = {
        decision.candidate_id: set(incidents_by_id[decision.candidate_id].alert_ids)
        for decision in decisions
        if decision.selected and decision.candidate_id in incidents_by_id
    }
    return (
        _evaluate_detector(
            detector=detector_name,
            status="ok",
            clusters=selected_clusters,
            labels=labels,
            ground_truth=ground_truth,
            notes=[
                (
                    "Local final verdict uses L3 analyst-style reasoning over candidate evidence without an explicit rule checklist."
                    if use_severity and not use_rarity
                    else "Local final verdict uses L3 analyst-style reasoning over candidate evidence, including rarity where available."
                    if use_rarity
                    else "Local final verdict uses L3 analyst-style reasoning over candidate evidence; severity was hidden."
                )
            ],
        ),
        decisions,
    )


def _gids_gemini_rationalizer_evaluation(
    gids_run: GIDSRunResult,
    edges: list[GIDSEdge],
    labels: dict[str, bool],
    ground_truth: dict[str, set[str]],
    *,
    run_gemini: bool,
    env_path: str | Path,
    include_severity: bool = True,
) -> tuple[DetectorEvaluation, list[LLMRationalizerDecision]]:
    if not run_gemini:
        return (
            _empty_detector(
                "GIDS + Gemini rationalizer",
                "not_run",
                ["Run with `--run-gemini` to execute the Part-B LLM decoder over GIDS candidates."],
            ),
            [],
        )

    settings = GeminiSettings.from_env(env_path)
    if not settings.is_configured:
        return (
            _empty_detector(
                "GIDS + Gemini rationalizer",
                "not_configured",
                ["Gemini settings are incomplete. Fill `.env` with GCP project/location or an API key."],
            ),
            [],
        )

    try:
        decisions = GIDSGeminiRationalizerAgent(
            GeminiTextClient(settings),
            include_severity=include_severity,
        ).rationalize(gids_run.incidents, edges)
    except Exception as exc:
        return (
            _empty_detector(
                "GIDS + Gemini rationalizer",
                "runtime_error",
                [f"Gemini rationalizer did not complete: {exc.__class__.__name__}: {exc}"],
            ),
            [],
        )
    if not decisions:
        return (
            _empty_detector(
                "GIDS + Gemini rationalizer",
                "empty_response",
                ["Gemini rationalizer returned no parseable candidate decisions."],
            ),
            [],
        )

    incidents_by_id = {incident.incident_id: incident for incident in gids_run.incidents}
    selected_clusters = {
        decision.candidate_id: set(incidents_by_id[decision.candidate_id].alert_ids)
        for decision in decisions
        if decision.selected and decision.candidate_id in incidents_by_id
    }
    return (
        _evaluate_detector(
            detector="GIDS + Gemini rationalizer",
            status="ok",
            clusters=selected_clusters,
            labels=labels,
            ground_truth=ground_truth,
            notes=[
                (
                    "Gemini received GIDS candidate subgraphs and GNN scores, then filtered candidates as true/false positives."
                    if include_severity
                    else "Gemini received GIDS candidate subgraphs and GNN scores with severity omitted, then filtered candidates as true/false positives."
                )
            ],
        ),
        decisions,
    )


def _plain_gemini_evaluation(
    rows: list[dict[str, str]],
    labels: dict[str, bool],
    ground_truth: dict[str, set[str]],
    *,
    run_gemini: bool,
    env_path: str | Path,
    include_severity: bool = True,
) -> DetectorEvaluation:
    if not run_gemini:
        return _empty_detector(
            "Plain Gemini raw-alert agent",
            "not_run",
            ["Run with `--run-gemini` after filling `.env` to execute the raw-alert Gemini baseline."],
        )

    settings = GeminiSettings.from_env(env_path)
    if not settings.is_configured:
        return _empty_detector(
            "Plain Gemini raw-alert agent",
            "not_configured",
            ["Gemini settings are incomplete. Fill `.env` with GCP project/location or an API key."],
        )

    try:
        agent = PlainGeminiIncidentAgent(GeminiTextClient(settings), include_severity=include_severity)
        incidents = agent.predict_incidents(rows)
    except Exception as exc:
        return _empty_detector(
            "Plain Gemini raw-alert agent",
            "runtime_error",
            [f"Gemini baseline did not complete: {exc.__class__.__name__}: {exc}"],
        )
    clusters = {
        str(item.get("cluster_id") or item.get("incident_id") or f"LLM-{idx:04d}"): {
            str(alert_id)
            for alert_id in item.get("alert_ids", [])
            if isinstance(alert_id, str)
        }
        for idx, item in enumerate(incidents, start=1)
        if isinstance(item, dict)
    }
    return _evaluate_detector(
        detector="Plain Gemini raw-alert agent",
        status="ok",
        clusters=clusters,
        labels=labels,
        ground_truth=ground_truth,
        notes=[
            (
                "Raw Falcon rows were sent with scoring labels stripped from the prompt."
                if include_severity
                else "Raw Falcon rows were sent with scoring labels and severity stripped from the prompt."
            )
        ],
    )


def _isolated_gemini_evaluation(
    rows: list[dict[str, str]],
    labels: dict[str, bool],
    ground_truth: dict[str, set[str]],
    *,
    run_gemini: bool,
    env_path: str | Path,
    include_severity: bool = True,
    max_alerts: int = 2_000,
    batch_size: int = 100,
) -> DetectorEvaluation:
    if not run_gemini:
        return _empty_detector(
            "Gemini isolated-alert classifier",
            "not_run",
            ["Run with `--run-isolated-gemini` to classify each raw alert independently."],
        )

    settings = GeminiSettings.from_env(env_path)
    if not settings.is_configured:
        return _empty_detector(
            "Gemini isolated-alert classifier",
            "not_configured",
            ["Gemini settings are incomplete. Fill `.env` with GCP project/location or an API key."],
        )

    try:
        agent = IsolatedGeminiAlertClassifier(
            GeminiTextClient(settings),
            include_severity=include_severity,
            max_alerts=max_alerts,
            batch_size=batch_size,
        )
        predictions = agent.predict_alerts(rows)
    except Exception as exc:
        return _empty_detector(
            "Gemini isolated-alert classifier",
            "runtime_error",
            [f"Gemini isolated-alert classifier did not complete: {exc.__class__.__name__}: {exc}"],
        )

    scored_rows = rows[:max_alerts]
    scored_ids = {str(row.get("alert_id", "")).strip() for row in scored_rows}
    scored_labels = {alert_id: value for alert_id, value in labels.items() if alert_id in scored_ids}
    scored_ground_truth = {
        incident_id: alert_ids & scored_ids
        for incident_id, alert_ids in ground_truth.items()
        if alert_ids & scored_ids
    }
    clusters = {
        f"ISO-{idx:04d}": {alert_id}
        for idx, alert_id in enumerate(_selected_isolated_alert_ids(predictions), start=1)
        if alert_id in scored_labels
    }
    return _evaluate_detector(
        detector="Gemini isolated-alert classifier",
        status="ok_sampled" if len(scored_rows) < len(rows) else "ok",
        clusters=clusters,
        labels=scored_labels,
        ground_truth=scored_ground_truth,
        notes=[
            (
                "Gemini classified each alert independently with scoring labels stripped."
                if include_severity
                else "Gemini classified each alert independently with scoring labels and severity stripped."
            ),
            f"Evaluation capped at {min(max_alerts, len(rows))} of {len(rows)} input alerts with batch_size={batch_size}.",
        ],
    )


def _local_analyst_evaluation(
    edges: list[GIDSEdge],
    labels: dict[str, bool],
    ground_truth: dict[str, set[str]],
    *,
    hide_severity: bool = False,
) -> DetectorEvaluation:
    if hide_severity:
        return _empty_detector(
            "Local analyst severity-chain review",
            "severity_hidden",
            ["Disabled in this ablation because this baseline's first-pass rule is explicitly severity-based."],
        )
    high_signal_edges = [edge for edge in edges if edge.severity >= 8]
    clusters = {
        f"LOCAL-{idx:04d}": {edge.alert_id for edge in cluster}
        for idx, cluster in enumerate(_cluster_edges_by_entity_time(high_signal_edges, max_gap_minutes=120, min_alerts=2), start=1)
    }
    return _evaluate_detector(
        detector="Local analyst severity-chain review",
        status="ok",
        clusters=clusters,
        labels=labels,
        ground_truth=ground_truth,
        notes=[
            "Local analysis selected High/Critical alerts and clustered them by shared user/host within a two-hour window.",
            "This is included because the synthetic Falcon file has unusually clean severity separation.",
        ],
    )


def _write_prompt_artifacts(
    prompt_dir: Path,
    *,
    rows: list[dict[str, str]],
    edges: list[GIDSEdge],
    gids_run: GIDSRunResult,
    include_severity: bool = True,
) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    plain_prompt = PlainGeminiIncidentAgent(
        _PromptOnlyGeminiClient(),
        include_severity=include_severity,
    ).build_prompt(rows)
    rationalizer_prompt = GIDSGeminiRationalizerAgent(
        _PromptOnlyGeminiClient(),
        include_severity=include_severity,
    ).build_prompt(gids_run.incidents, edges)
    isolated_prompt = IsolatedGeminiAlertClassifier(
        _PromptOnlyGeminiClient(),
        include_severity=include_severity,
    ).build_prompt(rows[:100])
    (prompt_dir / "plain_gemini_raw_alert_agent_prompt.txt").write_text(plain_prompt, encoding="utf-8")
    (prompt_dir / "gids_gemini_rationalizer_prompt.txt").write_text(rationalizer_prompt, encoding="utf-8")
    (prompt_dir / "gemini_isolated_alert_classifier_prompt.txt").write_text(isolated_prompt, encoding="utf-8")
    (prompt_dir / "gids_local_verdict_agent_prompt.md").write_text(
        _gids_local_verdict_prompt(include_severity=include_severity),
        encoding="utf-8",
    )
    (prompt_dir / "local_analyst_self_analysis_prompt.md").write_text(
        _local_analyst_prompt(include_severity=include_severity),
        encoding="utf-8",
    )


class _PromptOnlyGeminiClient:
    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("prompt-only client cannot call Gemini")


def _local_analyst_prompt(*, include_severity: bool = True) -> str:
    if not include_severity:
        return """# Local Analyst Self-Analysis Prompt

SYSTEM: You are Codex evaluating the benchmark locally without ground-truth leakage.

DATA: Sanitized Falcon alert rows. The scoring fields `is_incident` and `incident_id` are unavailable.

TASK:
1. Do not run this baseline as a detector, because its required first-pass field is unavailable.
2. Report the local analyst severity-chain row as disabled for this ablation.
3. Score only the remaining methods after prediction by comparing candidate alert IDs with held-out ground-truth labels.
"""
    return """# Local Analyst Self-Analysis Prompt

SYSTEM: You are Codex evaluating the benchmark locally without ground-truth leakage.

DATA: Sanitized Falcon alert rows. The scoring fields `is_incident` and `incident_id` are unavailable.

TASK:
1. Treat High/Critical alerts as high-signal candidate evidence.
2. Link candidate alerts when they share a host or user within a two-hour window.
3. Promote linked components with at least two alerts into incident candidates.
4. Score only after prediction by comparing candidate alert IDs with held-out ground-truth labels.

RATIONALE:
This is a transparent local analyst baseline, not a production detector. It is included because the synthetic
Falcon benchmark has unusually clean severity separation: all incident alerts are High/Critical and all background
alerts are Medium or lower.
"""


def _gids_local_verdict_prompt(*, include_severity: bool = True) -> str:
    if not include_severity:
        return """# GIDS Local Verdict Agent Prompt

SYSTEM: You are an L3 security analyst. You receive only GIDS suspected incident candidates.

DATA AVAILABLE PER CANDIDATE:
- alert_count, host/user counts, structural GIDS score
- ordered sanitized Falcon alerts without scoring labels
- tactic sequence, pattern matches, and process names

TASK:
Reason about each candidate as an L3 analyst and decide whether it is a true incident or a false positive.
Use the candidate evidence, the entity relationships, timing, process names, tactic flow, pattern matches, and graph score.
Do not use ground-truth labels.

OUTPUT:
Keep candidates you judge to be true incidents; scoring happens later against held-out labels.
"""
    return """# GIDS Local Verdict Agent Prompt

SYSTEM: You are an L3 security analyst. You receive only GIDS suspected incident candidates.

DATA AVAILABLE PER CANDIDATE:
- alert_count, host/user counts, structural GIDS score
- ordered sanitized Falcon alerts without `is_incident` or `incident_id`
- tactic sequence, pattern matches, severity mix, and process names

TASK:
Reason about each candidate as an L3 analyst and decide whether it is a true incident or a false positive.
Use the candidate evidence, the entity relationships, timing, process names, tactic flow, pattern matches, severity mix, and graph score.
Do not use ground-truth labels.

OUTPUT:
Keep candidates you judge to be true incidents; scoring happens later against held-out labels.
"""


def _evaluate_detector(
    *,
    detector: str,
    status: str,
    clusters: dict[str, set[str]],
    labels: dict[str, bool],
    ground_truth: dict[str, set[str]],
    notes: list[str],
) -> DetectorEvaluation:
    selected = set().union(*clusters.values(), set())
    ground_truth_ids = {alert_id for alert_id, is_incident in labels.items() if is_incident}
    true_positives = len(selected & ground_truth_ids)
    false_positives = len(selected - ground_truth_ids)
    false_negatives = len(ground_truth_ids - selected)
    precision = true_positives / len(selected) if selected else 0.0
    recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    matches = _match_clusters(ground_truth, clusters)
    return DetectorEvaluation(
        detector=detector,
        status=status,
        selected_alerts=len(selected),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        clusters_reported=len(clusters),
        known_overlap_clusters=sum(1 for cluster_ids in clusters.values() if any(cluster_ids & gt for gt in ground_truth.values())),
        candidate_new_clusters=sum(1 for cluster_ids in clusters.values() if not any(cluster_ids & gt for gt in ground_truth.values())),
        incident_recall_any=_mean(1.0 if match["any_detected"] else 0.0 for match in matches),
        incident_recall_all=_mean(1.0 if match["all_detected"] else 0.0 for match in matches),
        mean_incident_alert_recall=_mean(match["alert_recall"] for match in matches),
        notes=notes,
    )


def _empty_detector(detector: str, status: str, notes: list[str]) -> DetectorEvaluation:
    return DetectorEvaluation(
        detector=detector,
        status=status,
        selected_alerts=0,
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        clusters_reported=0,
        known_overlap_clusters=0,
        candidate_new_clusters=0,
        incident_recall_any=0.0,
        incident_recall_all=0.0,
        mean_incident_alert_recall=0.0,
        notes=notes,
    )


def _match_clusters(ground_truth: dict[str, set[str]], clusters: dict[str, set[str]]) -> list[dict[str, Any]]:
    matches = []
    recovered_all = set().union(*clusters.values(), set())
    for incident, alert_ids in ground_truth.items():
        recovered = alert_ids & recovered_all
        matches.append(
            {
                "incident": incident,
                "alert_recall": len(recovered) / len(alert_ids) if alert_ids else 0.0,
                "any_detected": bool(recovered),
                "all_detected": bool(alert_ids) and alert_ids <= recovered_all,
            }
        )
    return matches


def _candidate_order_score(edges: list[GIDSEdge]) -> float:
    sequence = _candidate_tactic_sequence(edges)
    if not sequence:
        return 0.0
    return max(_ordered_tactic_score(sequence, chain) for chain in LOCAL_ATTACK_CHAINS)


def _candidate_tactic_sequence(edges: list[GIDSEdge]) -> list[str]:
    sequence: list[str] = []
    for edge in sorted(edges, key=lambda item: (item.timestamp, item.alert_id)):
        if edge.tactic and (not sequence or sequence[-1] != edge.tactic):
            sequence.append(edge.tactic)
    return sequence


def _ordered_tactic_score(sequence: list[str], expected: tuple[str, ...]) -> float:
    previous = [0] * (len(expected) + 1)
    for observed in sequence:
        current = previous[:]
        for idx, tactic in enumerate(expected, start=1):
            if observed == tactic:
                current[idx] = max(current[idx], previous[idx - 1] + 1)
            else:
                current[idx] = max(current[idx], current[idx - 1], previous[idx])
        previous = current
    return previous[-1] / len(expected) if expected else 0.0


def _decision_from_payload(item: dict[str, Any]) -> LLMRationalizerDecision:
    verdict = str(item.get("verdict", "")).strip().lower() or "unknown"
    confidence = _bounded_float(item.get("confidence", 0.0))
    remediation = item.get("remediation_steps", [])
    if not isinstance(remediation, list):
        remediation = []
    selected = verdict in {"true_positive", "true positive", "tp", "incident"} and confidence >= 0.50
    return LLMRationalizerDecision(
        candidate_id=str(item.get("candidate_id", "")).strip(),
        verdict=verdict,
        confidence=confidence,
        selected=selected,
        rationale=str(item.get("rationale", "")).strip(),
        remediation_steps=[str(step).strip() for step in remediation if str(step).strip()][:3],
    )


def _selected_isolated_alert_ids(predictions: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for item in predictions:
        alert_id = str(item.get("alert_id", "")).strip()
        if not alert_id or alert_id in seen:
            continue
        raw_verdict = item.get("incident_candidate", item.get("is_incident", item.get("selected", False)))
        confidence = _bounded_float(item.get("confidence", 1.0 if _payload_truthy(raw_verdict) else 0.0))
        if _payload_truthy(raw_verdict) and confidence >= 0.50:
            selected.append(alert_id)
            seen.add(alert_id)
    return selected


def _payload_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "incident", "incident_candidate", "true_positive"}


def _cluster_edges_by_entity_time(edges: list[GIDSEdge], *, max_gap_minutes: int, min_alerts: int) -> list[list[GIDSEdge]]:
    if not edges:
        return []
    parent = {edge.alert_id: edge.alert_id for edge in edges}

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

    by_entity: dict[str, list[GIDSEdge]] = defaultdict(list)
    for edge in edges:
        for entity in edge.correlation_entities:
            by_entity[entity].append(edge)

    max_gap_seconds = max_gap_minutes * 60
    for entity_edges in by_entity.values():
        ordered = sorted(entity_edges, key=lambda edge: (edge.timestamp, edge.alert_id))
        for idx, edge in enumerate(ordered):
            previous_idx = idx - 1
            while previous_idx >= 0:
                previous = ordered[previous_idx]
                gap = (edge.timestamp - previous.timestamp).total_seconds()
                if gap > max_gap_seconds:
                    break
                union(edge.alert_id, previous.alert_id)
                previous_idx -= 1

    clusters: dict[str, list[GIDSEdge]] = defaultdict(list)
    for edge in edges:
        clusters[find(edge.alert_id)].append(edge)
    return sorted(
        (sorted(cluster, key=lambda edge: (edge.timestamp, edge.alert_id)) for cluster in clusters.values() if len(cluster) >= min_alerts),
        key=lambda cluster: (cluster[0].timestamp, cluster[0].alert_id),
    )


def _incident_alert_ids(edges: list[GIDSEdge], rows: list[dict[str, str]]) -> dict[str, set[str]]:
    incident_alert_ids: dict[str, set[str]] = defaultdict(set)
    for edge, row in zip(edges, rows):
        incident_id = _first_text(row, "incident_id", default="")
        if incident_id:
            incident_alert_ids[incident_id].add(edge.alert_id)
    return dict(incident_alert_ids)


def _edge_weight(severity: int, gap_seconds: float) -> float:
    return severity / math.log(gap_seconds + 2)


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key): "" if value is None else str(value) for key, value in row.items() if key is not None}


def _strip_label_fields(row: dict[str, str]) -> dict[str, str]:
    return _strip_model_fields(row, include_severity=True)


def _strip_model_fields(row: dict[str, str], *, include_severity: bool) -> dict[str, str]:
    stripped_fields = set(LABEL_FIELDS)
    if not include_severity:
        stripped_fields.add(SEVERITY_FIELD)
    return {key: value for key, value in row.items() if key not in stripped_fields}


def _first_text(row: dict[str, str], *names: str, default: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return default


def _lower_token(value: str) -> str:
    return str(value or "").strip().lower()


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


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(parsed, 1.0))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _first_list(payload: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _gemini_response_text(response: Any) -> str:
    text = str(getattr(response, "text", "") or "")
    if text:
        return text
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", "")
            if part_text:
                chunks.append(str(part_text))
    return "".join(chunks)


def _extract_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass
    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = _extract_json_payload(text)
    return payload if isinstance(payload, dict) else {}


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is not None:
        progress(message)


def _progress_printer(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GIDS against a plain Gemini raw-alert incident agent.")
    parser.add_argument("--input", default="falcon_graph_alerts.csv", help="Path to Falcon graph alerts CSV.")
    parser.add_argument("--output-json", help="Optional JSON report path.")
    parser.add_argument("--output-md", help="Optional Markdown report path.")
    parser.add_argument("--env", default=".env", help="Path to Gemini/GCP .env file.")
    parser.add_argument("--run-gemini", action="store_true", help="Execute the plain Gemini baseline.")
    parser.add_argument(
        "--run-isolated-gemini",
        action="store_true",
        help="Execute only the isolated-alert Gemini classifier baseline.",
    )
    parser.add_argument(
        "--isolated-gemini-max-alerts",
        type=int,
        default=2_000,
        help="Maximum number of alerts to send to the isolated-alert Gemini classifier.",
    )
    parser.add_argument(
        "--isolated-gemini-batch-size",
        type=int,
        default=100,
        help="Alerts per Gemini call for the isolated-alert classifier.",
    )
    parser.add_argument("--prompt-dir", default="reports/prompts", help="Directory for benchmark prompt artifacts.")
    parser.add_argument("--hide-severity", action="store_true", help="Remove severity from model-visible inputs.")
    parser.add_argument("--quiet", action="store_true", help="Disable timestamped progress output.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_gids_vs_plain_gemini(
        args.input,
        run_gemini=args.run_gemini,
        run_isolated_gemini=args.run_isolated_gemini,
        isolated_gemini_max_alerts=args.isolated_gemini_max_alerts,
        isolated_gemini_batch_size=args.isolated_gemini_batch_size,
        env_path=args.env,
        prompt_dir=args.prompt_dir,
        hide_severity=args.hide_severity,
        progress=None if args.quiet else _progress_printer,
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
