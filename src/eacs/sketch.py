from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from .models import Alert
from .ports import AlertStream, GraphStore


class CountMinSketch:
    """Small count-min sketch for approximate relationship frequency."""

    def __init__(self, width: int = 20_000, depth: int = 5) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self._rows = [[0] * width for _ in range(depth)]

    def add(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for row, idx in enumerate(self._indexes(key)):
            self._rows[row][idx] += count

    def estimate(self, key: str) -> int:
        return min(self._rows[row][idx] for row, idx in enumerate(self._indexes(key)))

    def _indexes(self, key: str) -> tuple[int, ...]:
        encoded = key.encode("utf-8")
        indexes: list[int] = []
        for salt in range(self.depth):
            digest = hashlib.blake2b(encoded, digest_size=8, person=f"eacs{salt}".encode())
            indexes.append(int.from_bytes(digest.digest(), "big") % self.width)
        return tuple(indexes)


@dataclass(frozen=True)
class SketchDecision:
    store: bool
    reason: str
    estimated_frequency: int


@dataclass(frozen=True)
class ProcessingStats:
    processed: int
    stored: int
    duration_seconds: float

    @property
    def alerts_per_second(self) -> float:
        if self.duration_seconds == 0:
            return float("inf")
        return self.processed / self.duration_seconds

    @property
    def stored_ratio(self) -> float:
        if self.processed == 0:
            return 0.0
        return self.stored / self.processed


class GraphSketchingFilter:
    """Threshold-based graph sketching gate for alert relationships."""

    ATTACK_KINDS = {
        "initial_access",
        "credential_access",
        "failed_login_burst",
        "execution",
        "lateral_movement",
        "privilege_escalation",
        "defense_evasion",
        "discovery",
        "collection",
        "data_exfiltration",
        "command_and_control",
        "impact",
    }

    def __init__(
        self,
        sketch: Optional[CountMinSketch] = None,
        baseline_threshold: int = 10_000,
        rare_relationship_max_seen: int = 3,
        rare_relationship_min_severity: int = 5,
    ) -> None:
        self.sketch = sketch or CountMinSketch()
        self.baseline_threshold = baseline_threshold
        self.rare_relationship_max_seen = rare_relationship_max_seen
        self.rare_relationship_min_severity = rare_relationship_min_severity

    def seed_baseline(self, relationship_key: str, count: int) -> None:
        self.sketch.add(relationship_key, count)

    def evaluate(self, alert: Alert) -> SketchDecision:
        key = alert.relationship_key
        seen = self.sketch.estimate(key)
        self.sketch.add(key)

        if alert.is_high_severity:
            return SketchDecision(True, "high_severity", seen)
        if seen > self.baseline_threshold:
            return SketchDecision(False, "baseline_too_common", seen)
        if self._matches_attack_topology(alert):
            return SketchDecision(True, "attack_topology", seen)
        if seen <= self.rare_relationship_max_seen and alert.severity >= self.rare_relationship_min_severity:
            return SketchDecision(True, "rare_relationship", seen)
        return SketchDecision(False, "not_interesting", seen)

    def _matches_attack_topology(self, alert: Alert) -> bool:
        labels = {alert.kind.lower(), alert.action.lower()}
        labels.update(tag.lower() for tag in alert.tags)
        return bool(labels & self.ATTACK_KINDS)


class StreamProcessor:
    """Async ingestion pipeline that stores only sketch-approved alerts."""

    def __init__(self, sketch_filter: GraphSketchingFilter, graph_store: GraphStore) -> None:
        self.sketch_filter = sketch_filter
        self.graph_store = graph_store

    async def process(self, stream: AlertStream, limit: Optional[int] = None) -> ProcessingStats:
        processed = 0
        stored = 0
        started = perf_counter()

        async for alert in stream:
            decision = self.sketch_filter.evaluate(alert)
            processed += 1
            if decision.store:
                await self.graph_store.upsert_alert(alert)
                stored += 1
            if limit is not None and processed >= limit:
                break

        return ProcessingStats(
            processed=processed,
            stored=stored,
            duration_seconds=perf_counter() - started,
        )
