from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from .graph import GraphController
from .models import Alert, AttackStory, Subgraph, ValidationResult
from .ports import LLMProvider, LogStore
from .telemetry import get_tracer


TRACER = get_tracer(__name__)


class SummarizerAgent:
    def summarize(self, alert: Alert) -> str:
        target = f" against {alert.target.type.value}:{alert.target.value}" if alert.target else ""
        tags = f" tags={','.join(sorted(alert.tags))}" if alert.tags else ""
        return (
            f"{alert.source.type.value}:{alert.source.value} performed {alert.action} "
            f"for {alert.kind}{target} with severity {alert.severity}.{tags}"
        )


class DiscoveryAgent:
    def __init__(self, graph: GraphController) -> None:
        self.graph = graph

    async def run(self, alert_id: str) -> Subgraph:
        with TRACER.start_as_current_span("agent.discovery") as span:
            subgraph = await self.graph.build_subgraph(alert_id)
            span.set_attribute("alert_id", alert_id)
            span.set_attribute("subgraph.alert_count", len(subgraph.alerts))
            return subgraph


class ValidationAgent:
    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm

    async def run(self, root: Alert, subgraph: Subgraph, event_intents: Sequence[str]) -> ValidationResult:
        with TRACER.start_as_current_span("agent.validation") as span:
            confidence = self._score(root, subgraph)
            reasons = self._reasons(root, subgraph)
            if self.llm:
                prompt = self._prompt(event_intents, confidence)
                response = await self.llm.generate(prompt)
                if response:
                    reasons.append(response[:500])
            span.set_attribute("alert_id", root.id)
            span.set_attribute("validation.confidence", confidence)
            return ValidationResult(confidence=confidence, reasons=reasons)

    def _score(self, root: Alert, subgraph: Subgraph) -> float:
        score = 0.15 + (root.severity / 10) * 0.4
        if any(tag in root.tags for tag in {"lateral_movement", "privilege_escalation", "data_exfiltration"}):
            score += 0.2
        score += min(len(subgraph.alerts) - 1, 5) * 0.05
        score += min(len(subgraph.entities), 5) * 0.02
        return min(score, 1.0)

    def _reasons(self, root: Alert, subgraph: Subgraph) -> list[str]:
        reasons = [f"root severity={root.severity}", f"related_alerts={max(len(subgraph.alerts) - 1, 0)}"]
        if root.tags:
            reasons.append(f"tags={','.join(sorted(root.tags))}")
        return reasons

    def _prompt(self, event_intents: Sequence[str], confidence: float) -> str:
        joined = "\n".join(f"- {intent}" for intent in event_intents)
        return (
            "Validate whether these compressed security events form a feasible attacker path. "
            f"Rule-based confidence is {confidence:.2f}. Return one concise reason.\n{joined}"
        )


class SynthesizerAgent:
    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm

    async def run(
        self,
        subgraph: Subgraph,
        event_intents: Sequence[str],
        validation: ValidationResult,
        hydrated_logs: list[dict[str, object]],
    ) -> str:
        fallback = self._fallback_story(event_intents, validation)
        if not self.llm:
            return fallback

        with TRACER.start_as_current_span("agent.synthesizer") as span:
            prompt = (
                "Compile a concise attack storyline from these event intents and validation notes.\n"
                f"Events:\n{chr(10).join(event_intents)}\n"
                f"Validation: {', '.join(validation.reasons)}\n"
                f"Hydrated logs available: {len(hydrated_logs)}"
            )
            story = await self.llm.generate(prompt)
            span.set_attribute("subgraph.alert_count", len(subgraph.alerts))
            return story or fallback

    def _fallback_story(self, event_intents: Sequence[str], validation: ValidationResult) -> str:
        if not event_intents:
            return "No related alerts were found for this investigation."
        first = event_intents[0]
        extra = f" {len(event_intents) - 1} related event(s) provide graph context." if len(event_intents) > 1 else ""
        return f"{first}{extra} Confidence is {validation.confidence:.2f}."


class AgentOrchestrator:
    def __init__(
        self,
        graph: GraphController,
        llm: Optional[LLMProvider] = None,
        log_store: Optional[LogStore] = None,
        hydration_threshold: float = 0.7,
    ) -> None:
        self.graph = graph
        self.log_store = log_store
        self.hydration_threshold = hydration_threshold
        self.summarizer = SummarizerAgent()
        self.discovery = DiscoveryAgent(graph)
        self.validation = ValidationAgent(llm)
        self.synthesizer = SynthesizerAgent(llm)

    async def investigate(self, alert_id: str) -> AttackStory:
        root = await self.graph.fetch_alert(alert_id)
        if root is None:
            raise ValueError(f"alert not found: {alert_id}")

        subgraph = await self.discovery.run(alert_id)
        event_intents = [self.summarizer.summarize(alert) for alert in subgraph.alerts]
        validation = await self.validation.run(root, subgraph, event_intents)

        hydrated_logs: list[dict[str, object]] = []
        if validation.confidence > self.hydration_threshold and self.log_store:
            with TRACER.start_as_current_span("tool.log_fetch") as span:
                hydrated_logs = await self.log_store.fetch_logs([alert.id for alert in subgraph.alerts])
                span.set_attribute("hydrated_logs", len(hydrated_logs))

        storyline = await self.synthesizer.run(subgraph, event_intents, validation, hydrated_logs)
        return AttackStory(
            storyline=storyline,
            confidence_score=validation.confidence,
            entity_graph=subgraph,
            event_intents=list(event_intents),
            hydrated_logs=hydrated_logs,
        )
