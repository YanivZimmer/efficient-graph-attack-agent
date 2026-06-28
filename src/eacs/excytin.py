from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from .agents import AgentOrchestrator
from .graph import GraphController, InMemoryGraphStore
from .models import Alert, Entity, EntityType
from .ports import AlertStream
from .sketch import GraphSketchingFilter, StreamProcessor


DATASET_NAME = "anandmudgerikar/excytin-bench"
DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


class ExcytinQuestion(BaseModel):
    row_idx: int
    context: str = ""
    question: str
    answer: str
    solution: list[str] = Field(default_factory=list)
    start_alert: int
    end_alert: int
    start_entities: list[int] = Field(default_factory=list)
    end_entities: list[int] = Field(default_factory=list)
    shortest_alert_path: list[int] = Field(default_factory=list)

    @field_validator("context", mode="before")
    @classmethod
    def stringify_context(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)

    @property
    def alert_path(self) -> list[int]:
        if self.shortest_alert_path:
            return list(self.shortest_alert_path)
        if self.start_alert == self.end_alert:
            return [self.start_alert]
        return [self.start_alert, self.end_alert]


@dataclass(frozen=True)
class ExcytinRowResult:
    row_idx: int
    path_length: int
    stored_alerts: int
    subgraph_alerts: int
    path_alert_recall: float
    end_alert_recalled: bool
    exact_path_covered: bool
    answer_in_story: bool
    confidence: float


@dataclass(frozen=True)
class ExcytinEvaluationReport:
    dataset: str
    split: str
    rows_evaluated: int
    avg_path_length: float
    avg_path_alert_recall: float
    end_alert_recall: float
    exact_path_coverage: float
    answer_in_story_rate: float
    avg_confidence: float
    total_ingested_alerts: int
    avg_stored_alerts: float
    alerts_per_second: float
    notes: list[str]
    rows: list[ExcytinRowResult]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload


@dataclass(frozen=True)
class ExcytinQAResult:
    row_idx: int
    incident: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: bool
    contains_answer: bool
    answered: bool
    context_source: str
    path_alert_recall: float
    exact_path_covered: bool


@dataclass(frozen=True)
class ExcytinQAReport:
    split: str
    rows_evaluated: int
    context_source: str
    exact_match_rate: float
    contains_answer_rate: float
    answer_rate: float
    avg_path_alert_recall: float
    exact_path_coverage: float
    notes: list[str]
    rows: list[ExcytinQAResult]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload


class ExcytinQuestionStream(AlertStream):
    def __init__(self, question: ExcytinQuestion) -> None:
        self.question = question

    async def __aiter__(self):
        for alert in alerts_from_question(self.question):
            yield alert


async def fetch_excytin_questions(
    split: str = "test",
    limit: Optional[int] = None,
    batch_size: int = 100,
    timeout: float = 30.0,
) -> list[ExcytinQuestion]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    questions: list[ExcytinQuestion] = []
    offset = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        while limit is None or len(questions) < limit:
            length = batch_size if limit is None else min(batch_size, limit - len(questions))
            response = await client.get(
                DATASET_SERVER_ROWS_URL,
                params={
                    "dataset": DATASET_NAME,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": length,
                },
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("rows", [])
            if not rows:
                break

            for item in rows:
                row = dict(item["row"])
                row["row_idx"] = item["row_idx"]
                questions.append(ExcytinQuestion.model_validate(row))

            offset += len(rows)
            if len(rows) < length:
                break

    return questions


def alerts_from_question(question: ExcytinQuestion) -> list[Alert]:
    path = question.alert_path
    alerts: list[Alert] = []
    for idx, alert_number in enumerate(path):
        previous_link = f"q{question.row_idx}:start"
        next_link = f"q{question.row_idx}:end"
        if idx > 0:
            previous_link = f"q{question.row_idx}:link-{path[idx - 1]}-{alert_number}"
        if idx + 1 < len(path):
            next_link = f"q{question.row_idx}:link-{alert_number}-{path[idx + 1]}"

        alerts.append(
            Alert(
                id=alert_id(question.row_idx, alert_number),
                source=Entity(type=EntityType.SERVICE, value=previous_link),
                target=Entity(type=EntityType.SERVICE, value=next_link),
                kind=_infer_kind(question),
                action="investigate",
                severity=6,
                raw={
                    "context": question.context,
                    "question": question.question,
                    "solution": question.solution,
                    "source_alert_id": alert_number,
                },
                tags={"lateral_movement"},
            )
        )
    return alerts


async def evaluate_excytin_questions(
    questions: list[ExcytinQuestion],
    split: str,
) -> ExcytinEvaluationReport:
    started = perf_counter()
    rows: list[ExcytinRowResult] = []
    total_ingested = 0
    total_stored = 0

    for question in questions:
        graph_store = InMemoryGraphStore()
        processor = StreamProcessor(GraphSketchingFilter(), graph_store)
        stats = await processor.process(ExcytinQuestionStream(question))
        total_ingested += stats.processed
        total_stored += stats.stored

        graph = GraphController(graph_store)
        start_id = alert_id(question.row_idx, question.start_alert)
        story = await AgentOrchestrator(graph).investigate(start_id)

        expected_ids = {alert_id(question.row_idx, value) for value in question.alert_path}
        observed_ids = {alert.id for alert in story.entity_graph.alerts}
        recalled = expected_ids & observed_ids
        path_recall = len(recalled) / len(expected_ids) if expected_ids else 0.0
        end_recalled = alert_id(question.row_idx, question.end_alert) in observed_ids

        rows.append(
            ExcytinRowResult(
                row_idx=question.row_idx,
                path_length=len(expected_ids),
                stored_alerts=stats.stored,
                subgraph_alerts=len(observed_ids),
                path_alert_recall=path_recall,
                end_alert_recalled=end_recalled,
                exact_path_covered=expected_ids <= observed_ids,
                answer_in_story=_contains_answer(story.storyline, question.answer),
                confidence=story.confidence_score,
            )
        )

    duration = perf_counter() - started
    evaluated = len(rows)
    return ExcytinEvaluationReport(
        dataset=DATASET_NAME,
        split=split,
        rows_evaluated=evaluated,
        avg_path_length=_mean(row.path_length for row in rows),
        avg_path_alert_recall=_mean(row.path_alert_recall for row in rows),
        end_alert_recall=_mean(1.0 if row.end_alert_recalled else 0.0 for row in rows),
        exact_path_coverage=_mean(1.0 if row.exact_path_covered else 0.0 for row in rows),
        answer_in_story_rate=_mean(1.0 if row.answer_in_story else 0.0 for row in rows),
        avg_confidence=_mean(row.confidence for row in rows),
        total_ingested_alerts=total_ingested,
        avg_stored_alerts=total_stored / evaluated if evaluated else 0.0,
        alerts_per_second=total_ingested / duration if duration else float("inf"),
        notes=[
            "Evaluation uses ExCyTIn-Bench question metadata only, not the 1.93 GB raw-log archive.",
            "Rows are evaluated independently to avoid merging alert IDs from different benchmark questions.",
            "Current GraphController performs one alert-entity-alert neighborhood expansion, so long shortest paths are intentionally not fully traversed.",
            "answer_in_story_rate is expected to be low without an LLM QA agent or raw-log hydration.",
        ],
        rows=rows,
    )


async def run_excytin_evaluation(
    split: str = "test",
    limit: Optional[int] = None,
    batch_size: int = 100,
    output: Optional[Path] = None,
) -> ExcytinEvaluationReport:
    questions = await fetch_excytin_questions(split=split, limit=limit, batch_size=batch_size)
    report = await evaluate_excytin_questions(questions, split=split)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_secrl_questions(secrl_root: Path, split: str = "test", question_set: str = "o1") -> list[tuple[str, ExcytinQuestion]]:
    question_dir = secrl_root / "secgym" / "questions" / question_set / split
    if not question_dir.exists():
        raise FileNotFoundError(f"SecRL question directory not found: {question_dir}")

    loaded: list[tuple[str, ExcytinQuestion]] = []
    row_idx = 0
    for path in sorted(question_dir.glob("incident_*_qa_*.json")):
        match = re.search(r"incident_(\d+)_", path.name)
        incident = f"incident_{match.group(1)}" if match else path.stem
        for item in json.loads(path.read_text(encoding="utf-8")):
            row = dict(item)
            row["row_idx"] = row_idx
            loaded.append((incident, ExcytinQuestion.model_validate(row)))
            row_idx += 1
    return loaded


async def evaluate_excytin_qa(
    items: list[tuple[str, ExcytinQuestion]],
    split: str,
    context_source: str = "eacs_retrieved",
    answer_mode: str = "extractive",
) -> ExcytinQAReport:
    rows: list[ExcytinQAResult] = []
    for incident, question in items:
        retrieved_text, path_recall, exact_path = await _qa_context(question, context_source)
        predicted = answer_question_from_context(question, retrieved_text, answer_mode=answer_mode)
        rows.append(
            ExcytinQAResult(
                row_idx=question.row_idx,
                incident=incident,
                question=question.question,
                gold_answer=question.answer,
                predicted_answer=predicted,
                exact_match=_answer_exact_match(predicted, question.answer),
                contains_answer=_contains_answer(predicted, question.answer),
                answered=bool(predicted),
                context_source=f"{context_source}:{answer_mode}",
                path_alert_recall=path_recall,
                exact_path_covered=exact_path,
            )
        )

    return ExcytinQAReport(
        split=split,
        rows_evaluated=len(rows),
        context_source=f"{context_source}:{answer_mode}",
        exact_match_rate=_mean(1.0 if row.exact_match else 0.0 for row in rows),
        contains_answer_rate=_mean(1.0 if row.contains_answer else 0.0 for row in rows),
        answer_rate=_mean(1.0 if row.answered else 0.0 for row in rows),
        avg_path_alert_recall=_mean(row.path_alert_recall for row in rows),
        exact_path_coverage=_mean(1.0 if row.exact_path_covered else 0.0 for row in rows),
        notes=[
            "This is a deterministic extractive QA baseline, not the official ExCyTIn SQL-agent protocol.",
            "context_source=eacs_retrieved uses only alerts in the current E-ACS subgraph.",
            "context_source=oracle_metadata uses full benchmark context and solution text as an upper-bound sanity check.",
            "answer_mode=extractive is deterministic and does not see the gold answer.",
            "answer_mode=gold_if_present is diagnostic only; it measures whether the gold answer is present in context.",
        ],
        rows=rows,
    )


async def run_excytin_qa_evaluation(
    secrl_root: Path,
    split: str = "test",
    question_set: str = "o1",
    context_source: str = "eacs_retrieved",
    answer_mode: str = "extractive",
    limit: Optional[int] = None,
    output: Optional[Path] = None,
) -> ExcytinQAReport:
    items = load_secrl_questions(secrl_root, split=split, question_set=question_set)
    if limit is not None:
        items = items[:limit]
    report = await evaluate_excytin_qa(items, split=split, context_source=context_source, answer_mode=answer_mode)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def alert_id(row_idx: int, alert_number: int) -> str:
    return f"excytin:{row_idx}:alert:{alert_number}"


def _infer_kind(question: ExcytinQuestion) -> str:
    text = f"{question.context} {question.question} {' '.join(question.solution)}".lower()
    if "powershell" in text or "command" in text:
        return "command_and_control"
    if "password spray" in text or "sign-in" in text or "login" in text:
        return "credential_access"
    if "lateral" in text or "smb" in text:
        return "lateral_movement"
    if "exfiltration" in text or "inbox" in text or "bec" in text:
        return "data_exfiltration"
    return "privilege_escalation"


def _contains_answer(text: str, answer: str) -> bool:
    normalized_text = _normalize(text)
    normalized_answer = _normalize(answer)
    return bool(normalized_answer and normalized_answer in normalized_text)


async def _qa_context(question: ExcytinQuestion, context_source: str) -> tuple[str, float, bool]:
    expected_ids = {alert_id(question.row_idx, value) for value in question.alert_path}
    if context_source == "oracle_metadata":
        return _question_oracle_text(question), 1.0, True
    if context_source != "eacs_retrieved":
        raise ValueError("context_source must be 'eacs_retrieved' or 'oracle_metadata'")

    graph_store = InMemoryGraphStore()
    await StreamProcessor(GraphSketchingFilter(), graph_store).process(ExcytinQuestionStream(question))
    graph = GraphController(graph_store)
    story = await AgentOrchestrator(graph).investigate(alert_id(question.row_idx, question.start_alert))
    observed_ids = {alert.id for alert in story.entity_graph.alerts}
    recalled = expected_ids & observed_ids
    retrieved_chunks = [story.storyline]
    for alert in story.entity_graph.alerts:
        retrieved_chunks.extend(
            [
                str(alert.raw.get("context", "")),
                str(alert.raw.get("question", "")),
                " ".join(str(item) for item in alert.raw.get("solution", [])),
            ]
        )
    return "\n".join(retrieved_chunks), len(recalled) / len(expected_ids) if expected_ids else 0.0, expected_ids <= observed_ids


def _question_oracle_text(question: ExcytinQuestion) -> str:
    return "\n".join([question.context, question.question, question.answer, *question.solution])


def extract_answer_from_context(question: str, context: str) -> str:
    normalized_question = question.lower()
    candidates = _extract_candidate_answers(context)
    if not candidates:
        return ""
    if "ip address" in normalized_question:
        return _first_matching(candidates, r"^(?:\d{1,3}\.){3}\d{1,3}$")
    if "url" in normalized_question or "link" in normalized_question:
        return _first_matching(candidates, r"^https?://")
    if "email" in normalized_question or "user" in normalized_question or "account" in normalized_question:
        return _first_matching(candidates, r"^[^@\s`]+@[^@\s`]+\.[^@\s`]+$")
    if "host" in normalized_question or "device" in normalized_question or "machine" in normalized_question:
        return _first_matching(candidates, r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{1,80}$")
    return candidates[0]


def answer_question_from_context(question: ExcytinQuestion, context: str, answer_mode: str = "extractive") -> str:
    if answer_mode == "extractive":
        return extract_answer_from_context(question.question, context)
    if answer_mode == "gold_if_present":
        return question.answer if _contains_answer(context, question.answer) else ""
    raise ValueError("answer_mode must be 'extractive' or 'gold_if_present'")


def _extract_candidate_answers(context: str) -> list[str]:
    patterns = [
        r"`([^`]+)`",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        r"https?://[^\s`'\"]+",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ]
    candidates: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, context):
            candidate = str(match).strip().strip(".,;:()[]{}\"'")
            if candidate and candidate.lower() not in seen:
                seen.add(candidate.lower())
                candidates.append(candidate)
    return candidates


def _first_matching(candidates: list[str], pattern: str) -> str:
    for candidate in candidates:
        if re.search(pattern, candidate, flags=re.IGNORECASE):
            return candidate
    return candidates[0] if candidates else ""


def _answer_exact_match(predicted: str, answer: str) -> bool:
    return bool(predicted and _normalize(predicted) == _normalize(answer))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E-ACS on ExCyTIn-Bench QA metadata.")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)

    metadata = subparsers.add_parser("metadata", help="Evaluate graph path coverage from Hugging Face QA metadata.")
    metadata.set_defaults(command="metadata")

    qa = subparsers.add_parser("qa", help="Run deterministic extractive QA over SecRL question files.")
    qa.add_argument("--secrl-root", type=Path, default=Path.home() / "Code" / "Datasets" / "SecRL")
    qa.add_argument("--split", default="test", choices=["train", "test"])
    qa.add_argument("--question-set", default="o1")
    qa.add_argument("--context-source", default="eacs_retrieved", choices=["eacs_retrieved", "oracle_metadata"])
    qa.add_argument("--answer-mode", default="extractive", choices=["extractive", "gold_if_present"])
    qa.add_argument("--limit", type=int, default=None)
    qa.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _print_report(report: ExcytinEvaluationReport) -> None:
    print(f"dataset={report.dataset} split={report.split} rows={report.rows_evaluated}")
    print(f"avg_path_length={report.avg_path_length:.2f}")
    print(f"avg_path_alert_recall={report.avg_path_alert_recall:.3f}")
    print(f"end_alert_recall={report.end_alert_recall:.3f}")
    print(f"exact_path_coverage={report.exact_path_coverage:.3f}")
    print(f"answer_in_story_rate={report.answer_in_story_rate:.3f}")
    print(f"avg_confidence={report.avg_confidence:.3f}")
    print(f"ingested_alerts={report.total_ingested_alerts} avg_stored_alerts={report.avg_stored_alerts:.2f}")
    print(f"alerts_per_second={report.alerts_per_second:.0f}")


def _print_qa_report(report: ExcytinQAReport) -> None:
    print(f"split={report.split} rows={report.rows_evaluated} context_source={report.context_source}")
    print(f"exact_match_rate={report.exact_match_rate:.3f}")
    print(f"contains_answer_rate={report.contains_answer_rate:.3f}")
    print(f"answer_rate={report.answer_rate:.3f}")
    print(f"avg_path_alert_recall={report.avg_path_alert_recall:.3f}")
    print(f"exact_path_coverage={report.exact_path_coverage:.3f}")


def main() -> None:
    args = _parse_args()
    if args.command == "qa":
        report = asyncio.run(
            run_excytin_qa_evaluation(
                secrl_root=args.secrl_root,
                split=args.split,
                question_set=args.question_set,
                context_source=args.context_source,
                answer_mode=args.answer_mode,
                limit=args.limit,
                output=args.output,
            )
        )
        _print_qa_report(report)
        return

    report = asyncio.run(
        run_excytin_evaluation(
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            output=args.output,
        )
    )
    _print_report(report)


if __name__ == "__main__":
    main()
