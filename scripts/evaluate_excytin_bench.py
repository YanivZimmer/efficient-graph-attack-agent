#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eacs.excytin import (  # noqa: E402
    ExcytinEvaluationReport,
    ExcytinQAReport,
    run_excytin_evaluation,
    run_excytin_qa_evaluation,
)


DEFAULT_SECRL_ROOT = Path.home() / "Code" / "Datasets" / "SecRL"
DEFAULT_OUTPUT_DIR = Path("reports") / "excytin_bench"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E-ACS evaluation on ExCyTIn-Bench metadata and, when available, "
            "local SecRL question files."
        )
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick smoke runs.")
    parser.add_argument("--batch-size", type=int, default=100, help="Hugging Face dataset-server batch size.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-metadata", action="store_true", help="Skip Hugging Face metadata/path evaluation.")
    parser.add_argument("--skip-qa", action="store_true", help="Skip local SecRL QA baseline evaluation.")
    parser.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    parser.add_argument("--question-set", default="o1")
    parser.add_argument(
        "--qa-context-source",
        action="append",
        choices=["eacs_retrieved", "oracle_metadata"],
        default=None,
        help="QA context source. Repeat to run multiple sources.",
    )
    parser.add_argument(
        "--qa-answer-mode",
        action="append",
        choices=["extractive", "gold_if_present"],
        default=None,
        help="QA answer mode. Repeat to run multiple modes.",
    )
    parser.add_argument(
        "--require-qa",
        action="store_true",
        help="Fail when local SecRL question files are missing instead of skipping QA.",
    )
    return parser.parse_args(argv)


def metadata_output_path(output_dir: Path, split: str, limit: int | None) -> Path:
    return output_dir / f"excytin_metadata_{split}{_limit_suffix(limit)}.json"


def qa_output_path(
    output_dir: Path,
    split: str,
    context_source: str,
    answer_mode: str,
    limit: int | None,
) -> Path:
    return output_dir / f"excytin_qa_{context_source}_{answer_mode}_{split}{_limit_suffix(limit)}.json"


def qa_context_sources(args: argparse.Namespace) -> list[str]:
    return args.qa_context_source or ["eacs_retrieved"]


def qa_answer_modes(args: argparse.Namespace) -> list[str]:
    return args.qa_answer_mode or ["extractive"]


async def run(args: argparse.Namespace) -> int:
    if args.skip_metadata and args.skip_qa:
        print("Nothing to run: both --skip-metadata and --skip-qa were provided.")
        return 0

    if not args.skip_metadata:
        output = metadata_output_path(args.output_dir, args.split, args.limit)
        print(f"[metadata] fetching split={args.split} limit={args.limit or 'all'}")
        report = await run_excytin_evaluation(
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            output=output,
        )
        _print_metadata_summary(report, output)

    if args.skip_qa:
        return 0

    question_dir = args.secrl_root / "secgym" / "questions" / args.question_set / args.split
    if not question_dir.exists():
        message = (
            f"[qa] SecRL question directory not found: {question_dir}. "
            "Clone/download SecRL there or pass --secrl-root. Use --require-qa to fail on this condition."
        )
        if args.require_qa:
            raise FileNotFoundError(message)
        print(message)
        return 0

    for context_source in qa_context_sources(args):
        for answer_mode in qa_answer_modes(args):
            output = qa_output_path(args.output_dir, args.split, context_source, answer_mode, args.limit)
            print(
                "[qa] "
                f"split={args.split} question_set={args.question_set} "
                f"context={context_source} answer_mode={answer_mode} limit={args.limit or 'all'}"
            )
            report = await run_excytin_qa_evaluation(
                secrl_root=args.secrl_root,
                split=args.split,
                question_set=args.question_set,
                context_source=context_source,
                answer_mode=answer_mode,
                limit=args.limit,
                output=output,
            )
            _print_qa_summary(report, output)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        print(f"ExCyTIn-Bench evaluation failed: {exc}", file=sys.stderr)
        return 1


def _limit_suffix(limit: int | None) -> str:
    return f"_limit{limit}" if limit is not None else ""


def _print_metadata_summary(report: ExcytinEvaluationReport, output: Path) -> None:
    print(
        "[metadata] "
        f"rows={report.rows_evaluated} "
        f"avg_path_recall={report.avg_path_alert_recall:.3f} "
        f"end_alert_recall={report.end_alert_recall:.3f} "
        f"exact_path_coverage={report.exact_path_coverage:.3f}"
    )
    print(f"[metadata] wrote {output}")


def _print_qa_summary(report: ExcytinQAReport, output: Path) -> None:
    print(
        "[qa] "
        f"rows={report.rows_evaluated} "
        f"exact_match={report.exact_match_rate:.3f} "
        f"contains_answer={report.contains_answer_rate:.3f} "
        f"answer_rate={report.answer_rate:.3f} "
        f"path_recall={report.avg_path_alert_recall:.3f}"
    )
    print(f"[qa] wrote {output}")


if __name__ == "__main__":
    raise SystemExit(main())
