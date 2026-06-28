#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eacs.excytin_raw_gids import evaluate_excytin_raw_gids  # noqa: E402
from eacs.secrl import DEFAULT_SECRL_ROOT  # noqa: E402


DEFAULT_DATA_ROOT = DEFAULT_SECRL_ROOT / "secgym" / "database" / "data_anonymized"
DEFAULT_OUTPUT_DIR = Path("reports") / "excytin_bench"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GIDS variants on ExCyTIn-Bench/SecRL raw SecurityAlert rows without QA."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--secrl-root", type=Path, default=DEFAULT_SECRL_ROOT)
    parser.add_argument("--scope", default="full", help="'full', 'incidents', or a single incident_<id> folder.")
    parser.add_argument("--ground-truth", default="incident-graphs", choices=["incident-graphs", "security-incidents"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hide-severity", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suffix = _suffix(args.scope, args.ground_truth, args.limit, args.hide_severity)
    output_json = args.output_json or args.output_dir / f"excytin_raw_gids_{suffix}.json"
    output_md = args.output_md or args.output_dir / f"excytin_raw_gids_{suffix}.md"
    try:
        report = evaluate_excytin_raw_gids(
            args.data_root,
            scope=args.scope,
            ground_truth=args.ground_truth,
            secrl_root=args.secrl_root,
            limit=args.limit,
            hide_severity=args.hide_severity,
        )
    except Exception as exc:
        print(f"ExCyTIn raw GIDS evaluation failed: {exc}", file=sys.stderr)
        return 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(report.to_markdown(), encoding="utf-8")
    _print_summary(report, output_json, output_md)
    return 0


def _suffix(scope: str, ground_truth: str, limit: int | None, hide_severity: bool) -> str:
    parts = [scope, ground_truth.replace("-", "_")]
    if limit is not None:
        parts.append(f"limit{limit}")
    if hide_severity:
        parts.append("no_severity")
    return "_".join(parts)


def _print_summary(report, output_json: Path, output_md: Path) -> None:
    print(
        f"scope={report.scope} candidate_alerts={report.candidate_alerts} "
        f"available_gt_alerts={report.available_ground_truth_alerts} "
        f"gt_incidents={report.ground_truth_incidents}"
    )
    for row in report.detectors:
        print(
            f"{row.detector}: status={row.status} selected={row.selected_alerts} "
            f"precision={row.precision:.3f} recall={row.recall:.3f} "
            f"f1={row.f1:.3f} incident_any={row.incident_recall_any:.3f}"
        )
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    raise SystemExit(main())
