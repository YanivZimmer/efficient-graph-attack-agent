"""Render LaTeX and markdown comparison tables from benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "-"
    if digits == 0:
        return str(int(round(numeric)))
    return f"{numeric:.{digits}f}"


def render_latex_table(summary_rows: list[dict]) -> str:
    """Render a LaTeX table from benchmark summary rows."""
    has_method = any("method" in row for row in summary_rows)
    if has_method:
        headers = [
            "Method",
            "Source",
            "AUC",
            "F1",
            "Precision",
            "Recall",
            "Clusters",
            "Tactic Coherence",
            "Cluster F1",
        ]
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{GNN baseline comparison}",
            "\\begin{tabular}{llrrrrrrr}",
            "\\toprule",
            " & ".join(headers) + " \\\\",
            "\\midrule",
        ]
        for row in summary_rows:
            lines.append(
                " & ".join(
                    [
                        str(row.get("method", "-")),
                        str(row.get("source", "-")),
                        _format_float(row.get("node_auc")),
                        _format_float(row.get("node_f1")),
                        _format_float(row.get("node_precision")),
                        _format_float(row.get("node_recall")),
                        _format_float(row.get("cluster_count"), 0),
                        _format_float(row.get("mean_tactic_coherence")),
                        _format_float(row.get("cluster_f1")),
                    ]
                )
                + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(lines)

    headers = [
        "Dataset",
        "AUC",
        "F1",
        "Clusters",
        "Tactic Coherence",
        "Time Span (h)",
        "Cluster P",
        "Cluster R",
        "Cluster F1",
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{GNN incident discovery benchmark summary}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            " & ".join(
                [
                    row.get("dataset", "-"),
                    _format_float(row.get("node_auc")),
                    _format_float(row.get("node_f1")),
                    _format_float(row.get("cluster_count"), 0),
                    _format_float(row.get("mean_tactic_coherence")),
                    _format_float(row.get("mean_time_span_hours")),
                    _format_float(row.get("cluster_precision")),
                    _format_float(row.get("cluster_recall")),
                    _format_float(row.get("cluster_f1")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main() -> None:
    """Print a LaTeX table for a benchmark or baseline comparison JSON file."""
    parser = argparse.ArgumentParser(description="Render benchmark summary as LaTeX")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/benchmarks/benchmark_summary.json"),
        help="Path to benchmark_summary.json or baseline_comparison.json",
    )
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    print(render_latex_table(rows))


if __name__ == "__main__":
    main()
