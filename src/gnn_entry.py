"""Console entry points for the GNN incident discovery pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def run_incident() -> None:
    """Run the end-to-end GNN incident discovery pipeline."""
    _ensure_project_root_on_path()
    from main import main

    main()


def run_benchmark() -> None:
    """Run benchmark datasets through the GNN pipeline."""
    _ensure_project_root_on_path()
    from benchmarks.run_benchmarks import main

    main()


def run_benchmark_table() -> None:
    """Render a LaTeX table from benchmark summary JSON."""
    _ensure_project_root_on_path()
    from benchmarks.comparison_table import main

    main()


def run_baseline_comparison() -> None:
    """Compare HGAT against literature GNN baselines."""
    _ensure_project_root_on_path()
    from benchmarks.run_baseline_comparison import main

    main()
