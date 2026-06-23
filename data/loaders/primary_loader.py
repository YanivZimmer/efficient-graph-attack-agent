"""Loader for the primary tenant JSONL alert dataset."""

from __future__ import annotations

import json
from pathlib import Path

from data.graph_builder import AlertGraphArtifacts, build_graph_from_records


def load_jsonl_records(path: Path) -> list[dict]:
    """Load alert records from a JSONL file."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_primary_graph(data_path: Path) -> AlertGraphArtifacts:
    """Load the primary SOC JSONL dataset into graph artifacts."""
    records = load_jsonl_records(data_path)
    return build_graph_from_records(records)
