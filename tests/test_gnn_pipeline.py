"""Tests for the GNN incident discovery pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.graph_builder import build_graph_from_records
from data.schema import infer_schema, label_to_binary


SAMPLE_RECORDS = [
    {
        "alert_id": "a1",
        "label": "True Positive - Malicious",
        "timestamp": "2026-01-01T10:00:00+00:00",
        "severity": "High",
        "traces": [
            {
                "timestamp": "2026-01-01T10:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-a",
                            "UserName": "alice",
                            "FileName": "cmd.exe",
                            "LocalIP": "10.0.0.1",
                            "MitreAttack": [{"Tactic": "Execution", "Technique": "T1059"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a2",
        "label": "True Positive - Malicious",
        "timestamp": "2026-01-01T11:00:00+00:00",
        "severity": "Medium",
        "traces": [
            {
                "timestamp": "2026-01-01T11:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-a",
                            "UserName": "alice",
                            "FileName": "powershell.exe",
                            "LocalIP": "10.0.0.1",
                            "MitreAttack": [{"Tactic": "Execution", "Technique": "T1059"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a3",
        "label": "False Positive",
        "timestamp": "2026-01-02T09:00:00+00:00",
        "severity": "Low",
        "traces": [
            {
                "timestamp": "2026-01-02T09:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-b",
                            "UserName": "bob",
                            "FileName": "explorer.exe",
                            "LocalIP": "10.0.0.2",
                            "MitreAttack": [{"Tactic": "Discovery", "Technique": "T1082"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a4",
        "label": "True Positive - Benign",
        "timestamp": "2026-01-02T10:00:00+00:00",
        "severity": "Low",
        "traces": [
            {
                "timestamp": "2026-01-02T10:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-c",
                            "UserName": "carol",
                            "FileName": "notepad.exe",
                            "LocalIP": "10.0.0.3",
                            "MitreAttack": [{"Tactic": "Discovery", "Technique": "T1082"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a5",
        "label": "True Positive - Malicious",
        "timestamp": "2026-01-03T10:00:00+00:00",
        "severity": "High",
        "traces": [
            {
                "timestamp": "2026-01-03T10:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-a",
                            "UserName": "alice",
                            "FileName": "evil.exe",
                            "LocalIP": "10.0.0.1",
                            "MitreAttack": [{"Tactic": "Execution", "Technique": "T1059"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a6",
        "label": "True Positive - Malicious",
        "timestamp": "2026-01-03T11:00:00+00:00",
        "severity": "High",
        "traces": [
            {
                "timestamp": "2026-01-03T11:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-d",
                            "UserName": "dave",
                            "FileName": "mal.exe",
                            "LocalIP": "10.0.0.4",
                            "MitreAttack": [{"Tactic": "Lateral Movement", "Technique": "T1021"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a7",
        "label": "False Positive",
        "timestamp": "2026-01-04T09:00:00+00:00",
        "severity": "Low",
        "traces": [
            {
                "timestamp": "2026-01-04T09:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-e",
                            "UserName": "erin",
                            "FileName": "browser.exe",
                            "LocalIP": "10.0.0.5",
                            "MitreAttack": [{"Tactic": "Discovery", "Technique": "T1082"}],
                        }
                    }
                },
            }
        ],
    },
    {
        "alert_id": "a8",
        "label": "True Positive - Benign",
        "timestamp": "2026-01-04T10:00:00+00:00",
        "severity": "Low",
        "traces": [
            {
                "timestamp": "2026-01-04T10:00:00+00:00",
                "input": {
                    "RawPayload": {
                        "event": {
                            "Hostname": "host-f",
                            "UserName": "frank",
                            "FileName": "update.exe",
                            "LocalIP": "10.0.0.6",
                            "MitreAttack": [{"Tactic": "Discovery", "Technique": "T1082"}],
                        }
                    }
                },
            }
        ],
    },
]


class GNNPipelineTests(unittest.TestCase):
    def test_schema_inference_and_label_mapping(self) -> None:
        schema = infer_schema(SAMPLE_RECORDS)
        self.assertEqual(schema.alert_id_field, "alert_id")
        self.assertEqual(schema.label_field, "label")
        self.assertIn("host", schema.entity_fields)
        self.assertEqual(label_to_binary("True Positive - Malicious"), 1)
        self.assertEqual(label_to_binary("False Positive"), 0)

    def test_graph_builder_creates_hetero_graph(self) -> None:
        artifacts = build_graph_from_records(SAMPLE_RECORDS)
        self.assertEqual(len(artifacts.alert_ids), 8)
        self.assertTrue(hasattr(artifacts.data["alert"], "x"))
        self.assertTrue(hasattr(artifacts.data["alert"], "train_mask"))
        self.assertIn(("alert", "connects_to", "host"), artifacts.data.edge_types)

    def test_graph_builder_can_add_alert_alert_edges(self) -> None:
        artifacts = build_graph_from_records(
            SAMPLE_RECORDS,
            include_alert_alert_edges=True,
            alert_link_hours=48.0,
            max_alert_neighbors_per_relation=4,
        )
        self.assertIn(("alert", "same_host", "alert"), artifacts.data.edge_types)
        self.assertIn(("alert", "precedes", "alert"), artifacts.data.edge_types)
        self.assertGreater(artifacts.data["alert", "same_host", "alert"].edge_index.size(1), 0)
        self.assertGreater(artifacts.data["alert", "precedes", "alert"].edge_index.size(1), 0)

    def test_primary_loader_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in SAMPLE_RECORDS:
                    handle.write(json.dumps(record) + "\n")
            from data.loaders.primary_loader import load_primary_graph

            artifacts = load_primary_graph(path)
            self.assertEqual(len(artifacts.alert_records), 8)


if __name__ == "__main__":
    unittest.main()
