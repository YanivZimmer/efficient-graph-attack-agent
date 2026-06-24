"""Tests for GraphWeaver baseline and alert-domain dataset loaders."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from clustering.graphweaver import cluster_graphweaver, run_graphweaver_baseline
from data.loaders.ait_ads_loader import load_ait_ads_graph
from data.loaders.darpa2000_loader import load_darpa2000_graph
from data.loaders.iscx2012_loader import load_iscx2012_graph
from tests.test_gnn_pipeline import SAMPLE_RECORDS
from data.graph_builder import build_graph_from_records


class GraphWeaverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = build_graph_from_records(SAMPLE_RECORDS)

    def test_graphweaver_clusters_shared_entity_alerts(self) -> None:
        result = run_graphweaver_baseline(self.artifacts, max_gap_minutes=180)
        grouped = [
            cluster
            for cluster in result.clusters
            if int(cluster["incident_id"]) >= 0 and len(cluster["alert_ids"]) >= 2
        ]
        self.assertGreaterEqual(len(grouped), 1)

    def test_graphweaver_respects_time_window(self) -> None:
        malicious_indices = [0, 1]
        wide_window = cluster_graphweaver(
            self.artifacts,
            malicious_indices=malicious_indices,
            max_gap_minutes=120,
        )
        narrow_window = cluster_graphweaver(
            self.artifacts,
            malicious_indices=malicious_indices,
            max_gap_minutes=30,
        )
        wide_grouped = [cluster for cluster in wide_window if int(cluster["incident_id"]) >= 0]
        narrow_grouped = [cluster for cluster in narrow_window if int(cluster["incident_id"]) >= 0]
        self.assertGreaterEqual(len(wide_grouped), 1)
        self.assertEqual(len(narrow_grouped), 0)


class AlertDomainLoaderTests(unittest.TestCase):
    def test_ait_ads_loader_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = [
                {
                    "timestamp": f"2024-01-01T10:{index:02d}:00+00:00",
                    "Label": "dirb" if index < 5 else "benign",
                    "AMiner": {"ID": "10.0.0.5" if index < 5 else f"10.0.0.{index}"},
                }
                for index in range(10)
            ]
            (root / "fox_aminer.json").write_text(json.dumps(payload), encoding="utf-8")
            artifacts = load_ait_ads_graph(root)
            self.assertEqual(len(artifacts.alert_ids), 10)
            self.assertTrue(artifacts.ground_truth_incidents)

    def test_darpa2000_loader_reads_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "lldos1.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["alert_id", "timestamp", "src_ip", "phase", "label"],
                )
                writer.writeheader()
                for index in range(10):
                    writer.writerow(
                        {
                            "alert_id": f"a{index}",
                            "timestamp": f"2000-02-16T10:{index:02d}:00+00:00",
                            "src_ip": f"1.2.3.{index}",
                            "phase": "ip_sweep" if index < 5 else "normal",
                            "label": "malicious" if index < 5 else "benign",
                        }
                    )
            artifacts = load_darpa2000_graph(root)
            self.assertEqual(len(artifacts.alert_ids), 10)

    def test_iscx2012_loader_reads_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "day3.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["alert_id", "timestamp", "scenario", "src_ip", "label"],
                )
                writer.writeheader()
                for index in range(10):
                    writer.writerow(
                        {
                            "alert_id": f"x{index}",
                            "timestamp": f"2012-06-15T12:{index:02d}:00+00:00",
                            "scenario": "Infiltration" if index < 5 else "Normal",
                            "src_ip": f"10.1.{index}.1",
                            "label": "malicious" if index < 5 else "benign",
                        }
                    )
            artifacts = load_iscx2012_graph(root)
            self.assertEqual(len(artifacts.alert_ids), 10)


if __name__ == "__main__":
    unittest.main()
