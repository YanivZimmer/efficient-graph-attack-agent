from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eacs.falcon import alert_from_falcon_row, evaluate_falcon_graph_alerts


class FalconTests(unittest.TestCase):
    def test_maps_falcon_row_to_eacs_alert_without_using_labels(self) -> None:
        row = {
            "alert_id": "a1",
            "timestamp": "2026-05-01 01:00:00",
            "source_node": "AID-1",
            "target_node": "AID-2",
            "user": "USER-1",
            "tactic": "Credential Access",
            "technique": "T1003",
            "severity": "High",
            "process": "analysis.exe",
            "is_incident": "True",
            "incident_id": "INC-1",
        }

        alert = alert_from_falcon_row(row)

        self.assertEqual(alert.id, "a1")
        self.assertEqual(alert.source.value, "AID-1")
        self.assertEqual(alert.target.value, "AID-2")
        self.assertEqual(alert.kind, "credential_access")
        self.assertEqual(alert.severity, 9)
        self.assertIn("credential_access", alert.tags)
        self.assertNotIn("is_incident", alert.raw)
        self.assertNotIn("incident_id", alert.raw)

    def test_can_strip_mitre_features_from_falcon_alert(self) -> None:
        row = {
            "alert_id": "a1",
            "timestamp": "2026-05-01 01:00:00",
            "source_node": "AID-1",
            "target_node": "AID-2",
            "user": "USER-1",
            "tactic": "Credential Access",
            "technique": "T1003",
            "severity": "High",
            "process": "analysis.exe",
            "is_incident": "True",
            "incident_id": "INC-1",
        }

        alert = alert_from_falcon_row(row, include_mitre_tactics=False)

        self.assertEqual(alert.kind, "falcon_alert")
        self.assertEqual(alert.tags, set())
        self.assertEqual(alert.raw["AlertName"], "analysis.exe")
        self.assertNotIn("tactic", alert.raw)
        self.assertNotIn("technique", alert.raw)

    def test_can_hide_severity_from_falcon_alert(self) -> None:
        row = {
            "alert_id": "a1",
            "timestamp": "2026-05-01 01:00:00",
            "source_node": "AID-1",
            "target_node": "AID-2",
            "user": "USER-1",
            "tactic": "Credential Access",
            "technique": "T1003",
            "severity": "Critical",
            "process": "analysis.exe",
            "is_incident": "True",
            "incident_id": "INC-1",
        }

        alert = alert_from_falcon_row(row, hide_severity=True)

        self.assertEqual(alert.severity, 5)
        self.assertNotIn("severity", alert.raw)

    def test_evaluates_falcon_detection_and_discovery(self) -> None:
        rows = [
            {
                "alert_id": "inc-1",
                "timestamp": "2026-05-01 01:00:00",
                "source_node": "AID-1",
                "target_node": "AID-1",
                "user": "USER-1",
                "tactic": "Initial Access",
                "technique": "T1566",
                "severity": "High",
                "process": "office.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "inc-2",
                "timestamp": "2026-05-01 01:10:00",
                "source_node": "AID-1",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Lateral Movement",
                "technique": "T1021",
                "severity": "Critical",
                "process": "admin.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "noise-1",
                "timestamp": "2026-05-01 02:00:00",
                "source_node": "AID-9",
                "target_node": "AID-10",
                "user": "USER-9",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "Low",
                "process": "utility.exe",
                "is_incident": "False",
                "incident_id": "",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "falcon.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            report = evaluate_falcon_graph_alerts(path)

        eacs = report.detection_rows[0]
        severity = report.detection_rows[1]
        self.assertEqual(eacs.true_positives, 2)
        self.assertEqual(eacs.false_positives, 1)
        self.assertEqual(eacs.recall, 1.0)
        self.assertEqual(severity.false_positives, 0)
        self.assertEqual(report.discovery.known_overlap_clusters, 1)
        self.assertEqual(report.discovery.detected_ground_truth_incidents, 1)

    def test_evaluates_falcon_without_mitre_tactics(self) -> None:
        rows = [
            {
                "alert_id": "inc-1",
                "timestamp": "2026-05-01 01:00:00",
                "source_node": "AID-1",
                "target_node": "AID-1",
                "user": "USER-1",
                "tactic": "Initial Access",
                "technique": "T1566",
                "severity": "High",
                "process": "office.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "inc-2",
                "timestamp": "2026-05-01 01:10:00",
                "source_node": "AID-1",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Lateral Movement",
                "technique": "T1021",
                "severity": "Critical",
                "process": "admin.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "noise-1",
                "timestamp": "2026-05-01 02:00:00",
                "source_node": "AID-9",
                "target_node": "AID-10",
                "user": "USER-9",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "Low",
                "process": "utility.exe",
                "is_incident": "False",
                "incident_id": "",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "falcon.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            report = evaluate_falcon_graph_alerts(path, include_mitre_tactics=False)

        eacs = report.detection_rows[0]
        self.assertEqual(report.feature_mode, "without_mitre_tactics")
        self.assertEqual(eacs.true_positives, 2)
        self.assertEqual(eacs.false_positives, 0)
        self.assertEqual(report.discovery.known_overlap_clusters, 1)

    def test_evaluates_falcon_with_hidden_severity(self) -> None:
        rows = [
            {
                "alert_id": "inc-1",
                "timestamp": "2026-05-01 01:00:00",
                "source_node": "AID-1",
                "target_node": "AID-1",
                "user": "USER-1",
                "tactic": "Initial Access",
                "technique": "T1566",
                "severity": "Critical",
                "process": "office.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "noise-1",
                "timestamp": "2026-05-01 02:00:00",
                "source_node": "AID-9",
                "target_node": "AID-10",
                "user": "USER-9",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "Critical",
                "process": "utility.exe",
                "is_incident": "False",
                "incident_id": "",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "falcon.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            report = evaluate_falcon_graph_alerts(path, hide_severity=True)

        severity = report.detection_rows[1]
        self.assertEqual(report.feature_mode, "with_mitre_tactics_severity_hidden")
        self.assertEqual(severity.selected_alerts, 0)
        self.assertEqual(severity.false_negatives, 1)


if __name__ == "__main__":
    unittest.main()
