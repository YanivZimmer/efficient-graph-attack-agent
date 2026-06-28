from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eacs.data_generator import generate_falcon_graph_alerts, write_alerts_csv
from eacs.error_analysis import analyze_falcon_errors


class FalconErrorAnalysisTests(unittest.TestCase):
    def test_analyzes_hard_dataset_shortcuts_and_local_verdict(self) -> None:
        rows = generate_falcon_graph_alerts(num_alerts=320, num_incidents=6, profile="hard", seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard.csv"
            write_alerts_csv(rows, path)

            report = analyze_falcon_errors(path)

        names = [detector.detector for detector in report.detectors]
        self.assertIn("High/Critical severity baseline", names)
        self.assertIn("GIDS + local verdict agent", names)
        self.assertIn("gids_rare", names)
        self.assertIn("gids_rare_with_agent", names)
        self.assertFalse(report.golden_feature_audit.strict_label_leakage_found)
        self.assertLess(report.dataset.noise_high_critical_rate, 1.0)
        severity = next(detector for detector in report.detectors if detector.detector == "High/Critical severity baseline")
        self.assertGreater(severity.false_positives, 0)
        self.assertGreater(severity.false_negatives, 0)

    def test_analyzes_hard_dataset_with_hidden_severity(self) -> None:
        rows = generate_falcon_graph_alerts(num_alerts=320, num_incidents=6, profile="hard", seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard.csv"
            write_alerts_csv(rows, path)

            report = analyze_falcon_errors(path, hide_severity=True)

        severity = next(detector for detector in report.detectors if detector.detector == "High/Critical severity baseline")
        local = next(detector for detector in report.detectors if detector.detector == "Local analyst severity-chain review")
        gids_rare = next(detector for detector in report.detectors if detector.detector == "gids_rare")
        self.assertEqual(severity.selected_alerts, 0)
        self.assertEqual(local.selected_alerts, 0)
        self.assertGreaterEqual(gids_rare.selected_alerts, 0)
        self.assertTrue(any("Severity was hidden" in item for item in report.golden_feature_audit.shortcut_findings))


if __name__ == "__main__":
    unittest.main()
