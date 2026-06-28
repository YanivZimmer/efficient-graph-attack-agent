from __future__ import annotations

import unittest

from eacs.data_generator import generate_falcon_graph_alerts, summarize_alerts


class FalconDataGeneratorTests(unittest.TestCase):
    def test_hard_profile_breaks_severity_shortcut(self) -> None:
        rows = generate_falcon_graph_alerts(
            num_alerts=300,
            num_incidents=6,
            profile="hard",
            seed=7,
        )
        incident_rows = [row for row in rows if row["is_incident"]]
        noise_rows = [row for row in rows if not row["is_incident"]]

        self.assertTrue(any(row["severity"] in {"Low", "Medium"} for row in incident_rows))
        self.assertTrue(any(row["severity"] in {"High", "Critical"} for row in noise_rows))

        high_critical_ids = {row["alert_id"] for row in rows if row["severity"] in {"High", "Critical"}}
        incident_ids = {row["alert_id"] for row in incident_rows}
        self.assertNotEqual(high_critical_ids, incident_ids)

    def test_generation_is_deterministic(self) -> None:
        left = generate_falcon_graph_alerts(num_alerts=100, num_incidents=3, profile="hard", seed=42)
        right = generate_falcon_graph_alerts(num_alerts=100, num_incidents=3, profile="hard", seed=42)

        self.assertEqual(left, right)

    def test_summary_reports_severity_overlap(self) -> None:
        rows = generate_falcon_graph_alerts(num_alerts=300, num_incidents=6, profile="hard", seed=7)
        summary = summarize_alerts(rows)

        self.assertGreater(summary["high_critical_incident_rate"], 0.0)
        self.assertGreater(summary["high_critical_noise_rate"], 0.0)
        self.assertLess(summary["high_critical_incident_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
