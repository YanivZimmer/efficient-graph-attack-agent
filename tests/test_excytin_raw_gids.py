from __future__ import annotations

import json
import unittest

from eacs.excytin_raw_gids import edge_from_security_alert_row
from eacs.gids import HIDDEN_SEVERITY_SCORE


class ExcytinRawGIDSTests(unittest.TestCase):
    def test_maps_security_alert_row_to_gids_edge_without_label_leakage(self) -> None:
        row = {
            "SystemAlertId": "alert-1",
            "StartTime": "2024-06-20 08:51:07+00:00",
            "AlertName": "Suspicious access to LSASS service",
            "AlertSeverity": "High",
            "Tactics": "DefenseEvasion, CredentialAccess, LateralMovement",
            "Techniques": '["T1003","T1550"]',
            "Entities": json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "user@example.com"},
                    {"Type": "host", "HostName": "host-1"},
                    {"Type": "ip", "Address": "10.0.0.5"},
                ]
            ),
            "IsIncident": "True",
            "IncidentNumber": "123",
        }

        edge = edge_from_security_alert_row(row, source_name="incident_5")

        self.assertEqual(edge.alert_id, "alert-1")
        self.assertEqual(edge.tactic, "Lateral Movement")
        self.assertEqual(edge.user_id, "user@example.com")
        self.assertEqual(edge.severity, 9)
        self.assertNotIn("IsIncident", edge.raw)
        self.assertNotIn("IncidentNumber", edge.raw)

    def test_can_hide_security_alert_severity(self) -> None:
        row = {
            "SystemAlertId": "alert-1",
            "StartTime": "2024-06-20 08:51:07+00:00",
            "AlertName": "Password spray attack",
            "AlertSeverity": "High",
            "Tactics": "CredentialAccess",
            "Entities": "[]",
        }

        edge = edge_from_security_alert_row(row, source_name="incident_5", hide_severity=True)

        self.assertEqual(edge.severity, HIDDEN_SEVERITY_SCORE)
        self.assertNotIn("AlertSeverity", edge.raw)


if __name__ == "__main__":
    unittest.main()
