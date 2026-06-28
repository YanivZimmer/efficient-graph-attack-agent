from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eacs.gids import (
    GIDSDetector,
    GIDSGeminiRationalizerAgent,
    GIDSIncident,
    GIDSLocalVerdictAgent,
    GIDSRelationshipNovelty,
    GNNLLMHybridReasoner,
    IsolatedGeminiAlertClassifier,
    PlainGeminiIncidentAgent,
    edge_from_falcon_row,
    evaluate_gids_vs_plain_gemini,
)


class GIDSTests(unittest.TestCase):
    def test_maps_falcon_row_to_gids_edge_without_label_leakage(self) -> None:
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

        edge = edge_from_falcon_row(row)

        self.assertEqual(edge.alert_id, "a1")
        self.assertEqual(edge.source_id, "AID-1")
        self.assertEqual(edge.target_id, "AID-2")
        self.assertEqual(edge.severity, 9)
        self.assertIn("host:AID-1", edge.correlation_entities)
        self.assertIn("user:USER-1", edge.correlation_entities)
        self.assertNotIn("is_incident", edge.raw)
        self.assertNotIn("incident_id", edge.raw)

    def test_can_hide_severity_from_gids_edge(self) -> None:
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

        edge = edge_from_falcon_row(row, hide_severity=True)

        self.assertEqual(edge.severity, 5)
        self.assertNotIn("severity", edge.raw)

    def test_plain_gemini_prompt_strips_scoring_labels(self) -> None:
        class DummyClient:
            def generate_text(self, prompt: str) -> str:
                return '{"incidents":[]}'

        row = {
            "alert_id": "a1",
            "timestamp": "2026-05-01 01:00:00",
            "source_node": "AID-1",
            "target_node": "AID-2",
            "user": "USER-1",
            "tactic": "Execution",
            "technique": "T1059",
            "severity": "High",
            "process": "admin.exe",
            "is_incident": "True",
            "incident_id": "INC-1",
        }

        prompt = PlainGeminiIncidentAgent(DummyClient()).build_prompt([row])

        self.assertIn("alert_id", prompt)
        self.assertIn("private analysis pass", prompt)
        self.assertNotIn("is_incident", prompt)
        self.assertNotIn("incident_id", prompt)

    def test_plain_gemini_prompt_can_hide_severity_field(self) -> None:
        class DummyClient:
            def generate_text(self, prompt: str) -> str:
                return '{"incidents":[]}'

        row = {
            "alert_id": "a1",
            "timestamp": "2026-05-01 01:00:00",
            "source_node": "AID-1",
            "target_node": "AID-2",
            "user": "USER-1",
            "tactic": "Execution",
            "technique": "T1059",
            "severity": "Critical",
            "process": "admin.exe",
            "is_incident": "True",
            "incident_id": "INC-1",
        }

        prompt = PlainGeminiIncidentAgent(DummyClient(), include_severity=False).build_prompt([row])

        self.assertNotIn('"severity"', prompt)
        self.assertNotIn("Critical", prompt)

    def test_isolated_gemini_prompt_classifies_alerts_independently(self) -> None:
        class DummyClient:
            def generate_text(self, prompt: str) -> str:
                return '{"alerts":[]}'

        row = {
            "alert_id": "a1",
            "timestamp": "2026-05-01 01:00:00",
            "source_node": "AID-1",
            "target_node": "AID-2",
            "user": "USER-1",
            "tactic": "Execution",
            "technique": "T1059",
            "severity": "Critical",
            "process": "admin.exe",
            "is_incident": "True",
            "incident_id": "INC-1",
        }

        prompt = IsolatedGeminiAlertClassifier(DummyClient(), include_severity=False).build_prompt([row])

        self.assertIn("Classify each CrowdStrike Falcon alert independently", prompt)
        self.assertIn("Do not group alerts", prompt)
        self.assertNotIn("is_incident", prompt)
        self.assertNotIn("incident_id", prompt)
        self.assertNotIn('"severity"', prompt)
        self.assertNotIn("Critical", prompt)

    def test_gids_rationalizer_prompt_uses_candidate_metadata_without_labels(self) -> None:
        class DummyClient:
            def generate_text(self, prompt: str) -> str:
                return '{"candidates":[]}'

        edge = edge_from_falcon_row(
            {
                "alert_id": "a1",
                "timestamp": "2026-05-01 01:00:00",
                "source_node": "AID-1",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "High",
                "process": "admin.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            }
        )
        incident = GIDSIncident(
            incident_id="GIDS-0001",
            alert_ids=["a1"],
            start_time="2026-05-01T01:00:00+00:00",
            end_time="2026-05-01T01:00:00+00:00",
            alert_count=1,
            host_count=2,
            user_count=1,
            max_severity=9,
            tactics=["Execution"],
            pattern_matches=["credential_execution"],
            structural_score=0.9,
            confidence=0.9,
            narrative="candidate",
        )

        prompt = GIDSGeminiRationalizerAgent(DummyClient()).build_prompt([incident], [edge])

        self.assertIn("GNN_SCORE", prompt)
        self.assertIn("GIDS-0001", prompt)
        self.assertNotIn("is_incident", prompt)
        self.assertNotIn("incident_id", prompt)

    def test_gids_rationalizer_prompt_can_hide_severity_field(self) -> None:
        class DummyClient:
            def generate_text(self, prompt: str) -> str:
                return '{"candidates":[]}'

        edge = edge_from_falcon_row(
            {
                "alert_id": "a1",
                "timestamp": "2026-05-01 01:00:00",
                "source_node": "AID-1",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "Critical",
                "process": "admin.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            hide_severity=True,
        )
        incident = GIDSIncident(
            incident_id="GIDS-0001",
            alert_ids=["a1"],
            start_time="2026-05-01T01:00:00+00:00",
            end_time="2026-05-01T01:00:00+00:00",
            alert_count=1,
            host_count=2,
            user_count=1,
            max_severity=5,
            tactics=["Execution"],
            pattern_matches=["credential_execution"],
            structural_score=0.9,
            confidence=0.9,
            narrative="candidate",
        )

        prompt = GIDSGeminiRationalizerAgent(DummyClient(), include_severity=False).build_prompt([incident], [edge])

        self.assertNotIn('"severity"', prompt)
        self.assertNotIn("max_severity", prompt)

    def test_local_verdict_agent_selects_ordered_multistage_candidate(self) -> None:
        rows = [
            ("a1", "Initial Access", "High", "AID-1", "AID-1"),
            ("a2", "Execution", "Medium", "AID-1", "AID-1"),
            ("a3", "Credential Access", "Critical", "AID-1", "AID-1"),
            ("a4", "Lateral Movement", "Medium", "AID-1", "AID-2"),
            ("a5", "Privilege Escalation", "High", "AID-2", "AID-2"),
            ("a6", "Exfiltration", "Medium", "AID-2", "AID-3"),
        ]
        edges = [
            edge_from_falcon_row(
                {
                    "alert_id": alert_id,
                    "timestamp": f"2026-05-01 01:{idx:02d}:00",
                    "source_node": source,
                    "target_node": target,
                    "user": "USER-1",
                    "tactic": tactic,
                    "technique": "T1003",
                    "severity": severity,
                    "process": "agent.exe",
                    "is_incident": "True",
                    "incident_id": "INC-1",
                }
            )
            for idx, (alert_id, tactic, severity, source, target) in enumerate(rows)
        ]
        incident = GIDSIncident(
            incident_id="GIDS-0001",
            alert_ids=[edge.alert_id for edge in edges],
            start_time="2026-05-01T01:00:00+00:00",
            end_time="2026-05-01T01:05:00+00:00",
            alert_count=6,
            host_count=3,
            user_count=1,
            max_severity=10,
            tactics=[row[1] for row in rows],
            pattern_matches=["credential_execution", "credential_lateral_movement", "privilege_exfiltration"],
            structural_score=0.82,
            confidence=0.82,
            narrative="candidate",
        )

        decision = GIDSLocalVerdictAgent().validate([incident], edges)[0]

        self.assertTrue(decision.selected)
        self.assertEqual(decision.verdict, "true_positive")

    def test_relationship_novelty_marks_first_relationships_as_rare(self) -> None:
        rows = [
            {
                "alert_id": f"a{idx}",
                "timestamp": f"2026-05-01 01:{idx:02d}:00",
                "source_node": "AID-1",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "Medium",
                "process": "powershell.exe",
                "is_incident": "False",
                "incident_id": "",
            }
            for idx in range(5)
        ]
        edges = [edge_from_falcon_row(row) for row in rows]

        annotated = GIDSRelationshipNovelty(max_seen=3).annotate(edges)

        self.assertEqual([edge.relationship_seen for edge in annotated], [0, 1, 2, 3, 4])
        self.assertEqual([edge.rare_relationship for edge in annotated], [True, True, True, True, False])

    def test_gids_rare_uses_rare_relationships_as_supporting_signal(self) -> None:
        rows = [
            ("a1", "Initial Access", "AID-1", "AID-1"),
            ("a2", "Execution", "AID-1", "AID-1"),
            ("a3", "Credential Access", "AID-1", "AID-1"),
            ("a4", "Lateral Movement", "AID-1", "AID-2"),
            ("a5", "Exfiltration", "AID-2", "AID-2"),
        ]
        edges = [
            edge_from_falcon_row(
                {
                    "alert_id": alert_id,
                    "timestamp": f"2026-05-01 01:{idx:02d}:00",
                    "source_node": source,
                    "target_node": target,
                    "user": "USER-1",
                    "tactic": tactic,
                    "technique": "T1000",
                    "severity": "Medium",
                    "process": "agent.exe",
                    "is_incident": "True",
                    "incident_id": "INC-1",
                },
                hide_severity=True,
            )
            for idx, (alert_id, tactic, source, target) in enumerate(rows)
        ]
        annotated = GIDSRelationshipNovelty().annotate(edges)

        run = GIDSDetector(reasoner=GNNLLMHybridReasoner(use_rarity=True)).run(annotated)

        self.assertGreater(run.rare_relationships, 0)
        self.assertTrue(run.incidents)

    def test_evaluates_gids_when_gemini_is_not_run(self) -> None:
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
                "alert_id": "inc-3",
                "timestamp": "2026-05-01 01:20:00",
                "source_node": "AID-2",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Credential Access",
                "technique": "T1003",
                "severity": "High",
                "process": "auth.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "inc-4",
                "timestamp": "2026-05-01 01:30:00",
                "source_node": "AID-2",
                "target_node": "AID-3",
                "user": "USER-1",
                "tactic": "Exfiltration",
                "technique": "T1041",
                "severity": "Critical",
                "process": "sync.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
            {
                "alert_id": "noise-1",
                "timestamp": "2026-05-01 06:00:00",
                "source_node": "AID-8",
                "target_node": "AID-9",
                "user": "USER-8",
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

            report = evaluate_gids_vs_plain_gemini(path)

        detectors = {row.detector: row for row in report.detectors}
        gids = detectors["GIDS"]
        gids_local = detectors["GIDS + local verdict agent"]
        gids_rare = detectors["gids_rare"]
        gids_rare_agent = detectors["gids_rare_with_agent"]
        gids_llm = detectors["GIDS + Gemini rationalizer"]
        gemini = detectors["Plain Gemini raw-alert agent"]
        isolated = detectors["Gemini isolated-alert classifier"]
        local = detectors["Local analyst severity-chain review"]
        self.assertEqual(gids.status, "ok")
        self.assertEqual(gids.true_positives, 4)
        self.assertEqual(gids.false_negatives, 0)
        self.assertEqual(gids.incident_recall_any, 1.0)
        self.assertEqual(gids_local.status, "ok")
        self.assertEqual(gids_rare.status, "ok")
        self.assertEqual(gids_rare_agent.status, "ok")
        self.assertEqual(gids_llm.status, "not_run")
        self.assertEqual(gemini.status, "not_run")
        self.assertEqual(isolated.status, "not_run")
        self.assertEqual(local.status, "ok")
        self.assertEqual(local.true_positives, 4)

    def test_evaluation_can_hide_severity(self) -> None:
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
                "alert_id": "inc-2",
                "timestamp": "2026-05-01 01:10:00",
                "source_node": "AID-1",
                "target_node": "AID-2",
                "user": "USER-1",
                "tactic": "Execution",
                "technique": "T1059",
                "severity": "Low",
                "process": "admin.exe",
                "is_incident": "True",
                "incident_id": "INC-1",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "falcon.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            report = evaluate_gids_vs_plain_gemini(path, hide_severity=True)

        local = next(row for row in report.detectors if row.detector == "Local analyst severity-chain review")
        self.assertEqual(local.status, "severity_hidden")
        self.assertTrue(any("severity" in note for note in report.notes))


if __name__ == "__main__":
    unittest.main()
