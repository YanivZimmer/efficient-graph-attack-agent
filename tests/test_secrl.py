from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eacs.secrl import (
    REDUCE_C2_WEIGHT,
    REQUIRE_PROGRESSION_OR_SEVERITY,
    SEPARATOR,
    STRICT_ENTITY_CONTINUITY,
    SUPPRESS_GENERIC_ICS,
    alert_from_security_alert_row,
    analyze_secrl_errors,
    audit_secrl_discovery_leakage,
    compare_secrl_discovery_baselines,
    discover_secrl_incidents,
    evaluate_secrl_alert_detection,
    run_secrl_discovery_ablation,
)


class SecRLTests(unittest.IsolatedAsyncioTestCase):
    def test_maps_security_alert_to_eacs_alert(self) -> None:
        row = {
            "SystemAlertId": "a1",
            "AlertName": "Ongoing hands-on-keyboard attack via Impacket toolkit",
            "AlertSeverity": "High",
            "Entities": json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "admin@example.com"},
                    {"Type": "host", "HostName": "server-1"},
                ]
            ),
        }

        alert = alert_from_security_alert_row(row, "incident_1")

        self.assertEqual(alert.id, "a1")
        self.assertEqual(alert.source.value, "admin@example.com")
        self.assertEqual(alert.target.value, "server-1")
        self.assertEqual(alert.severity, 9)
        self.assertIn("lateral_movement", alert.tags)

    async def test_evaluates_detection_against_security_incident_alert_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["positive-1", "positive-2"])])
                + "\n",
                encoding="utf-8",
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "Entities"]) + "\n"
                + SEPARATOR.join(["positive-1", "Suspicious PowerShell command", "High", "[]"])
                + "\n"
                + SEPARATOR.join(["positive-2", "Suspicious credential access", "Medium", "[]"])
                + "\n",
                encoding="utf-8",
            )

            report = await evaluate_secrl_alert_detection(root, scope="incident_5", ground_truth="security-incidents")

        self.assertEqual(report.ground_truth_alerts, 2)
        self.assertEqual(report.available_ground_truth_alerts, 2)
        self.assertEqual(report.true_positives, 2)
        self.assertEqual(report.false_negatives, 0)
        self.assertEqual(report.available_recall, 1.0)
        self.assertEqual(report.incident_recall_any, 1.0)
        self.assertEqual(report.incidents[0].detected_ground_truth_alerts, 2)

    def test_error_analysis_explains_false_positives_and_filtered_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["positive", "filtered", "other-incident"])])
                + "\n",
                encoding="utf-8",
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "Entities"]) + "\n"
                + SEPARATOR.join(["positive", "Suspicious PowerShell command", "High", "[]"])
                + "\n"
                + SEPARATOR.join(["filtered", "Routine notification", "Low", "[]"])
                + "\n"
                + SEPARATOR.join(["false-positive", "Suspicious PowerShell command", "High", "[]"])
                + "\n"
                + SEPARATOR.join(["other-incident", "Credential theft", "High", "[]"])
                + "\n",
                encoding="utf-8",
            )

            report = analyze_secrl_errors(root, scope="incident_5", ground_truth="security-incidents")

        self.assertEqual(report.false_negatives, 1)
        self.assertEqual(report.false_positives, 1)
        self.assertEqual(report.false_positives_with_security_incident_ref, 0)
        self.assertEqual(report.incident_analyses[0].missing_filtered, 1)
        self.assertEqual(report.incident_analyses[0].miss_reasons, {"not_interesting": 1})
        self.assertIn("dominant false-positive gate", " ".join(report.conclusions))

    def test_discovers_candidate_new_incident_without_known_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known-1"])])
                + "\n",
                encoding="utf-8",
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known-1", "Credential theft", "High", "2024-06-20 08:00:00+00:00", "[]"])
                + "\n"
                + SEPARATOR.join(
                    [
                        "new-1",
                        "Suspicious PowerShell command",
                        "High",
                        "2024-06-20 09:00:00+00:00",
                        json.dumps(
                            [
                                {"Type": "account", "UserPrincipalName": "attacker@example.com"},
                                {"Type": "host", "HostName": "workstation-9"},
                            ]
                        ),
                    ]
                )
                + "\n"
                + SEPARATOR.join(
                    [
                        "new-2",
                        "Suspicious WMI process creation",
                        "Medium",
                        "2024-06-20 09:30:00+00:00",
                        json.dumps(
                            [
                                {"Type": "account", "UserPrincipalName": "attacker@example.com"},
                                {"Type": "host", "HostName": "workstation-9"},
                            ]
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                min_alerts=2,
                min_score=0.55,
            )

        candidates = [incident for incident in report.incidents if incident.status == "candidate_new_incident"]
        self.assertEqual(report.candidate_new_incidents, 1)
        self.assertEqual(report.ground_truth_incidents, 1)
        self.assertEqual(report.detected_ground_truth_incidents, 0)
        self.assertEqual(report.missed_ground_truth_incidents, 1)
        self.assertEqual(report.incident_recall_any, 0.0)
        self.assertEqual(report.ground_truth_matches[0].matched_clusters, [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(set(candidates[0].alert_ids), {"new-1", "new-2"})
        self.assertEqual(candidates[0].known_ground_truth_incidents, [])
        self.assertEqual(candidates[0].raw_security_incident_refs, [])
        self.assertGreaterEqual(candidates[0].score, 0.55)

    def test_discovery_marks_cluster_with_security_incident_overlap_as_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known-1", "known-2"])])
                + "\n",
                encoding="utf-8",
            )
            entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "admin@example.com"},
                    {"Type": "host", "HostName": "server-1"},
                ]
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known-1", "Mimikatz credential theft tool", "High", "2024-06-20 10:00:00+00:00", entities])
                + "\n"
                + SEPARATOR.join(["known-2", "Suspicious WMI process creation", "Medium", "2024-06-20 10:20:00+00:00", entities])
                + "\n",
                encoding="utf-8",
            )

            report = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                min_alerts=2,
                min_score=0.55,
            )

        self.assertEqual(report.candidate_new_incidents, 0)
        self.assertEqual(report.known_overlap_clusters, 1)
        self.assertEqual(report.ground_truth_incidents, 1)
        self.assertEqual(report.detected_ground_truth_incidents, 1)
        self.assertEqual(report.missed_ground_truth_incidents, 0)
        self.assertEqual(report.incident_recall_any, 1.0)
        self.assertEqual(report.mean_available_alert_recall, 1.0)
        self.assertEqual(report.incidents[0].status, "known_incident_overlap")
        self.assertEqual(report.incidents[0].known_ground_truth_incidents, ["incident_5:5"])
        self.assertEqual(report.incidents[0].raw_security_incident_refs, ["incident_5:5"])
        self.assertEqual(report.ground_truth_matches[0].matched_clusters, [report.incidents[0].cluster_id])
        self.assertTrue(report.ground_truth_matches[0].all_available_recovered)

    def test_generic_ics_refinement_suppresses_unlabeled_ics_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known"])])
                + "\n",
                encoding="utf-8",
            )
            entities = json.dumps(
                [
                    {"Type": "host", "HostName": "plc-gateway"},
                    {"Type": "ip", "Address": "10.0.0.5"},
                ]
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known", "Credential theft", "High", "2024-06-20 08:00:00+00:00", "[]"])
                + "\n"
                + SEPARATOR.join(["ics-1", "New Activity Detected - CIP Class Command", "High", "2024-06-20 09:00:00+00:00", entities])
                + "\n"
                + SEPARATOR.join(["ics-2", "New Activity Detected - CIP Class Service Command", "High", "2024-06-20 09:10:00+00:00", entities])
                + "\n",
                encoding="utf-8",
            )

            baseline = discover_secrl_incidents(root, scope="incident_5", ground_truth="security-incidents")
            refined = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                refinements={SUPPRESS_GENERIC_ICS},
            )

        self.assertEqual(baseline.candidate_new_incidents, 1)
        self.assertEqual(refined.candidate_new_incidents, 0)

    def test_progression_refinement_suppresses_repeated_signin_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known"])])
                + "\n",
                encoding="utf-8",
            )
            entities = json.dumps([{"Type": "account", "UserPrincipalName": "user@example.com"}])
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known", "Credential theft", "High", "2024-06-20 08:00:00+00:00", "[]"])
                + "\n"
                + SEPARATOR.join(["signin-1", "Unfamiliar sign-in properties", "Medium", "2024-06-20 09:00:00+00:00", entities])
                + "\n"
                + SEPARATOR.join(["signin-2", "Unfamiliar sign-in properties", "Low", "2024-06-20 09:20:00+00:00", entities])
                + "\n",
                encoding="utf-8",
            )

            baseline = discover_secrl_incidents(root, scope="incident_5", ground_truth="security-incidents")
            refined = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                refinements={REQUIRE_PROGRESSION_OR_SEVERITY},
            )

        self.assertEqual(baseline.candidate_new_incidents, 1)
        self.assertEqual(refined.candidate_new_incidents, 0)

    def test_reduced_c2_weight_suppresses_c2_only_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known"])])
                + "\n",
                encoding="utf-8",
            )
            entities = json.dumps([{"Type": "host", "HostName": "controller-1"}])
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known", "Credential theft", "High", "2024-06-20 08:00:00+00:00", "[]"])
                + "\n"
                + SEPARATOR.join(["c2-1", "C2 beacon command and control activity", "Medium", "2024-06-20 09:00:00+00:00", entities])
                + "\n"
                + SEPARATOR.join(["c2-2", "C2 beacon command and control activity", "Medium", "2024-06-20 09:30:00+00:00", entities])
                + "\n",
                encoding="utf-8",
            )

            baseline = discover_secrl_incidents(root, scope="incident_5", ground_truth="security-incidents")
            refined = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                refinements={REDUCE_C2_WEIGHT},
            )

        self.assertEqual(baseline.candidate_new_incidents, 1)
        self.assertEqual(refined.candidate_new_incidents, 0)

    def test_attack_graph_topology_match_is_reported_for_known_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known-1", "known-2", "known-3"])])
                + "\n",
                encoding="utf-8",
            )
            entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "admin@example.com"},
                    {"Type": "host", "HostName": "server-1"},
                ]
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known-1", "Credential theft detected", "High", "2024-06-20 08:00:00+00:00", entities])
                + "\n"
                + SEPARATOR.join(["known-2", "Suspicious command execution via WMI", "Medium", "2024-06-20 08:10:00+00:00", entities])
                + "\n"
                + SEPARATOR.join(["known-3", "Lateral movement with privilege escalation", "Medium", "2024-06-20 08:20:00+00:00", entities])
                + "\n",
                encoding="utf-8",
            )

            report = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                min_alerts=2,
                min_score=0.55,
            )

        self.assertEqual(report.known_overlap_clusters, 1)
        self.assertEqual(report.incidents[0].attack_graph_pattern, "credential_lateral_privilege")
        self.assertGreaterEqual(report.incidents[0].attack_graph_score, 0.6)
        self.assertTrue(report.incidents[0].attack_graph_edges)
        self.assertIn("attack_graph=", " ".join(report.incidents[0].rationale))

    def test_leakage_audit_compares_normal_and_blind_labeling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known-1", "known-2"])])
                + "\n",
                encoding="utf-8",
            )
            known_entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "admin@example.com"},
                    {"Type": "host", "HostName": "server-1"},
                ]
            )
            new_entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "operator@example.com"},
                    {"Type": "host", "HostName": "server-2"},
                ]
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known-1", "Credential theft detected", "High", "2024-06-20 08:00:00+00:00", known_entities])
                + "\n"
                + SEPARATOR.join(["known-2", "Suspicious command execution via WMI", "Medium", "2024-06-20 08:10:00+00:00", known_entities])
                + "\n"
                + SEPARATOR.join(["new-1", "Credential theft detected", "High", "2024-06-20 09:00:00+00:00", new_entities])
                + "\n"
                + SEPARATOR.join(["new-2", "Suspicious command execution via WMI", "Medium", "2024-06-20 09:10:00+00:00", new_entities])
                + "\n",
                encoding="utf-8",
            )

            report = audit_secrl_discovery_leakage(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                min_alerts=2,
                min_score=0.55,
            )

        self.assertFalse(report.potential_leakage_detected)
        self.assertTrue(report.cluster_generation_stable)
        self.assertTrue(report.score_generation_stable)
        self.assertEqual(report.ground_truth_labeled_clusters, 1)
        self.assertEqual(report.normal_candidate_new_incidents, 1)
        self.assertEqual(report.blind_candidate_incidents, 2)
        self.assertEqual(report.posthoc_label_delta, 1)
        self.assertIn("Leakage Audit", report.to_markdown())

    def test_strict_entity_continuity_suppresses_weak_bridge_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known"])])
                + "\n",
                encoding="utf-8",
            )

            def host_pair(left: str, right: str) -> str:
                return json.dumps(
                    [
                        {"Type": "host", "HostName": left},
                        {"Type": "host", "HostName": right},
                    ]
                )

            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known", "Credential theft", "High", "2024-06-20 08:00:00+00:00", "[]"])
                + "\n"
                + SEPARATOR.join(["bridge-1", "Suspicious PowerShell command", "Medium", "2024-06-20 09:00:00+00:00", host_pair("a", "b")])
                + "\n"
                + SEPARATOR.join(["bridge-2", "Suspicious PowerShell command", "Medium", "2024-06-20 09:10:00+00:00", host_pair("b", "c")])
                + "\n"
                + SEPARATOR.join(["bridge-3", "Suspicious PowerShell command", "Medium", "2024-06-20 09:20:00+00:00", host_pair("c", "d")])
                + "\n"
                + SEPARATOR.join(["bridge-4", "Suspicious PowerShell command", "Medium", "2024-06-20 09:30:00+00:00", host_pair("d", "e")])
                + "\n",
                encoding="utf-8",
            )

            baseline = discover_secrl_incidents(root, scope="incident_5", ground_truth="security-incidents")
            refined = discover_secrl_incidents(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                refinements={STRICT_ENTITY_CONTINUITY},
            )

        self.assertEqual(baseline.candidate_new_incidents, 1)
        self.assertEqual(refined.candidate_new_incidents, 0)

    def test_discovery_ablation_reports_each_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known-1", "known-2"])])
                + "\n",
                encoding="utf-8",
            )
            known_entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "admin@example.com"},
                    {"Type": "host", "HostName": "server-1"},
                ]
            )
            noisy_entities = json.dumps([{"Type": "account", "UserPrincipalName": "user@example.com"}])
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known-1", "Mimikatz credential theft tool", "High", "2024-06-20 08:00:00+00:00", known_entities])
                + "\n"
                + SEPARATOR.join(["known-2", "Suspicious WMI process creation", "Medium", "2024-06-20 08:20:00+00:00", known_entities])
                + "\n"
                + SEPARATOR.join(["noise-1", "Unfamiliar sign-in properties", "Medium", "2024-06-20 09:00:00+00:00", noisy_entities])
                + "\n"
                + SEPARATOR.join(["noise-2", "Unfamiliar sign-in properties", "Low", "2024-06-20 09:10:00+00:00", noisy_entities])
                + "\n",
                encoding="utf-8",
            )

            report = run_secrl_discovery_ablation(root, scope="incident_5", ground_truth="security-incidents")

        variants = {row.variant for row in report.rows}
        self.assertIn("baseline", variants)
        self.assertIn(f"only_{REQUIRE_PROGRESSION_OR_SEVERITY}", variants)
        self.assertIn("all_refinements", variants)
        baseline = next(row for row in report.rows if row.variant == "baseline")
        all_refinements = next(row for row in report.rows if row.variant == "all_refinements")
        self.assertGreater(baseline.candidate_new_incidents, all_refinements.candidate_new_incidents)
        self.assertEqual(all_refinements.incident_recall_any, 1.0)

    def test_baseline_comparison_reports_reference_and_oracle_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incident_dir = root / "incidents" / "incident_5"
            incident_dir.mkdir(parents=True)
            (incident_dir / "SecurityIncident.csv").write_text(
                SEPARATOR.join(["IncidentNumber", "AlertIds"]) + "\n"
                + SEPARATOR.join(["5", json.dumps(["known-1", "known-2"])])
                + "\n",
                encoding="utf-8",
            )
            known_entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "admin@example.com"},
                    {"Type": "host", "HostName": "server-1"},
                ]
            )
            noisy_entities = json.dumps(
                [
                    {"Type": "account", "UserPrincipalName": "other@example.com"},
                    {"Type": "host", "HostName": "server-2"},
                ]
            )
            (incident_dir / "SecurityAlert.csv").write_text(
                SEPARATOR.join(["SystemAlertId", "AlertName", "AlertSeverity", "StartTime", "Entities"]) + "\n"
                + SEPARATOR.join(["known-1", "Mimikatz credential theft tool", "High", "2024-06-20 08:00:00+00:00", known_entities])
                + "\n"
                + SEPARATOR.join(["known-2", "Suspicious WMI process creation", "Medium", "2024-06-20 08:20:00+00:00", known_entities])
                + "\n"
                + SEPARATOR.join(["noise-1", "Suspicious PowerShell command", "High", "2024-06-20 09:00:00+00:00", noisy_entities])
                + "\n"
                + SEPARATOR.join(["noise-2", "Suspicious PowerShell command", "Medium", "2024-06-20 09:15:00+00:00", noisy_entities])
                + "\n",
                encoding="utf-8",
            )

            report = compare_secrl_discovery_baselines(
                root,
                scope="incident_5",
                ground_truth="security-incidents",
                min_alerts=2,
                min_score=0.55,
            )

        rows = {row.baseline: row for row in report.rows}
        self.assertIn("eacs_refined", rows)
        self.assertIn("high_severity_only", rows)
        self.assertIn("graph_oracle", rows)
        self.assertTrue(rows["graph_oracle"].uses_ground_truth)
        self.assertFalse(rows["eacs_refined"].uses_ground_truth)
        self.assertEqual(rows["graph_oracle"].incident_recall_any, 1.0)
        self.assertEqual(rows["eacs_refined"].incident_recall_any, 1.0)
        self.assertGreater(rows["attack_keyword_only"].selected_alerts, 0)
        self.assertIn("Baseline Table", report.to_markdown())


if __name__ == "__main__":
    unittest.main()
