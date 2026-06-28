from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path

from eacs.excytin import (
    ExcytinQuestion,
    alert_id,
    answer_question_from_context,
    alerts_from_question,
    evaluate_excytin_qa,
    evaluate_excytin_questions,
    extract_answer_from_context,
    load_secrl_questions,
)


class ExcytinTests(unittest.IsolatedAsyncioTestCase):
    def test_alert_adapter_builds_chain_entities(self) -> None:
        question = ExcytinQuestion(
            row_idx=7,
            context="PowerShell was detected.",
            question="What process ran PowerShell?",
            answer="powershell.exe",
            solution=["A PowerShell command ran."],
            start_alert=1,
            end_alert=3,
            shortest_alert_path=[1, 2, 3],
        )

        alerts = alerts_from_question(question)

        self.assertEqual([alert.id for alert in alerts], [alert_id(7, 1), alert_id(7, 2), alert_id(7, 3)])
        self.assertEqual(alerts[0].target, alerts[1].source)
        self.assertEqual(alerts[1].target, alerts[2].source)
        self.assertEqual(alerts[0].kind, "command_and_control")

    async def test_evaluation_reports_path_recall_for_two_hop_graph(self) -> None:
        question = ExcytinQuestion(
            row_idx=1,
            context="Suspicious sign-in.",
            question="What IP?",
            answer="10.0.0.1",
            solution=["A suspicious sign-in occurred."],
            start_alert=10,
            end_alert=12,
            shortest_alert_path=[10, 11, 12],
        )

        report = await evaluate_excytin_questions([question], split="test")

        self.assertEqual(report.rows_evaluated, 1)
        self.assertEqual(report.total_ingested_alerts, 3)
        self.assertEqual(report.rows[0].path_alert_recall, 2 / 3)
        self.assertFalse(report.rows[0].end_alert_recalled)

    def test_extract_answer_from_context_prefers_question_type(self) -> None:
        answer = extract_answer_from_context(
            "What is the IP address used by the attacker?",
            "The user `admin@example.com` connected from `198.43.121.209`.",
        )

        self.assertEqual(answer, "198.43.121.209")

    def test_gold_if_present_answer_mode_measures_context_availability(self) -> None:
        question = ExcytinQuestion(
            row_idx=3,
            context="",
            question="What IP address was used?",
            answer="198.43.121.209",
            solution=[],
            start_alert=1,
            end_alert=1,
        )

        self.assertEqual(
            answer_question_from_context(question, "Observed `198.43.121.209`.", answer_mode="gold_if_present"),
            "198.43.121.209",
        )
        self.assertEqual(answer_question_from_context(question, "No matching IP.", answer_mode="gold_if_present"), "")

    def test_question_accepts_structured_context(self) -> None:
        question = ExcytinQuestion(
            row_idx=4,
            context={"security_report": {"incident": "Password spray"}},
            question="What happened?",
            answer="Password spray",
            solution=[],
            start_alert=1,
            end_alert=1,
        )

        self.assertIn("security_report", question.context)
        self.assertIn("Password spray", question.context)

    async def test_qa_evaluation_scores_extractive_answer(self) -> None:
        question = ExcytinQuestion(
            row_idx=3,
            context="The attacker used IP `198.43.121.209`.",
            question="What IP address was used?",
            answer="198.43.121.209",
            solution=["The IP address was `198.43.121.209`."],
            start_alert=1,
            end_alert=1,
            shortest_alert_path=[1],
        )

        report = await evaluate_excytin_qa(
            [("incident_1", question)],
            split="test",
            context_source="oracle_metadata",
            answer_mode="gold_if_present",
        )

        self.assertEqual(report.rows_evaluated, 1)
        self.assertEqual(report.exact_match_rate, 1.0)
        self.assertEqual(report.contains_answer_rate, 1.0)

    def test_loads_secrl_question_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            question_dir = root / "secgym" / "questions" / "o1" / "test"
            question_dir.mkdir(parents=True)
            (question_dir / "incident_5_qa_incident_o1-ga_c42.json").write_text(
                json.dumps(
                    [
                        {
                            "context": "Context",
                            "question": "Question?",
                            "answer": "Answer",
                            "solution": ["Solution"],
                            "start_alert": 1,
                            "end_alert": 1,
                            "shortest_alert_path": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            loaded = load_secrl_questions(root, split="test", question_set="o1")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0], "incident_5")


if __name__ == "__main__":
    unittest.main()
