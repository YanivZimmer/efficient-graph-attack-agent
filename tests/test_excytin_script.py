from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_excytin_bench.py"
SPEC = importlib.util.spec_from_file_location("evaluate_excytin_bench", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
evaluate_excytin_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_excytin_bench)


class ExcytinBenchScriptTests(unittest.TestCase):
    def test_metadata_output_path_includes_split_and_limit(self) -> None:
        path = evaluate_excytin_bench.metadata_output_path(Path("reports/out"), "test", 25)

        self.assertEqual(path, Path("reports/out/excytin_metadata_test_limit25.json"))

    def test_qa_output_path_includes_configuration(self) -> None:
        path = evaluate_excytin_bench.qa_output_path(
            Path("reports/out"),
            "train",
            "eacs_retrieved",
            "extractive",
            None,
        )

        self.assertEqual(path, Path("reports/out/excytin_qa_eacs_retrieved_extractive_train.json"))

    def test_defaults_use_fair_qa_baseline(self) -> None:
        args = evaluate_excytin_bench.parse_args([])

        self.assertEqual(evaluate_excytin_bench.qa_context_sources(args), ["eacs_retrieved"])
        self.assertEqual(evaluate_excytin_bench.qa_answer_modes(args), ["extractive"])


if __name__ == "__main__":
    unittest.main()
