from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clustering.incident_clusterer import cluster_incidents


class IncidentClustererTests(unittest.TestCase):
    def test_probability_selection_mode_respects_threshold(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        alert_ids = ["a", "b", "c"]
        probabilities = np.array([0.95, 0.92, 0.40], dtype=np.float32)
        predictions = np.array([1, 1, 1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.jsonl"
            clusters = cluster_incidents(
                embeddings=embeddings,
                alert_ids=alert_ids,
                predictions=predictions,
                probabilities=probabilities,
                output_path=output_path,
                threshold=0.9,
                eps=0.2,
                min_samples=2,
                selection_mode="probability",
            )
            self.assertEqual(len(clusters), 1)
            self.assertEqual(clusters[0]["alert_ids"], ["a", "b"])
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(written[0]["alert_ids"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
