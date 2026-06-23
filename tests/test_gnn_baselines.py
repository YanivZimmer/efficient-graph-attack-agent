"""Tests for literature GNN baseline models."""

from __future__ import annotations

import unittest

from data.graph_builder import build_graph_from_records
from models.baselines.anomal_e import AnomalE
from models.baselines.gnn_ids import GNNIDS
from models.baselines.graph_ids import GraphIDS
from models.baselines.graph_utils import build_alert_homogeneous_graph
from models.baselines.registry import BASELINE_MODELS, build_model
from tests.test_gnn_pipeline import SAMPLE_RECORDS
from training.baseline_trainer import train_baseline


class GNNBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = build_graph_from_records(SAMPLE_RECORDS)

    def test_registry_contains_literature_baselines(self) -> None:
        self.assertIn("hgat", BASELINE_MODELS)
        self.assertIn("gnn_ids", BASELINE_MODELS)
        self.assertIn("graph_ids", BASELINE_MODELS)
        self.assertIn("anomal_e", BASELINE_MODELS)

    def test_homogeneous_projection_has_edges(self) -> None:
        homo = build_alert_homogeneous_graph(self.artifacts.data)
        self.assertEqual(homo.num_nodes, len(SAMPLE_RECORDS))
        self.assertGreater(homo.edge_index.size(1), 0)

    def test_models_emit_logits_and_embeddings(self) -> None:
        for model_name in ("gnn_ids", "graph_ids", "anomal_e", "hgat"):
            model = build_model(model_name, self.artifacts)
            logits = model(self.artifacts.data)
            embeddings = model.encode(self.artifacts.data)
            self.assertEqual(logits.shape[0], len(SAMPLE_RECORDS))
            self.assertEqual(embeddings.shape[0], len(SAMPLE_RECORDS))

    def test_train_gnn_ids_baseline(self) -> None:
        result = train_baseline("gnn_ids", self.artifacts, epochs=3, pretrain_epochs=0)
        self.assertEqual(result.predictions.shape[0], len(SAMPLE_RECORDS))
        self.assertIn("auc", result.metrics)

    def test_train_graph_ids_baseline(self) -> None:
        result = train_baseline("graph_ids", self.artifacts, epochs=2, pretrain_epochs=2)
        self.assertEqual(result.probabilities.shape[0], len(SAMPLE_RECORDS))

    def test_train_anomal_e_baseline(self) -> None:
        result = train_baseline("anomal_e", self.artifacts, epochs=2, pretrain_epochs=2)
        self.assertEqual(result.probabilities.shape[0], len(SAMPLE_RECORDS))


if __name__ == "__main__":
    unittest.main()
