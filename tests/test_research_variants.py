from __future__ import annotations

import unittest

import torch

from data.graph_builder import build_graph_from_records
from models.baselines.graph_utils import build_semantic_alert_graph, build_sparse_causal_alert_graph
from models.research_variants import (
    DifferentiableClusterHGAT,
    MultiViewHGAT,
    MultiViewHGATv2,
    PrototypeMemoryHGAT,
    TemporalCausalHGATv2,
)
from tests.test_gnn_pipeline import SAMPLE_RECORDS
from training.research_variant_trainer import augment_time_harmonics
from training.trainer import _entity_counts


class ResearchVariantsTests(unittest.TestCase):
    def test_differentiable_cluster_hgat_forward(self) -> None:
        artifacts = build_graph_from_records(
            SAMPLE_RECORDS,
            include_alert_alert_edges=True,
            alert_link_hours=48.0,
            max_alert_neighbors_per_relation=4,
        )
        augment_time_harmonics(artifacts, num_frequencies=1)
        model = DifferentiableClusterHGAT(
            metadata=artifacts.data.metadata(),
            alert_in_channels=int(artifacts.data["alert"].x.size(-1)),
            entity_counts=_entity_counts(artifacts),
            num_slots=4,
        )
        embeddings = model.encode(artifacts.data)
        logits = model(artifacts.data)
        assignments = model.hard_assignments(embeddings)
        self.assertEqual(embeddings.size(0), len(artifacts.alert_ids))
        self.assertEqual(logits.size(0), len(artifacts.alert_ids))
        self.assertEqual(assignments.size(0), len(artifacts.alert_ids))

    def test_prototype_memory_hgat_forward(self) -> None:
        artifacts = build_graph_from_records(
            SAMPLE_RECORDS,
            include_alert_alert_edges=True,
            alert_link_hours=48.0,
            max_alert_neighbors_per_relation=4,
        )
        augment_time_harmonics(artifacts, num_frequencies=1)
        model = PrototypeMemoryHGAT(
            metadata=artifacts.data.metadata(),
            alert_in_channels=int(artifacts.data["alert"].x.size(-1)),
            entity_counts=_entity_counts(artifacts),
            num_prototypes=4,
        )
        embeddings = model.encode(artifacts.data)
        assignments = model.hard_assignments(embeddings)
        self.assertEqual(assignments.size(0), len(artifacts.alert_ids))

    def test_multiview_hgat_forward(self) -> None:
        artifacts = build_graph_from_records(SAMPLE_RECORDS)
        augment_time_harmonics(artifacts, num_frequencies=1)
        temporal_graph = build_sparse_causal_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=48.0,
            max_neighbors_per_alert=4,
        )
        semantic_graph = build_semantic_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=48.0,
            max_neighbors_per_alert=4,
        )
        model = MultiViewHGAT(
            metadata=artifacts.data.metadata(),
            alert_in_channels=int(artifacts.data["alert"].x.size(-1)),
            entity_counts=_entity_counts(artifacts),
        )
        embeddings, views = model.encode(
            artifacts.data,
            temporal_graph=temporal_graph,
            semantic_graph=semantic_graph,
        )
        logits = model(
            artifacts.data,
            temporal_graph=temporal_graph,
            semantic_graph=semantic_graph,
        )
        self.assertEqual(embeddings.size(0), len(artifacts.alert_ids))
        self.assertEqual(logits.size(0), len(artifacts.alert_ids))
        self.assertIn("gates", views)
        self.assertTrue(torch.allclose(views["gates"].sum(dim=-1), torch.ones(len(artifacts.alert_ids)), atol=1e-5))

    def test_temporal_causal_hgat_v2_forward(self) -> None:
        artifacts = build_graph_from_records(
            SAMPLE_RECORDS,
            include_alert_alert_edges=True,
            alert_link_hours=48.0,
            max_alert_neighbors_per_relation=4,
        )
        augment_time_harmonics(artifacts, num_frequencies=1)
        model = TemporalCausalHGATv2(
            metadata=artifacts.data.metadata(),
            alert_in_channels=int(artifacts.data["alert"].x.size(-1)),
            entity_counts=_entity_counts(artifacts),
        )
        embeddings = model.encode(artifacts.data)
        logits = model.classify_embeddings(embeddings)
        time_scores = model.time_scores(embeddings)
        self.assertEqual(embeddings.size(0), len(artifacts.alert_ids))
        self.assertEqual(logits.size(0), len(artifacts.alert_ids))
        self.assertEqual(time_scores.size(0), len(artifacts.alert_ids))

    def test_multiview_hgat_v2_forward(self) -> None:
        artifacts = build_graph_from_records(
            SAMPLE_RECORDS,
            include_alert_alert_edges=True,
            alert_link_hours=48.0,
            max_alert_neighbors_per_relation=4,
        )
        augment_time_harmonics(artifacts, num_frequencies=1)
        temporal_graph = build_sparse_causal_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=48.0,
            max_neighbors_per_alert=4,
        )
        semantic_graph = build_semantic_alert_graph(
            artifacts.data,
            artifacts.alert_records,
            max_gap_hours=48.0,
            max_neighbors_per_alert=4,
        )
        model = MultiViewHGATv2(
            metadata=artifacts.data.metadata(),
            alert_in_channels=int(artifacts.data["alert"].x.size(-1)),
            entity_counts=_entity_counts(artifacts),
        )
        embeddings, views = model.encode(
            artifacts.data,
            temporal_graph=temporal_graph,
            semantic_graph=semantic_graph,
        )
        logits = model(
            artifacts.data,
            temporal_graph=temporal_graph,
            semantic_graph=semantic_graph,
        )
        self.assertEqual(embeddings.size(0), len(artifacts.alert_ids))
        self.assertEqual(logits.size(0), len(artifacts.alert_ids))
        self.assertIn("projections", views)
        self.assertTrue(torch.allclose(views["gates"].sum(dim=-1), torch.ones(len(artifacts.alert_ids)), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
