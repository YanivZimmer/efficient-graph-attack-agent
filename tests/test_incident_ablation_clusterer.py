from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from clustering.incident_ablation_clusterer import ClusterAblationStrategy, cluster_with_strategy
from data.graph_builder import AlertRecord, build_graph_from_records
from tests.test_gnn_pipeline import SAMPLE_RECORDS


class IncidentAblationClustererTests(unittest.TestCase):
    def test_temporal_splitter_fragments_dbscan_cluster(self) -> None:
        artifacts = build_graph_from_records(SAMPLE_RECORDS)
        embeddings = np.ones((len(artifacts.alert_ids), 4), dtype=np.float32)
        probabilities = np.ones(len(artifacts.alert_ids), dtype=np.float32)
        predictions = np.ones(len(artifacts.alert_ids), dtype=np.int64)
        strategy = ClusterAblationStrategy(
            name="dbscan_split_time_12h",
            split_time_gap_hours=12.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clusters = cluster_with_strategy(
                strategy=strategy,
                embeddings=embeddings,
                alert_ids=artifacts.alert_ids,
                records=artifacts.alert_records,
                predictions=predictions,
                probabilities=probabilities,
                output_path=Path(tmpdir) / "clusters.jsonl",
                threshold=0.5,
                selection_mode="probability",
            )

        incident_clusters = [cluster for cluster in clusters if int(cluster["incident_id"]) >= 0]
        self.assertGreater(len(incident_clusters), 1)

    def test_graph_entity_components_use_shared_entities(self) -> None:
        artifacts = build_graph_from_records(SAMPLE_RECORDS)
        embeddings = np.eye(len(artifacts.alert_ids), dtype=np.float32)
        probabilities = np.ones(len(artifacts.alert_ids), dtype=np.float32)
        predictions = np.ones(len(artifacts.alert_ids), dtype=np.int64)
        strategy = ClusterAblationStrategy(
            name="graph_entity_48h",
            base="graph",
            relation_policy="entity",
            max_gap_hours=48.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clusters = cluster_with_strategy(
                strategy=strategy,
                embeddings=embeddings,
                alert_ids=artifacts.alert_ids,
                records=artifacts.alert_records,
                predictions=predictions,
                probabilities=probabilities,
                output_path=Path(tmpdir) / "clusters.jsonl",
                threshold=0.5,
                selection_mode="probability",
            )

        non_noise_alerts = {
            alert_id
            for cluster in clusters
            if int(cluster["incident_id"]) >= 0
            for alert_id in cluster["alert_ids"]
        }
        self.assertIn("a1", non_noise_alerts)
        self.assertIn("a2", non_noise_alerts)

    def test_temporal_only_adaptive_quantile_clusters_without_embeddings(self) -> None:
        artifacts = build_graph_from_records(SAMPLE_RECORDS)
        embeddings = np.eye(len(artifacts.alert_ids), dtype=np.float32)
        probabilities = np.ones(len(artifacts.alert_ids), dtype=np.float32)
        predictions = np.ones(len(artifacts.alert_ids), dtype=np.int64)
        strategy = ClusterAblationStrategy(
            name="temporal_only_adaptive_q50",
            base="temporal",
            split_time_gap_quantile=0.50,
            split_time_gap_min_hours=0.25,
            split_time_gap_max_hours=72.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clusters = cluster_with_strategy(
                strategy=strategy,
                embeddings=embeddings,
                alert_ids=artifacts.alert_ids,
                records=artifacts.alert_records,
                predictions=predictions,
                probabilities=probabilities,
                output_path=Path(tmpdir) / "clusters.jsonl",
                threshold=0.5,
                selection_mode="probability",
            )

        incident_clusters = [cluster for cluster in clusters if int(cluster["incident_id"]) >= 0]
        self.assertGreaterEqual(len(incident_clusters), 2)

    def test_bayesian_blocks_clusters_event_stream(self) -> None:
        artifacts = build_graph_from_records(SAMPLE_RECORDS)
        embeddings = np.eye(len(artifacts.alert_ids), dtype=np.float32)
        probabilities = np.ones(len(artifacts.alert_ids), dtype=np.float32)
        predictions = np.ones(len(artifacts.alert_ids), dtype=np.int64)
        strategy = ClusterAblationStrategy(
            name="bayesian_blocks_p1",
            base="bayesian_blocks",
            bayesian_blocks_prior=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clusters = cluster_with_strategy(
                strategy=strategy,
                embeddings=embeddings,
                alert_ids=artifacts.alert_ids,
                records=artifacts.alert_records,
                predictions=predictions,
                probabilities=probabilities,
                output_path=Path(tmpdir) / "clusters.jsonl",
                threshold=0.5,
                selection_mode="probability",
            )

        self.assertTrue(clusters)
        clustered_ids = {
            alert_id
            for cluster in clusters
            for alert_id in cluster["alert_ids"]
        }
        self.assertEqual(clustered_ids, set(artifacts.alert_ids))

    def test_macro_elbow_estimator_ignores_microburst_gaps(self) -> None:
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        gap_hours = [0.5, 0.6, 1.0, 1.5, 2.5, 4.8, 0.5, 14.0, 0.5]
        timestamps = [base_time]
        for gap in gap_hours:
            timestamps.append(timestamps[-1] + timedelta(hours=gap))
        records = [
            AlertRecord(
                alert_id=f"a{index}",
                label=1,
                timestamp=timestamp,
                entities={"host": "host-a"},
            )
            for index, timestamp in enumerate(timestamps)
        ]
        alert_ids = [record.alert_id for record in records]
        embeddings = np.ones((len(records), 4), dtype=np.float32)
        probabilities = np.ones(len(records), dtype=np.float32)
        predictions = np.ones(len(records), dtype=np.int64)
        strategy = ClusterAblationStrategy(
            name="temporal_only_macro_elbow",
            base="temporal",
            split_time_gap_estimator="macro_elbow",
            split_time_gap_min_hours=0.25,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clusters = cluster_with_strategy(
                strategy=strategy,
                embeddings=embeddings,
                alert_ids=alert_ids,
                records=records,
                predictions=predictions,
                probabilities=probabilities,
                output_path=Path(tmpdir) / "clusters.jsonl",
                threshold=0.5,
                selection_mode="probability",
            )

        incident_sizes = [
            len(cluster["alert_ids"])
            for cluster in clusters
            if int(cluster["incident_id"]) >= 0
        ]
        self.assertEqual(incident_sizes, [6, 2, 2])


if __name__ == "__main__":
    unittest.main()
