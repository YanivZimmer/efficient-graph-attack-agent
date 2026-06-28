from __future__ import annotations

import unittest

from eacs.adapters import InMemoryLogStore, MockAlertStream
from eacs.agents import AgentOrchestrator
from eacs.graph import GraphController, InMemoryGraphStore
from eacs.models import Alert, Entity, EntityType
from eacs.sketch import GraphSketchingFilter, StreamProcessor


class CoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_stream_stores_interesting_ten_percent(self) -> None:
        graph = InMemoryGraphStore()
        processor = StreamProcessor(GraphSketchingFilter(), graph)

        stats = await processor.process(MockAlertStream(count=10_000, interesting_every=10))

        self.assertEqual(stats.processed, 10_000)
        self.assertEqual(stats.stored, 1_000)
        self.assertLess(stats.stored_ratio, 0.11)
        self.assertGreater(stats.alerts_per_second, 1_000)

    async def test_common_relationship_filtered_unless_high_severity(self) -> None:
        source = Entity(type=EntityType.HOST, value="workstation-1")
        target = Entity(type=EntityType.SERVICE, value="dns")
        low = Alert(id="low", source=source, target=target, kind="dns_lookup", severity=1)
        high = Alert(id="high", source=source, target=target, kind="dns_lookup", severity=9)

        sketch_filter = GraphSketchingFilter()
        sketch_filter.seed_baseline(low.relationship_key, 10_001)

        self.assertFalse(sketch_filter.evaluate(low).store)
        self.assertTrue(sketch_filter.evaluate(high).store)

    async def test_graph_controller_builds_two_hop_subgraph(self) -> None:
        graph = InMemoryGraphStore()
        user = Entity(type=EntityType.USER, value="admin")
        host = Entity(type=EntityType.HOST, value="server-1")
        await graph.upsert_alert(Alert(id="a1", source=user, target=host, kind="failed_login_burst", severity=6))
        await graph.upsert_alert(Alert(id="a2", source=user, target=host, kind="privilege_escalation", severity=7))

        subgraph = await GraphController(graph).build_subgraph("a1")

        self.assertEqual({alert.id for alert in subgraph.alerts}, {"a1", "a2"})
        self.assertEqual({entity.id for entity in subgraph.entities}, {user.id, host.id})

    async def test_orchestrator_hydrates_only_after_validation_threshold(self) -> None:
        graph_store = InMemoryGraphStore()
        user = Entity(type=EntityType.USER, value="admin")
        host = Entity(type=EntityType.HOST, value="server-1")
        root = Alert(
            id="root",
            source=user,
            target=host,
            kind="lateral_movement",
            action="remote_login",
            severity=9,
            tags={"lateral_movement"},
        )
        await graph_store.upsert_alert(root)
        await graph_store.upsert_alert(
            Alert(id="related", source=user, target=host, kind="privilege_escalation", severity=7)
        )

        story = await AgentOrchestrator(
            GraphController(graph_store),
            log_store=InMemoryLogStore({"root": {"raw": "root log"}, "related": {"raw": "related log"}}),
        ).investigate("root")

        self.assertGreater(story.confidence_score, 0.7)
        self.assertEqual(len(story.hydrated_logs), 2)

    async def test_orchestrator_skips_hydration_below_threshold(self) -> None:
        graph_store = InMemoryGraphStore()
        alert = Alert(
            id="low",
            source=Entity(type=EntityType.HOST, value="workstation-1"),
            target=Entity(type=EntityType.SERVICE, value="dns"),
            kind="dns_lookup",
            severity=1,
        )
        await graph_store.upsert_alert(alert)

        story = await AgentOrchestrator(
            GraphController(graph_store),
            log_store=InMemoryLogStore({"low": {"raw": "should not hydrate"}}),
        ).investigate("low")

        self.assertLessEqual(story.confidence_score, 0.7)
        self.assertEqual(story.hydrated_logs, [])


if __name__ == "__main__":
    unittest.main()
