from __future__ import annotations

import asyncio

from .adapters import InMemoryLogStore, MockAlertStream
from .agents import AgentOrchestrator
from .graph import GraphController, InMemoryGraphStore
from .sketch import GraphSketchingFilter, StreamProcessor


async def main() -> None:
    graph_store = InMemoryGraphStore()
    processor = StreamProcessor(GraphSketchingFilter(), graph_store)
    stats = await processor.process(MockAlertStream())

    graph = GraphController(graph_store)
    log_store = InMemoryLogStore({"alert-0": {"source": "demo", "raw": "hydrated log"}})
    story = await AgentOrchestrator(graph, log_store=log_store).investigate("alert-0")

    print(f"processed={stats.processed} stored={stats.stored} ratio={stats.stored_ratio:.2%}")
    print(f"throughput={stats.alerts_per_second:.0f} alerts/sec")
    print(f"confidence={story.confidence_score:.2f}")
    print(story.storyline)


if __name__ == "__main__":
    asyncio.run(main())
