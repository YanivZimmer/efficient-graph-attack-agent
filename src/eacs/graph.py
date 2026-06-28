from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .models import Alert, Entity, GraphEdge, Subgraph
from .ports import GraphStore


class InMemoryGraphStore(GraphStore):
    """Repository used for tests, demos, and local development."""

    def __init__(self) -> None:
        self._alerts: dict[str, Alert] = {}
        self._entities: dict[str, Entity] = {}
        self._entity_alerts: dict[str, set[str]] = defaultdict(set)

    @property
    def alert_ids(self) -> set[str]:
        return set(self._alerts)

    async def upsert_alert(self, alert: Alert) -> None:
        self._alerts[alert.id] = alert
        for entity in alert.entities:
            self._entities[entity.id] = entity
            self._entity_alerts[entity.id].add(alert.id)

    async def fetch_alert(self, alert_id: str) -> Optional[Alert]:
        return self._alerts.get(alert_id)

    async def two_hop_neighborhood(self, alert_id: str) -> Subgraph:
        root = self._alerts.get(alert_id)
        if root is None:
            return Subgraph(root_alert_id=alert_id)

        related_ids: set[str] = {root.id}
        for entity in root.entities:
            related_ids.update(self._entity_alerts.get(entity.id, set()))

        alerts = [self._alerts[id_] for id_ in sorted(related_ids)]
        entities: dict[str, Entity] = {}
        edges: list[GraphEdge] = []
        for alert in alerts:
            entities[alert.source.id] = alert.source
            edges.append(GraphEdge(alert_id=alert.id, entity_id=alert.source.id, role="source"))
            if alert.target:
                entities[alert.target.id] = alert.target
                edges.append(GraphEdge(alert_id=alert.id, entity_id=alert.target.id, role="target"))

        return Subgraph(
            root_alert_id=alert_id,
            alerts=alerts,
            entities=[entities[id_] for id_ in sorted(entities)],
            edges=edges,
        )


class GraphController:
    """Thin domain facade over the graph repository."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    async def store_alert(self, alert: Alert) -> None:
        await self.store.upsert_alert(alert)

    async def fetch_alert(self, alert_id: str) -> Optional[Alert]:
        return await self.store.fetch_alert(alert_id)

    async def build_subgraph(self, alert_id: str) -> Subgraph:
        return await self.store.two_hop_neighborhood(alert_id)
