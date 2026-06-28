from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Optional

import httpx

from .models import Alert, Entity, EntityType, GraphEdge, Subgraph
from .ports import AlertStream, GraphStore, LLMProvider, LogStore


class OpenAIResponsesProvider(LLMProvider):
    """HTTPX-based OpenAI Responses adapter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/responses",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": prompt},
            )
            response.raise_for_status()
            payload = response.json()
        return self._extract_text(payload)

    def _extract_text(self, payload: Mapping[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        chunks: list[str] = []
        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)


class Neo4jGraphStore(GraphStore):
    """Optional Neo4j repository. Importing this class does not require neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install the neo4j extra to use Neo4jGraphStore") from exc
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    async def close(self) -> None:
        await self._driver.close()

    async def upsert_alert(self, alert: Alert) -> None:
        query = """
        MERGE (a:Alert {id: $alert.id})
        SET a.kind = $alert.kind,
            a.action = $alert.action,
            a.severity = $alert.severity,
            a.timestamp = $alert.timestamp,
            a.raw_json = $alert.raw_json,
            a.tags = $alert.tags
        WITH a
        UNWIND $entities AS item
        MERGE (e:Entity {id: item.id})
        SET e.type = item.type, e.value = item.value
        MERGE (a)-[r:MENTIONS {role: item.role}]->(e)
        """
        params = {
            "alert": {
                "id": alert.id,
                "kind": alert.kind,
                "action": alert.action,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "raw_json": json.dumps(alert.raw, sort_keys=True),
                "tags": sorted(alert.tags),
            },
            "entities": self._entity_params(alert),
        }
        async with self._driver.session(database=self._database) as session:
            await session.run(query, params)

    async def fetch_alert(self, alert_id: str) -> Optional[Alert]:
        query = """
        MATCH (a:Alert {id: $alert_id})
        OPTIONAL MATCH (a)-[r:MENTIONS]->(e:Entity)
        RETURN a, collect({entity: e, role: r.role}) AS refs
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, alert_id=alert_id)
            record = await result.single()
        if record is None:
            return None
        return self._record_to_alert(record["a"], record["refs"])

    async def two_hop_neighborhood(self, alert_id: str) -> Subgraph:
        query = """
        MATCH (:Alert {id: $alert_id})-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(a:Alert)
        OPTIONAL MATCH (a)-[r:MENTIONS]->(e:Entity)
        RETURN a, collect({entity: e, role: r.role}) AS refs
        ORDER BY a.id
        """
        alerts: list[Alert] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, alert_id=alert_id)
            async for record in result:
                alerts.append(self._record_to_alert(record["a"], record["refs"]))

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

    def _entity_params(self, alert: Alert) -> list[dict[str, str]]:
        items = [
            {
                "id": alert.source.id,
                "type": alert.source.type.value,
                "value": alert.source.value,
                "role": "source",
            }
        ]
        if alert.target:
            items.append(
                {
                    "id": alert.target.id,
                    "type": alert.target.type.value,
                    "value": alert.target.value,
                    "role": "target",
                }
            )
        return items

    def _record_to_alert(self, node: Any, refs: Sequence[Mapping[str, Any]]) -> Alert:
        source = None
        target = None
        for ref in refs:
            entity_node = ref.get("entity")
            if entity_node is None:
                continue
            entity = Entity(type=EntityType(entity_node["type"]), value=entity_node["value"])
            if ref.get("role") == "source":
                source = entity
            elif ref.get("role") == "target":
                target = entity
        if source is None:
            raise ValueError(f"alert {node['id']} has no source entity")
        return Alert(
            id=node["id"],
            source=source,
            target=target,
            kind=node["kind"],
            action=node["action"],
            severity=node["severity"],
            raw=json.loads(node.get("raw_json") or "{}"),
            tags=set(node.get("tags") or []),
        )


class InMemoryLogStore(LogStore):
    def __init__(self, logs_by_alert_id: Mapping[str, dict[str, Any]]) -> None:
        self.logs_by_alert_id = dict(logs_by_alert_id)

    async def fetch_logs(self, alert_ids: Sequence[str]) -> list[dict[str, Any]]:
        return [self.logs_by_alert_id[id_] for id_ in alert_ids if id_ in self.logs_by_alert_id]


class MockAlertStream(AlertStream):
    """Deterministic stream with about 10 percent attack-topology alerts."""

    def __init__(self, count: int = 10_000, interesting_every: int = 10) -> None:
        self.count = count
        self.interesting_every = interesting_every

    async def __aiter__(self) -> AsyncIterator[Alert]:
        benign_source = Entity(type=EntityType.HOST, value="workstation-1")
        benign_target = Entity(type=EntityType.SERVICE, value="dns")
        for idx in range(self.count):
            if idx % self.interesting_every == 0:
                yield Alert(
                    id=f"alert-{idx}",
                    source=Entity(type=EntityType.USER, value=f"user-{idx % 50}"),
                    target=Entity(type=EntityType.HOST, value=f"server-{idx % 20}"),
                    kind="lateral_movement",
                    action="remote_login",
                    severity=6,
                    tags={"lateral_movement"},
                    raw={"idx": idx, "event": "remote_login"},
                )
            else:
                yield Alert(
                    id=f"alert-{idx}",
                    source=benign_source,
                    target=benign_target,
                    kind="dns_lookup",
                    action="observed",
                    severity=1,
                    raw={"idx": idx, "event": "dns_lookup"},
                )
            if idx and idx % 1000 == 0:
                await asyncio.sleep(0)
