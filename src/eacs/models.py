from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityType(str, Enum):
    IP = "ip"
    USER = "user"
    HOST = "host"
    SERVICE = "service"
    FILE = "file"


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: EntityType
    value: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def strip_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("entity value cannot be blank")
        return value

    @property
    def id(self) -> str:
        return f"{self.type.value}:{self.value.lower()}"


class Alert(BaseModel):
    id: str = Field(min_length=1)
    source: Entity
    target: Optional[Entity] = None
    kind: str = Field(min_length=1)
    action: str = Field(default="observed", min_length=1)
    severity: int = Field(default=1, ge=0, le=10)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict[str, Any] = Field(default_factory=dict)
    tags: set[str] = Field(default_factory=set)

    @field_validator("id", "kind", "action")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text fields cannot be blank")
        return value

    @property
    def relationship_key(self) -> str:
        target = self.target.id if self.target else "-"
        return "|".join(
            [
                self.source.id,
                target,
                self.kind.lower(),
                self.action.lower(),
            ]
        )

    @property
    def entities(self) -> tuple[Entity, ...]:
        return (self.source,) if self.target is None else (self.source, self.target)

    @property
    def is_high_severity(self) -> bool:
        return self.severity >= 8


class GraphEdge(BaseModel):
    alert_id: str
    entity_id: str
    role: Literal["source", "target"]


class Subgraph(BaseModel):
    root_alert_id: str
    alerts: list[Alert] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class ValidationResult(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class AttackStory(BaseModel):
    storyline: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    entity_graph: Subgraph
    event_intents: list[str] = Field(default_factory=list)
    hydrated_logs: list[dict[str, Any]] = Field(default_factory=list)
