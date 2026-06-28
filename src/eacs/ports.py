from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, Optional

from .models import Alert, Subgraph


class LLMProvider(ABC):
    """Strategy interface for reasoning providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GraphStore(ABC):
    """Repository interface for alert/entity graph persistence."""

    @abstractmethod
    async def upsert_alert(self, alert: Alert) -> None:
        raise NotImplementedError

    @abstractmethod
    async def fetch_alert(self, alert_id: str) -> Optional[Alert]:
        raise NotImplementedError

    @abstractmethod
    async def two_hop_neighborhood(self, alert_id: str) -> Subgraph:
        raise NotImplementedError


class AlertStream(ABC):
    """Async source of validated alerts."""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Alert]:
        raise NotImplementedError


class LogStore(ABC):
    """Repository interface for cold raw-log hydration."""

    @abstractmethod
    async def fetch_logs(self, alert_ids: Sequence[str]) -> list[dict[str, Any]]:
        raise NotImplementedError
