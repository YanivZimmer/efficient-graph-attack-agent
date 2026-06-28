from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


class _NoopSpan(AbstractContextManager):
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def set_attribute(self, _key: str, _value: Any) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, _name: str) -> _NoopSpan:
        return _NoopSpan()


def get_tracer(name: str) -> Any:
    try:
        from opentelemetry import trace
    except Exception:
        return _NoopTracer()
    return trace.get_tracer(name)
