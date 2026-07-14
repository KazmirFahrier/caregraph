"""Tracing.

`Tracer` is the seam CareGraph instruments against. `NoopTracer` is the default (zero
overhead in tests). `InMemoryTracer` records spans with duration and attributes so we
can assert on latency and per-stage metrics offline and print a trace in the demo.

In production you swap in `OpenTelemetryTracer` (below, import-guarded), which exports
the same spans to Arize Phoenix / any OTLP collector for latency, token-cost, and
faithfulness dashboards across the prompt -> retrieval -> generation path.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass
class Span:
    name: str
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer(Protocol):
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator["_SpanHandle"]:
        ...


@dataclass
class _SpanHandle:
    attributes: dict[str, Any]

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class NoopTracer:
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_SpanHandle]:
        yield _SpanHandle(attributes=dict(attributes))


class InMemoryTracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_SpanHandle]:
        handle = _SpanHandle(attributes=dict(attributes))
        start = time.perf_counter()
        try:
            yield handle
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.spans.append(Span(name=name, duration_ms=elapsed, attributes=handle.attributes))

    def by_name(self, name: str) -> Span | None:
        return next((s for s in self.spans if s.name == name), None)

    def reset(self) -> None:
        self.spans.clear()


class OpenTelemetryTracer:  # pragma: no cover - requires the `serve` extra
    """Exports spans to an OTLP collector / Arize Phoenix. Requires `caregraph[serve]`."""

    def __init__(self, service_name: str = "caregraph") -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError as exc:
            raise RuntimeError("OpenTelemetryTracer needs pip install 'caregraph[serve]'") from exc
        trace.set_tracer_provider(TracerProvider())
        self._tracer = trace.get_tracer(service_name)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_SpanHandle]:
        with self._tracer.start_as_current_span(name) as otel_span:
            handle = _SpanHandle(attributes=dict(attributes))
            for k, v in attributes.items():
                otel_span.set_attribute(k, v)
            try:
                yield handle
            finally:
                for k, v in handle.attributes.items():
                    otel_span.set_attribute(k, v)
