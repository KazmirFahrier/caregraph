"""Store for interrupted agent runs awaiting clinician sign-off.

`InMemoryPendingStore` is thread-safe (fixes A4-L2) and bounded by TTL + size. It is
correct for a single process only: the human-in-the-loop resume state does not survive
across replicas or restarts (A4-M2). For horizontal scale-out, implement `PendingStore`
against a shared TTL store (Redis/DB) with the same interface — callers are agnostic.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PendingStore(Protocol):
    def put(self, token: str, state: dict[str, Any]) -> None: ...
    def pop(self, token: str) -> dict[str, Any] | None: ...
    def __len__(self) -> int: ...


class InMemoryPendingStore:
    def __init__(self, ttl_s: float = 900.0, maxsize: int = 128) -> None:
        self.ttl_s = ttl_s
        self.maxsize = maxsize
        self._data: dict[str, tuple[dict[str, Any], float]] = {}
        self._lock = threading.Lock()

    def put(self, token: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._evict_locked()
            self._data[token] = (state, time.monotonic())

    def pop(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            self._evict_locked()
            entry = self._data.pop(token, None)
            return entry[0] if entry else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _evict_locked(self) -> None:
        # Snapshot under the lock so we never mutate while iterating (A4-L2).
        now = time.monotonic()
        for tok in [t for t, (_, ts) in list(self._data.items()) if now - ts > self.ttl_s]:
            del self._data[tok]
        if len(self._data) > self.maxsize:
            oldest = sorted(self._data, key=lambda k: self._data[k][1])
            for tok in oldest[: len(self._data) - self.maxsize]:
                del self._data[tok]
