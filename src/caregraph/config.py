"""Runtime configuration.

Evaluated at *instantiation* time (not import time) so environment changes are
never silently ignored — a defect fixed the hard way in campaign-copilot.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    embedding_dim: int = 256
    top_k: int = 4
    # Backend selection; falls back to offline defaults when unset.
    vector_backend: str = field(default_factory=lambda: os.getenv("CAREGRAPH_VECTOR_BACKEND", "memory"))
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL"))
    # Fail loudly rather than degrade: refuse answers without citations.
    require_citations: bool = True

    @classmethod
    def load(cls) -> "Settings":
        return cls()
