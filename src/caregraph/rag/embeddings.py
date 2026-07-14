"""Embeddings.

`HashingEmbedder` is a deterministic, dependency-free bag-of-words hashing embedder
so retrieval works offline in CI. In production you swap in a
`SentenceTransformerEmbedder` (or an OpenAI/HF endpoint) behind the same `Embedder`
protocol — the vector store and pipeline are agnostic to which one is used.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    """L2-normalized hashed bag-of-words vectors. Deterministic and offline."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            h = _stable_hash(token) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _stable_hash(token: str) -> int:
    """Process-independent hash (unlike builtin `hash`) so CI is reproducible."""
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
