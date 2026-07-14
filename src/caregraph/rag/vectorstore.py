"""Vector store abstraction.

`VectorStore` is the seam that lets CareGraph run offline (`InMemoryVectorStore`) or
against a managed service (`QdrantVectorStore`) without touching the pipeline. The
Qdrant adapter is import-guarded: it only requires `qdrant-client` when actually
instantiated, so the offline slice has zero heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from caregraph.rag.embeddings import Embedder, cosine


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    # "trusted" = curated guideline / structured record; "untrusted" = ingested,
    # patient-authored, or retrieved prose. Untrusted text may be retrieved for
    # relevance but must never ground or be cited for a clinical claim (A3-H1).
    trust: str = "trusted"


@dataclass
class ScoredDocument:
    document: Document
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def add(self, documents: list[Document]) -> None: ...
    def search(self, query: str, top_k: int) -> list[ScoredDocument]: ...


class InMemoryVectorStore:
    """Brute-force cosine search. Correct and dependency-free; not for scale."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._docs: list[Document] = []
        self._vectors: list[list[float]] = []

    def add(self, documents: list[Document]) -> None:
        for doc in documents:
            self._docs.append(doc)
            self._vectors.append(self._embedder.embed(doc.text))

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        q = self._embedder.embed(query)
        scored = [
            ScoredDocument(doc, cosine(q, vec))
            for doc, vec in zip(self._docs, self._vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class QdrantVectorStore:
    """Managed-vector-DB adapter. Requires the `rag` extra (`qdrant-client`).

    Kept behind a lazy import so the offline path never pays for it. The public
    surface is identical to `InMemoryVectorStore`, so `ClinicalRAG` is agnostic. Not
    exercised in CI (no Qdrant service), but it is a complete implementation, not a
    stub — selected via `make_vector_store` when CAREGRAPH_VECTOR_BACKEND=qdrant.
    """

    def __init__(self, embedder: Embedder, url: str, collection: str = "caregraph") -> None:
        try:  # pragma: no cover - requires the rag extra + a running Qdrant
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("QdrantVectorStore requires `pip install caregraph[rag]`") from exc
        self._embedder = embedder
        self._collection = collection
        self._PointStruct = PointStruct
        self._client = QdrantClient(url=url)
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )
        self._next_id = 0

    def add(self, documents: list[Document]) -> None:  # pragma: no cover - needs Qdrant
        points = []
        for doc in documents:
            points.append(self._PointStruct(
                id=self._next_id,
                vector=self._embedder.embed(doc.text),
                payload={"doc_id": doc.doc_id, "text": doc.text, "trust": doc.trust,
                         **doc.metadata},
            ))
            self._next_id += 1
        self._client.upsert(self._collection, points=points)

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:  # pragma: no cover
        hits = self._client.search(
            self._collection, query_vector=self._embedder.embed(query), limit=top_k
        )
        results: list[ScoredDocument] = []
        for h in hits:
            payload = h.payload or {}
            doc = Document(doc_id=payload.get("doc_id", ""), text=payload.get("text", ""),
                           metadata={k: v for k, v in payload.items()
                                     if k not in ("doc_id", "text", "trust")},
                           trust=payload.get("trust", "trusted"))
            results.append(ScoredDocument(document=doc, score=float(h.score)))
        return results


def make_vector_store(embedder: Embedder, backend: str = "memory",
                      qdrant_url: str | None = None) -> VectorStore:
    """Factory honoring `Settings.vector_backend`. Defaults to the offline store."""
    if backend == "qdrant":
        if not qdrant_url:
            raise ValueError("qdrant backend selected but QDRANT_URL is unset")
        return QdrantVectorStore(embedder, url=qdrant_url)
    if backend == "memory":
        return InMemoryVectorStore(embedder)
    raise ValueError(f"unknown vector backend: {backend!r}")
