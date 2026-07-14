import pytest

from caregraph.rag import HashingEmbedder, InMemoryVectorStore, make_vector_store


def test_factory_returns_memory_store_by_default() -> None:
    store = make_vector_store(HashingEmbedder(dim=32), backend="memory")
    assert isinstance(store, InMemoryVectorStore)


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown vector backend"):
        make_vector_store(HashingEmbedder(dim=32), backend="pinecone")


def test_qdrant_backend_requires_url() -> None:
    # Selecting qdrant without a URL fails loudly instead of silently degrading.
    with pytest.raises(ValueError, match="QDRANT_URL"):
        make_vector_store(HashingEmbedder(dim=32), backend="qdrant", qdrant_url=None)
