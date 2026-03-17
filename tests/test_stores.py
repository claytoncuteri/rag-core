"""Tests for vector stores."""

import numpy as np
import pytest

from rag_core.stores.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    """Create a fresh InMemoryStore for each test."""
    return InMemoryStore()


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embedding vectors (4 vectors of dimension 8)."""
    rng = np.random.RandomState(42)
    return rng.randn(4, 8).astype(np.float32)


@pytest.fixture
def sample_ids() -> list[str]:
    """Sample vector IDs."""
    return ["vec_0", "vec_1", "vec_2", "vec_3"]


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    def test_add_and_count(
        self,
        store: InMemoryStore,
        sample_ids: list[str],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Adding vectors should increase the count."""
        assert store.count() == 0
        store.add(ids=sample_ids, embeddings=sample_embeddings)
        assert store.count() == 4

    def test_add_with_documents(
        self,
        store: InMemoryStore,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Adding with documents should store text alongside vectors."""
        ids = ["a", "b", "c", "d"]
        docs = ["Hello", "World", "Foo", "Bar"]
        store.add(ids=ids, embeddings=sample_embeddings, documents=docs)

        # Search should return documents
        query = sample_embeddings[0]
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["document"] in docs

    def test_search_returns_correct_top_k(
        self,
        store: InMemoryStore,
        sample_ids: list[str],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Search should return exactly top_k results."""
        store.add(ids=sample_ids, embeddings=sample_embeddings)
        results = store.search(sample_embeddings[0], top_k=2)
        assert len(results) == 2

    def test_search_most_similar_first(
        self,
        store: InMemoryStore,
    ) -> None:
        """The most similar vector should be the first result."""
        # Create orthogonal-ish vectors
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)  # Similar to v1

        store.add(
            ids=["v1", "v2", "v3"],
            embeddings=np.array([v1, v2, v3]),
        )

        results = store.search(v1, top_k=3)
        # v1 should be the most similar to itself, v3 second
        assert results[0]["id"] == "v1"
        assert results[1]["id"] == "v3"

    def test_search_scores_between_zero_and_one(
        self,
        store: InMemoryStore,
        sample_ids: list[str],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Cosine similarity scores should be between -1 and 1."""
        store.add(ids=sample_ids, embeddings=sample_embeddings)
        results = store.search(sample_embeddings[0], top_k=4)
        for result in results:
            assert -1.0 <= result["score"] <= 1.0 + 1e-6

    def test_search_empty_store(self, store: InMemoryStore) -> None:
        """Searching an empty store should return an empty list."""
        query = np.random.randn(8).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_delete_vectors(
        self,
        store: InMemoryStore,
        sample_ids: list[str],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Deleting vectors should reduce the count and exclude them from search."""
        store.add(ids=sample_ids, embeddings=sample_embeddings)
        assert store.count() == 4

        store.delete(["vec_0", "vec_2"])
        assert store.count() == 2

        results = store.search(sample_embeddings[0], top_k=4)
        result_ids = {r["id"] for r in results}
        assert "vec_0" not in result_ids
        assert "vec_2" not in result_ids

    def test_update_embeddings(
        self,
        store: InMemoryStore,
    ) -> None:
        """Updating embeddings should change search results."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        store.add(
            ids=["a", "b"],
            embeddings=np.array([v1, v2]),
            documents=["doc_a", "doc_b"],
        )

        # Initially, v1 is most similar to query [1, 0, 0]
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)
        assert results[0]["id"] == "a"

        # Update "a" to point in v2's direction
        new_v = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        store.update(ids=["a"], embeddings=new_v)

        # Now "a" should be less similar to [1, 0, 0]
        results = store.search(query, top_k=2)
        assert results[0]["id"] == "b" or results[0]["score"] < 0.5

    def test_update_metadata(
        self,
        store: InMemoryStore,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Updating metadata should reflect in search results."""
        store.add(
            ids=["a", "b", "c", "d"],
            embeddings=sample_embeddings,
            metadatas=[{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}],
        )

        store.update(ids=["b"], metadatas=[{"v": 99}])
        results = store.search(sample_embeddings[1], top_k=1)
        assert results[0]["id"] == "b"
        assert results[0]["metadata"]["v"] == 99

    def test_clear(
        self,
        store: InMemoryStore,
        sample_ids: list[str],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Clearing the store should remove all vectors."""
        store.add(ids=sample_ids, embeddings=sample_embeddings)
        assert store.count() == 4

        store.clear()
        assert store.count() == 0

        query = np.random.randn(8).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_add_with_metadata(
        self,
        store: InMemoryStore,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Metadata should be stored and returned in search results."""
        ids = ["a", "b", "c", "d"]
        metas = [
            {"source": "file1.txt"},
            {"source": "file2.txt"},
            {"source": "file3.txt"},
            {"source": "file4.txt"},
        ]
        store.add(ids=ids, embeddings=sample_embeddings, metadatas=metas)

        results = store.search(sample_embeddings[0], top_k=1)
        assert "source" in results[0]["metadata"]

    def test_top_k_larger_than_store(
        self,
        store: InMemoryStore,
    ) -> None:
        """Requesting more results than stored vectors should return all vectors."""
        embeddings = np.random.randn(2, 4).astype(np.float32)
        store.add(ids=["a", "b"], embeddings=embeddings)

        results = store.search(embeddings[0], top_k=10)
        assert len(results) == 2
