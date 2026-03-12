"""End-to-end tests for the RAG pipeline.

All tests use a fake embedding provider so they work without API keys.
"""

import numpy as np
import pytest

from rag_core.embeddings.base import EmbeddingProvider
from rag_core.models import Document, RAGResponse
from rag_core.pipeline import RAGPipeline
from rag_core.stores.memory import InMemoryStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """A deterministic embedding provider for testing.

    Produces embeddings based on simple character-frequency features so that
    similar texts produce similar vectors. No external dependencies needed.
    """

    model_name = "fake-test-model"
    dimension = 26

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using character frequency features.

        Args:
            texts: List of text strings.

        Returns:
            Array of shape (len(texts), 26) with character frequency vectors.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            lower = text.lower()
            for char in lower:
                if "a" <= char <= "z":
                    vec[ord(char) - ord("a")] += 1.0
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            text: The query text.

        Returns:
            A vector of shape (26,).
        """
        return self.embed([text])[0]


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    """Create a fake embedding provider."""
    return FakeEmbeddingProvider()


@pytest.fixture
def store() -> InMemoryStore:
    """Create a fresh in-memory store."""
    return InMemoryStore()


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            content=(
                "Python is a high-level programming language. "
                "It is widely used for web development, data analysis, "
                "and machine learning applications."
            ),
            metadata={"source": "python_intro.txt", "topic": "programming"},
            source="python_intro.txt",
        ),
        Document(
            content=(
                "Machine learning algorithms learn patterns from data. "
                "Common algorithms include decision trees, neural networks, "
                "and support vector machines."
            ),
            metadata={"source": "ml_basics.txt", "topic": "ml"},
            source="ml_basics.txt",
        ),
        Document(
            content=(
                "Cooking pasta requires boiling water and adding salt. "
                "Fresh pasta takes about three minutes to cook, "
                "while dried pasta needs eight to twelve minutes."
            ),
            metadata={"source": "cooking.txt", "topic": "food"},
            source="cooking.txt",
        ),
    ]


class TestRAGPipeline:
    """End-to-end tests for RAGPipeline."""

    def test_ingest_documents(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
        sample_documents: list[Document],
    ) -> None:
        """Ingesting documents should add chunks to the store."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
            chunk_strategy="recursive",
            chunk_size=500,
        )

        count = pipeline.ingest(sample_documents)
        assert count > 0
        assert store.count() > 0

    def test_query_returns_rag_response(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
        sample_documents: list[Document],
    ) -> None:
        """Querying should return a RAGResponse with chunks and scores."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
            chunk_strategy="fixed",
            chunk_size=500,
        )

        pipeline.ingest(sample_documents)
        response = pipeline.query("What is Python?", top_k=2)

        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_chunks) <= 2
        assert response.confidence_score >= 0.0

    def test_query_empty_store(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
    ) -> None:
        """Querying an empty pipeline should return an empty response."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
        )

        response = pipeline.query("What is Python?")
        assert response.answer == ""
        assert response.retrieved_chunks == []
        assert response.confidence_score == 0.0

    def test_query_relevance(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
        sample_documents: list[Document],
    ) -> None:
        """Queries about programming should rank programming docs higher than cooking docs."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
            chunk_strategy="fixed",
            chunk_size=1000,
        )

        pipeline.ingest(sample_documents)
        response = pipeline.query("programming language python", top_k=3)

        # The first result should be more relevant to programming than cooking
        assert len(response.retrieved_chunks) > 0
        first_chunk = response.retrieved_chunks[0]
        # The character-frequency embedding should place programming text
        # closer to a programming query
        assert "python" in first_chunk.text.lower() or "programming" in first_chunk.text.lower()

    def test_clear_pipeline(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
        sample_documents: list[Document],
    ) -> None:
        """Clearing the pipeline should remove all stored data."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
        )

        pipeline.ingest(sample_documents)
        assert store.count() > 0

        pipeline.clear()
        assert store.count() == 0

    def test_different_chunk_strategies(
        self,
        fake_provider: FakeEmbeddingProvider,
        sample_documents: list[Document],
    ) -> None:
        """All chunk strategies should produce valid results."""
        for strategy in ["fixed", "semantic", "recursive"]:
            store = InMemoryStore()
            pipeline = RAGPipeline(
                embedding_provider=fake_provider,
                store=store,
                chunk_strategy=strategy,
                chunk_size=200,
                chunk_overlap=20,
            )

            count = pipeline.ingest(sample_documents)
            assert count > 0, f"Strategy '{strategy}' produced no chunks"

            response = pipeline.query("machine learning", top_k=2)
            assert isinstance(response, RAGResponse)
            assert len(response.retrieved_chunks) > 0

    def test_invalid_chunk_strategy(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
    ) -> None:
        """An invalid chunk strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            RAGPipeline(
                embedding_provider=fake_provider,
                store=store,
                chunk_strategy="nonexistent",  # type: ignore[arg-type]
            )

    def test_ingest_empty_list(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
    ) -> None:
        """Ingesting an empty document list should return 0."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
        )

        count = pipeline.ingest([])
        assert count == 0
        assert store.count() == 0

    def test_sources_populated(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
        sample_documents: list[Document],
    ) -> None:
        """Query response sources should contain the document source paths."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
            chunk_strategy="fixed",
            chunk_size=1000,
        )

        pipeline.ingest(sample_documents)
        response = pipeline.query("python programming", top_k=3)

        assert len(response.sources) > 0
        # Sources should be file paths from our test documents
        valid_sources = {"python_intro.txt", "ml_basics.txt", "cooking.txt"}
        for source in response.sources:
            assert source in valid_sources

    def test_answer_contains_context(
        self,
        fake_provider: FakeEmbeddingProvider,
        store: InMemoryStore,
        sample_documents: list[Document],
    ) -> None:
        """The answer field should contain formatted prompt with context."""
        pipeline = RAGPipeline(
            embedding_provider=fake_provider,
            store=store,
            chunk_strategy="fixed",
            chunk_size=1000,
        )

        pipeline.ingest(sample_documents)
        response = pipeline.query("What is machine learning?")

        # Answer should contain the QA template structure
        assert "Context:" in response.answer
        assert "Question:" in response.answer
        assert "Answer:" in response.answer
