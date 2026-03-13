"""Retriever that combines a vector store with an embedding provider."""

from __future__ import annotations

from typing import Any

from rag_core.embeddings.base import EmbeddingProvider
from rag_core.stores.base import VectorStore


class Retriever:
    """Combines a VectorStore and EmbeddingProvider for top-k retrieval.

    Embeds the query text and searches the vector store for the most
    similar chunks.

    Args:
        store: The vector store to search.
        embedding_provider: The provider used to embed query text.

    Example:
        >>> retriever = Retriever(store=my_store, embedding_provider=my_embeddings)
        >>> results = retriever.retrieve("What is machine learning?", top_k=5)
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant results for a query.

        Args:
            query: The natural language query string.
            top_k: Number of results to return.

        Returns:
            A list of result dicts from the vector store, each containing
            "id", "score", "metadata", and "document" keys.
        """
        query_embedding = self.embedding_provider.embed_query(query)
        return self.store.search(query_embedding=query_embedding, top_k=top_k)
