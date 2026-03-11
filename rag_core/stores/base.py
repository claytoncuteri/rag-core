"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector storage and similarity search.

    Provides the interface that all vector store backends must implement.
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Add vectors and their associated data to the store.

        Args:
            ids: Unique identifiers for each vector.
            embeddings: Numpy array of shape (n, dimension) containing vectors.
            metadatas: Optional list of metadata dicts, one per vector.
            documents: Optional list of raw text strings, one per vector.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for the most similar vectors to a query.

        Args:
            query_embedding: A numpy array of shape (dimension,) to search for.
            top_k: Number of results to return.

        Returns:
            A list of dicts, each containing at minimum:
                - "id": The vector's unique identifier.
                - "score": The similarity score (higher is more similar).
                - "metadata": The associated metadata dict.
                - "document": The associated raw text (if stored).
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by their IDs.

        Args:
            ids: List of vector IDs to remove.
        """

    @abstractmethod
    def update(
        self,
        ids: list[str],
        embeddings: np.ndarray | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Update existing vectors and/or their metadata.

        Args:
            ids: Identifiers of the vectors to update.
            embeddings: Optional new embedding vectors.
            metadatas: Optional new metadata dicts.
            documents: Optional new document texts.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors from the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
