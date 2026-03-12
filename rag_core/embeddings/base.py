"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding providers.

    Subclasses must implement ``embed`` and ``embed_query`` to convert text
    into dense vector representations.

    Attributes:
        model_name: Identifier for the embedding model.
        dimension: Dimensionality of the output vectors.
    """

    model_name: str = ""
    dimension: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            A numpy array of shape (len(texts), dimension) containing the
            embedding vectors.
        """

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text into a vector.

        Args:
            text: The query text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding vector.
        """
