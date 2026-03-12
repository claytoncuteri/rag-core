"""OpenAI embedding provider using the text-embedding-3-small model."""

from __future__ import annotations

import numpy as np

from rag_core.embeddings.base import EmbeddingProvider

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


class OpenAIEmbeddings(EmbeddingProvider):
    """Embedding provider using OpenAI's text-embedding-3-small model.

    Requires the ``openai`` package and a valid API key. Install with:
    ``pip install rag-core[openai]``

    Args:
        model: The OpenAI embedding model name.
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY
            environment variable.
        batch_size: Maximum number of texts to embed per API call.

    Example:
        >>> provider = OpenAIEmbeddings(api_key="sk-...")
        >>> vectors = provider.embed(["Hello world", "Test text"])
        >>> print(vectors.shape)
        (2, 1536)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        if openai is None:
            raise ImportError(
                "openai is required for OpenAIEmbeddings. "
                "Install it with: pip install rag-core[openai]"
            )

        self.model_name = model
        self.dimension = 1536
        self._batch_size = batch_size
        self._client = openai.OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using the OpenAI API.

        Args:
            texts: List of text strings to embed.

        Returns:
            A numpy array of shape (len(texts), 1536).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self.model_name,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text using the OpenAI API.

        Args:
            text: The query text to embed.

        Returns:
            A numpy array of shape (1536,).
        """
        result = self.embed([text])
        return result[0]
