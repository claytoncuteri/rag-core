"""Local embedding provider using sentence-transformers."""

from __future__ import annotations

import numpy as np

from rag_core.embeddings.base import EmbeddingProvider


class LocalEmbeddings(EmbeddingProvider):
    """Embedding provider using local sentence-transformers models.

    Uses the all-MiniLM-L6-v2 model by default, which produces 384-dimensional
    vectors. The model is lazy-loaded on first use to avoid unnecessary startup
    cost.

    Requires the ``sentence-transformers`` package. Install with:
    ``pip install rag-core[local]``

    Args:
        model_name: Name of the sentence-transformers model to use.

    Example:
        >>> provider = LocalEmbeddings()
        >>> vectors = provider.embed(["Hello world"])
        >>> print(vectors.shape)
        (1, 384)
    """

    _DIMENSIONS: dict[str, int] = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L12-v2": 384,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dimension = self._DIMENSIONS.get(model_name, 384)
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddings. "
                "Install it with: pip install rag-core[local]"
            )

        self._model = SentenceTransformer(self.model_name)
        # Update dimension from the loaded model
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using the local model.

        Args:
            texts: List of text strings to embed.

        Returns:
            A numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text using the local model.

        Args:
            text: The query text to embed.

        Returns:
            A numpy array of shape (dimension,).
        """
        result = self.embed([text])
        return result[0]
