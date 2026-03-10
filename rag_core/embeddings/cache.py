"""Embedding cache that wraps any provider with hash-based caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from rag_core.embeddings.base import EmbeddingProvider


class EmbeddingCache(EmbeddingProvider):
    """Caching wrapper around any EmbeddingProvider.

    Caches embeddings in memory keyed by a hash of (text, model_name). The
    cache can also be persisted to disk as a ``.npz`` file for reuse across
    sessions.

    Args:
        provider: The underlying EmbeddingProvider to cache.
        cache_path: Optional path to a ``.npz`` file for persisting the
            cache to disk. If None, the cache is in-memory only.

    Example:
        >>> from rag_core.embeddings import LocalEmbeddings
        >>> provider = LocalEmbeddings()
        >>> cached = EmbeddingCache(provider, cache_path="./embeddings.npz")
        >>> vectors = cached.embed(["text"])  # Computed and cached
        >>> vectors = cached.embed(["text"])  # Served from cache
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_path: str | Path | None = None,
    ) -> None:
        self._provider = provider
        self.model_name = provider.model_name
        self.dimension = provider.dimension
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path = Path(cache_path) if cache_path else None

        if self._cache_path and self._cache_path.exists():
            self._load_cache()

    def _make_key(self, text: str) -> str:
        """Generate a cache key from text and model name.

        Args:
            text: The input text.

        Returns:
            A hex digest string used as the cache key.
        """
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts, using cached results where available.

        Args:
            texts: List of text strings to embed.

        Returns:
            A numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        results: list[np.ndarray] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            key = self._make_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(np.zeros(0))  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            new_embeddings = self._provider.embed(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                key = self._make_key(uncached_texts[j])
                embedding = new_embeddings[j]
                self._cache[key] = embedding
                results[idx] = embedding

        return np.stack(results, axis=0)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text, using cache if available.

        Args:
            text: The query text to embed.

        Returns:
            A numpy array of shape (dimension,).
        """
        key = self._make_key(text)
        if key in self._cache:
            return self._cache[key]

        embedding = self._provider.embed_query(text)
        self._cache[key] = embedding
        return embedding

    def save_cache(self) -> None:
        """Save the cache to disk as a .npz file.

        Does nothing if no cache_path was configured.
        """
        if not self._cache_path:
            return

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self._cache_path), **self._cache)

    def _load_cache(self) -> None:
        """Load the cache from a .npz file on disk."""
        if not self._cache_path or not self._cache_path.exists():
            return

        data = np.load(str(self._cache_path))
        self._cache = {key: data[key] for key in data.files}

    def clear_cache(self) -> None:
        """Clear all cached embeddings from memory."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._cache)
