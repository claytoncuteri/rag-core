"""In-memory vector store using numpy for similarity search."""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_core.stores.base import VectorStore


class InMemoryStore(VectorStore):
    """In-memory vector store backed by a numpy matrix.

    Uses cosine similarity for search. Vectors are L2-normalized at insert
    time so that cosine similarity reduces to a dot product, which is fast
    to compute.

    Suitable for small to medium datasets (up to ~100k vectors). For larger
    datasets, consider using ChromaStore instead.

    Example:
        >>> store = InMemoryStore()
        >>> store.add(
        ...     ids=["a", "b"],
        ...     embeddings=np.random.randn(2, 384).astype(np.float32),
        ...     documents=["Hello", "World"],
        ... )
        >>> results = store.search(np.random.randn(384).astype(np.float32))
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._metadatas: list[dict[str, Any]] = []
        self._documents: list[str] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via dot product.

        Args:
            vectors: Array of shape (n, d) or (d,).

        Returns:
            Normalized array of the same shape.
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm > 0 else vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vectors / norms

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Add vectors to the in-memory store.

        Args:
            ids: Unique identifiers for each vector.
            embeddings: Numpy array of shape (n, dimension).
            metadatas: Optional metadata dicts.
            documents: Optional raw text strings.
        """
        if len(ids) == 0:
            return

        normalized = self._normalize(embeddings.astype(np.float32))

        if self._embeddings is None:
            self._embeddings = normalized
        else:
            self._embeddings = np.vstack([self._embeddings, normalized])

        self._ids.extend(ids)
        self._metadatas.extend(metadatas or [{} for _ in ids])
        self._documents.extend(documents or ["" for _ in ids])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors using cosine similarity.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            A list of result dicts sorted by descending similarity score.
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        query_norm = self._normalize(query_embedding.astype(np.float32))

        # Cosine similarity via dot product (vectors are pre-normalized)
        scores = self._embeddings @ query_norm

        # Get top-k indices
        k = min(top_k, len(self._ids))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            results.append({
                "id": self._ids[idx],
                "score": float(scores[idx]),
                "metadata": self._metadatas[idx],
                "document": self._documents[idx],
            })

        return results

    def delete(self, ids: list[str]) -> None:
        """Remove vectors by their IDs.

        Args:
            ids: List of vector IDs to remove.
        """
        ids_set = set(ids)
        keep_mask = [i for i, vid in enumerate(self._ids) if vid not in ids_set]

        if not keep_mask:
            self.clear()
            return

        self._ids = [self._ids[i] for i in keep_mask]
        self._metadatas = [self._metadatas[i] for i in keep_mask]
        self._documents = [self._documents[i] for i in keep_mask]
        if self._embeddings is not None:
            self._embeddings = self._embeddings[keep_mask]

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
        id_to_idx = {vid: i for i, vid in enumerate(self._ids)}

        for j, target_id in enumerate(ids):
            if target_id not in id_to_idx:
                continue
            idx = id_to_idx[target_id]

            if embeddings is not None and self._embeddings is not None:
                normalized = self._normalize(embeddings[j].astype(np.float32))
                self._embeddings[idx] = normalized

            if metadatas is not None:
                self._metadatas[idx] = metadatas[j]

            if documents is not None:
                self._documents[idx] = documents[j]

    def clear(self) -> None:
        """Remove all vectors from the store."""
        self._ids.clear()
        self._embeddings = None
        self._metadatas.clear()
        self._documents.clear()

    def count(self) -> int:
        """Return the number of vectors in the store."""
        return len(self._ids)
