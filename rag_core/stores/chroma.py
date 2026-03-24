"""ChromaDB vector store wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_core.exceptions import StoreError
from rag_core.stores.base import VectorStore

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore[assignment]


class ChromaStore(VectorStore):
    """Vector store backed by ChromaDB for persistent storage.

    Requires the ``chromadb`` package. Install with:
    ``pip install rag-core[chroma]``

    Args:
        collection_name: Name of the ChromaDB collection to use or create.
        persist_directory: Directory for persistent storage. If None, uses
            an ephemeral in-memory ChromaDB client.

    Example:
        >>> store = ChromaStore(
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_data",
        ... )
        >>> store.add(ids=["a"], embeddings=vectors, documents=["Hello"])
    """

    def __init__(
        self,
        collection_name: str = "rag_core",
        persist_directory: str | None = None,
    ) -> None:
        if chromadb is None:
            raise StoreError(
                "chromadb is required for ChromaStore. "
                "Install it with: pip install rag-core[chroma]"
            )

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Add vectors to the ChromaDB collection.

        Args:
            ids: Unique identifiers for each vector.
            embeddings: Numpy array of shape (n, dimension).
            metadatas: Optional metadata dicts.
            documents: Optional raw text strings.
        """
        if len(ids) == 0:
            return

        # ChromaDB expects lists of lists for embeddings
        embedding_list = embeddings.tolist()

        kwargs: dict[str, Any] = {
            "ids": ids,
            "embeddings": embedding_list,
        }
        if metadatas:
            # ChromaDB requires metadata values to be str, int, float, or bool
            cleaned = [self._clean_metadata(m) for m in metadatas]
            kwargs["metadatas"] = cleaned
        if documents:
            kwargs["documents"] = documents

        self._collection.add(**kwargs)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in the ChromaDB collection.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            A list of result dicts sorted by descending similarity.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        output: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results["metadatas"] else [{} for _ in ids]
        documents = results["documents"][0] if results["documents"] else ["" for _ in ids]

        for i, doc_id in enumerate(ids):
            # ChromaDB returns distances; convert to similarity score
            # For cosine distance: similarity = 1 - distance
            score = 1.0 - distances[i]
            output.append({
                "id": doc_id,
                "score": float(score),
                "metadata": metadatas[i] or {},
                "document": documents[i] or "",
            })

        return output

    def delete(self, ids: list[str]) -> None:
        """Remove vectors by their IDs.

        Args:
            ids: List of vector IDs to remove.
        """
        if ids:
            self._collection.delete(ids=ids)

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
        kwargs: dict[str, Any] = {"ids": ids}

        if embeddings is not None:
            kwargs["embeddings"] = embeddings.tolist()
        if metadatas is not None:
            kwargs["metadatas"] = [self._clean_metadata(m) for m in metadatas]
        if documents is not None:
            kwargs["documents"] = documents

        self._collection.update(**kwargs)

    def clear(self) -> None:
        """Remove all vectors from the collection."""
        # ChromaDB does not have a bulk-clear, so delete and recreate
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        return self._collection.count()

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata values are ChromaDB-compatible types.

        Args:
            metadata: Raw metadata dict.

        Returns:
            A cleaned dict with only str, int, float, or bool values.
        """
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
