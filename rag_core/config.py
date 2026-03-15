"""Configuration defaults for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RAGConfig:
    """Configuration for a RAG pipeline instance.

    Attributes:
        chunk_strategy: The chunking strategy to use. One of "fixed", "semantic",
            or "recursive".
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        embedding_model: Name of the embedding model to use.
        embedding_dimension: Dimensionality of the embedding vectors.
        top_k: Default number of results to retrieve per query.
        similarity_metric: Distance metric for vector search. Currently only
            "cosine" is supported.
        store_type: The vector store backend. One of "memory" or "chroma".
        store_params: Additional parameters passed to the vector store constructor.
    """

    chunk_strategy: Literal["fixed", "semantic", "recursive"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    top_k: int = 5
    similarity_metric: Literal["cosine"] = "cosine"
    store_type: Literal["memory", "chroma"] = "memory"
    store_params: dict[str, Any] = field(default_factory=dict)
