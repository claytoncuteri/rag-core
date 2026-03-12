"""Abstract base class for text chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_core.models import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for splitting documents into chunks.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into a list of chunks.

        Args:
            document: The Document to split.

        Returns:
            A list of Chunk objects with text, metadata, and indices populated.
        """

    def _make_chunk(
        self,
        text: str,
        document: Document,
        chunk_index: int,
    ) -> Chunk:
        """Create a Chunk from text with inherited metadata.

        Args:
            text: The chunk text content.
            document: The parent Document for metadata inheritance.
            chunk_index: The position of this chunk in the sequence.

        Returns:
            A new Chunk instance.
        """
        metadata = dict(document.metadata)
        metadata["source"] = document.source

        return Chunk(
            text=text.strip(),
            metadata=metadata,
            chunk_index=chunk_index,
            doc_id=document.doc_id,
        )
