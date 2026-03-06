"""Fixed-size character count chunker."""

from __future__ import annotations

from rag_core.chunkers.base import BaseChunker
from rag_core.models import Chunk, Document


class FixedSizeChunker(BaseChunker):
    """Split documents into fixed-size chunks by character count.

    Splits text into chunks of at most ``chunk_size`` characters, with
    ``chunk_overlap`` characters shared between consecutive chunks.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Example:
        >>> chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        >>> chunks = chunker.chunk(document)
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into fixed-size character chunks.

        Args:
            document: The Document to split.

        Returns:
            A list of Chunk objects, each with at most ``chunk_size`` characters.
        """
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(self._make_chunk(chunk_text, document, index))
                index += 1

            # Move forward by chunk_size minus overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks
