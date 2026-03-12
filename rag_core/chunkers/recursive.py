"""Recursive chunker that tries progressively finer separators."""

from __future__ import annotations

from rag_core.chunkers.base import BaseChunker
from rag_core.models import Chunk, Document


class RecursiveChunker(BaseChunker):
    """Split documents using a hierarchy of separators.

    Tries to split on double newlines first, then single newlines, then
    sentence-ending periods followed by a space, and finally falls back
    to raw character-count splitting. This approach preserves the most
    meaningful boundaries possible while guaranteeing a maximum chunk size.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Ordered list of separator strings to try. Defaults to
            ["\\n\\n", "\\n", ". ", " "].

    Example:
        >>> chunker = RecursiveChunker(chunk_size=400, chunk_overlap=40)
        >>> chunks = chunker.chunk(document)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document using recursive separator hierarchy.

        Args:
            document: The Document to split.

        Returns:
            A list of Chunk objects, each within ``chunk_size`` characters.
        """
        text = document.content
        if not text.strip():
            return []

        texts = self._recursive_split(text, self.separators)

        chunks: list[Chunk] = []
        for i, chunk_text in enumerate(texts):
            if chunk_text.strip():
                chunks.append(self._make_chunk(chunk_text, document, i))

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using the separator hierarchy.

        Args:
            text: The text to split.
            separators: Remaining separators to try, in order.

        Returns:
            A list of text segments, each within chunk_size characters.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Final fallback: character-level split with overlap
            return self._char_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on current separator
        parts = text.split(separator)

        # Merge small parts, recursively split large ones
        result: list[str] = []
        buffer = ""

        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)

                if len(part) <= self.chunk_size:
                    buffer = part
                else:
                    # This part is still too large: try next separator
                    sub_parts = self._recursive_split(part, remaining_separators)
                    result.extend(sub_parts)
                    buffer = ""

        if buffer:
            result.append(buffer)

        return result

    def _char_split(self, text: str) -> list[str]:
        """Split text by raw character count with overlap.

        Args:
            text: The text to split.

        Returns:
            A list of text segments.
        """
        result: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end]
            if segment.strip():
                result.append(segment)
            start += step

        return result
