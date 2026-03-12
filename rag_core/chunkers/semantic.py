"""Semantic chunker that splits on paragraph breaks, headers, and sentence groups."""

from __future__ import annotations

import re

from rag_core.chunkers.base import BaseChunker
from rag_core.models import Chunk, Document


class SemanticChunker(BaseChunker):
    """Split documents on semantic boundaries.

    Splits text at paragraph breaks (double newlines) and Markdown-style
    headers first. If individual sections exceed ``chunk_size``, they are
    further split at sentence boundaries. Sentences that still exceed the
    limit are split at the character level as a final fallback.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
            Applied when sentence-level splitting is needed.

    Example:
        >>> chunker = SemanticChunker(chunk_size=300)
        >>> chunks = chunker.chunk(document)
    """

    # Pattern for splitting on paragraph breaks and Markdown headers
    _section_pattern = re.compile(r"\n\n+|(?=^#{1,6}\s)", flags=re.MULTILINE)

    # Pattern for splitting on sentence boundaries
    _sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document on semantic boundaries.

        Args:
            document: The Document to split.

        Returns:
            A list of Chunk objects split at paragraph and sentence boundaries.
        """
        text = document.content
        if not text.strip():
            return []

        # First pass: split on paragraphs and headers
        sections = self._section_pattern.split(text)
        sections = [s.strip() for s in sections if s.strip()]

        # Second pass: merge small sections, split large ones
        final_texts: list[str] = []
        buffer = ""

        for section in sections:
            if len(buffer) + len(section) + 1 <= self.chunk_size:
                buffer = f"{buffer}\n\n{section}" if buffer else section
            else:
                if buffer:
                    final_texts.append(buffer)
                if len(section) <= self.chunk_size:
                    buffer = section
                else:
                    # Section too large: split on sentences
                    sentence_chunks = self._split_by_sentences(section)
                    final_texts.extend(sentence_chunks)
                    buffer = ""

        if buffer:
            final_texts.append(buffer)

        chunks: list[Chunk] = []
        for i, chunk_text in enumerate(final_texts):
            if chunk_text.strip():
                chunks.append(self._make_chunk(chunk_text, document, i))

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text at sentence boundaries, respecting chunk_size.

        Args:
            text: The text to split.

        Returns:
            A list of text segments, each within chunk_size characters.
        """
        sentences = self._sentence_pattern.split(text)
        result: list[str] = []
        buffer = ""

        for sentence in sentences:
            if len(buffer) + len(sentence) + 1 <= self.chunk_size:
                buffer = f"{buffer} {sentence}" if buffer else sentence
            else:
                if buffer:
                    result.append(buffer.strip())
                if len(sentence) <= self.chunk_size:
                    buffer = sentence
                else:
                    # Sentence exceeds chunk_size: fall back to char split
                    for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                        result.append(sentence[i : i + self.chunk_size].strip())
                    buffer = ""

        if buffer:
            result.append(buffer.strip())

        return result
