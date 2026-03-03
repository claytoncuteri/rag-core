"""Data models for the RAG pipeline.

Defines the core data structures used throughout the library: Document, Chunk,
and RAGResponse.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A loaded document ready for processing.

    Attributes:
        content: The raw text content of the document.
        metadata: Arbitrary key-value metadata (e.g., title, author, date).
        source: The file path or URI where the document was loaded from.
        doc_id: A unique identifier for the document. Auto-generated if not provided.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Document content must not be empty.")


@dataclass
class Chunk:
    """A chunk of text extracted from a document.

    Attributes:
        text: The chunk's text content.
        metadata: Inherited and chunk-specific metadata.
        chunk_index: The position of this chunk within its source document.
        token_count: Approximate token count for the chunk text.
        doc_id: The ID of the parent document this chunk was extracted from.
        chunk_id: A unique identifier for this chunk. Auto-generated if not provided.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    token_count: int = 0
    doc_id: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.token_count == 0:
            # Rough approximation: 1 token per 4 characters
            self.token_count = max(1, len(self.text) // 4)


@dataclass
class RAGResponse:
    """The response from a RAG pipeline query.

    Attributes:
        answer: The generated answer text. Empty string if no generation step is used.
        sources: List of source identifiers (file paths or URIs) for the retrieved chunks.
        confidence_score: A score between 0.0 and 1.0 indicating retrieval confidence.
            Computed from the similarity scores of retrieved chunks.
        retrieved_chunks: The chunks that were retrieved and used to build the response.
    """

    answer: str = ""
    sources: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    retrieved_chunks: list[Chunk] = field(default_factory=list)
