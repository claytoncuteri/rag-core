"""Tests for chunking strategies."""

import pytest

from rag_core.chunkers.fixed import FixedSizeChunker
from rag_core.chunkers.recursive import RecursiveChunker
from rag_core.chunkers.semantic import SemanticChunker
from rag_core.models import Document


@pytest.fixture
def short_document() -> Document:
    """A short document that fits within one chunk."""
    return Document(content="Hello world. This is a short document.", source="test.txt")


@pytest.fixture
def long_document() -> Document:
    """A longer document with paragraphs and structure."""
    paragraphs = [
        "The quick brown fox jumped over the lazy dog. " * 5,
        "Machine learning is a subset of artificial intelligence. " * 5,
        "Python is a popular programming language for data science. " * 5,
    ]
    return Document(content="\n\n".join(paragraphs), source="test.txt")


@pytest.fixture
def structured_document() -> Document:
    """A document with headers and varied paragraph sizes."""
    return Document(
        content=(
            "# Introduction\n\n"
            "This is the introduction section. It provides an overview.\n\n"
            "# Methods\n\n"
            "We used several methods in this study. "
            "The first method involved data collection. "
            "The second method focused on analysis.\n\n"
            "# Results\n\n"
            "The results were significant. We found clear patterns in the data."
        ),
        source="paper.md",
    )


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_short_document_single_chunk(self, short_document: Document) -> None:
        """A short document should produce a single chunk."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(short_document)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == short_document.content.strip()

    def test_long_document_multiple_chunks(self, long_document: Document) -> None:
        """A long document should be split into multiple chunks."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(long_document)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, long_document: Document) -> None:
        """No chunk should exceed the specified chunk_size."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(long_document)
        for chunk in chunks:
            assert len(chunk.text) <= 100

    def test_chunk_overlap(self) -> None:
        """Consecutive chunks should share overlapping text."""
        text = "A" * 50 + "B" * 50 + "C" * 50
        doc = Document(content=text, source="test.txt")
        chunker = FixedSizeChunker(chunk_size=60, chunk_overlap=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3

        # Check that the end of one chunk overlaps with the start of the next
        for i in range(len(chunks) - 1):
            current_end = chunks[i].text[-20:]
            next_start = chunks[i + 1].text[:20]
            assert current_end == next_start

    def test_metadata_inherited(self, short_document: Document) -> None:
        """Chunks should inherit metadata from the parent document."""
        short_document.metadata = {"author": "test"}
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(short_document)
        assert chunks[0].metadata["author"] == "test"
        assert chunks[0].doc_id == short_document.doc_id

    def test_empty_content(self) -> None:
        """Empty content should produce no chunks."""
        doc = Document(content="   ", source="empty.txt")
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_chunk_indices_sequential(self, long_document: Document) -> None:
        """Chunk indices should be sequential starting from 0."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(long_document)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_invalid_parameters(self) -> None:
        """Invalid chunk parameters should raise ValueError."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0, chunk_overlap=0)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=-1)


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_splits_on_paragraphs(self) -> None:
        """Should split on double newlines (paragraph boundaries)."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = Document(content=text, source="test.txt")
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        # All paragraphs should fit in one chunk at size 500
        assert len(chunks) >= 1

    def test_splits_large_paragraphs(self) -> None:
        """Large paragraphs should be split at sentence boundaries."""
        sentence = "This is a test sentence. "
        big_paragraph = sentence * 50  # ~1250 chars
        doc = Document(content=big_paragraph, source="test.txt")
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 200

    def test_preserves_short_sections(self, structured_document: Document) -> None:
        """Short sections that fit within chunk_size should remain intact."""
        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.chunk(structured_document)
        # With a large chunk_size, everything fits in one chunk
        assert len(chunks) == 1


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_short_text_single_chunk(self, short_document: Document) -> None:
        """Short text should produce a single chunk without splitting."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(short_document)
        assert len(chunks) == 1

    def test_splits_on_double_newline_first(self) -> None:
        """Should prefer splitting on double newlines over finer separators."""
        text = "Part one content here.\n\nPart two content here."
        doc = Document(content=text, source="test.txt")
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert "Part one" in chunks[0].text
        assert "Part two" in chunks[1].text

    def test_falls_back_to_finer_separators(self) -> None:
        """When text has no double newlines, should split on finer separators."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        doc = Document(content=text, source="test.txt")
        chunker = RecursiveChunker(chunk_size=35, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_size_limit(self, long_document: Document) -> None:
        """All chunks should respect the chunk_size limit."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(long_document)
        for chunk in chunks:
            assert len(chunk.text) <= 100

    def test_custom_separators(self) -> None:
        """Custom separator lists should be respected."""
        text = "item1|item2|item3|item4"
        doc = Document(content=text, source="test.txt")
        chunker = RecursiveChunker(
            chunk_size=10,
            chunk_overlap=0,
            separators=["|"],
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3

    def test_token_count_populated(self, short_document: Document) -> None:
        """Chunks should have a non-zero token count."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(short_document)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count > 0
