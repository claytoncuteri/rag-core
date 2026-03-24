"""Custom exception classes for the rag-core library.

Provides a hierarchy of exceptions for different pipeline stages so that
callers can catch specific failures without relying on generic ValueError
or RuntimeError.
"""


class RAGError(Exception):
    """Base exception for all rag-core errors."""


class DocumentLoadError(RAGError):
    """Raised when a document cannot be loaded from a file.

    Common causes: file not found, unsupported format, encoding issues,
    or corrupt file content.
    """


class ChunkingError(RAGError):
    """Raised when a document cannot be chunked.

    Common causes: empty document content, invalid chunker configuration,
    or an unrecognized chunking strategy.
    """


class EmbeddingError(RAGError):
    """Raised when text embedding fails.

    Common causes: model not loaded, API key missing, API rate limit
    exceeded, or input text exceeds model token limit.
    """


class StoreError(RAGError):
    """Raised when a vector store operation fails.

    Common causes: dimension mismatch between vectors, duplicate IDs,
    or backend connection issues (e.g., ChromaDB unavailable).
    """


class PipelineError(RAGError):
    """Raised when the RAG pipeline encounters an unrecoverable error.

    Wraps lower-level exceptions that occur during ingest or query
    operations to provide a single catch point for pipeline callers.
    """
