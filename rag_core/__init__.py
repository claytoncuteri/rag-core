"""rag-core: A lightweight, modular RAG pipeline library for Python."""

from rag_core.config import RAGConfig
from rag_core.models import Chunk, Document, RAGResponse
from rag_core.pipeline import RAGPipeline

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "Document",
    "Chunk",
    "RAGResponse",
]

__version__ = "0.1.0"
