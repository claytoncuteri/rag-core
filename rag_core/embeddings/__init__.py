"""Embedding providers for converting text to vectors."""

from rag_core.embeddings.base import EmbeddingProvider
from rag_core.embeddings.cache import EmbeddingCache

__all__ = [
    "EmbeddingProvider",
    "EmbeddingCache",
]

# Optional providers that require extra dependencies
try:
    from rag_core.embeddings.openai_embeddings import OpenAIEmbeddings  # noqa: F401

    __all__.append("OpenAIEmbeddings")
except ImportError:
    pass

try:
    from rag_core.embeddings.local_embeddings import LocalEmbeddings  # noqa: F401

    __all__.append("LocalEmbeddings")
except ImportError:
    pass
