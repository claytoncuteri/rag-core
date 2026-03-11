"""Vector store backends for embedding storage and search."""

from rag_core.stores.base import VectorStore
from rag_core.stores.memory import InMemoryStore

__all__ = [
    "VectorStore",
    "InMemoryStore",
]

try:
    from rag_core.stores.chroma import ChromaStore

    __all__.append("ChromaStore")
except ImportError:
    pass
