"""Document loaders for various file formats."""

from rag_core.loaders.base import BaseLoader
from rag_core.loaders.csv_loader import CSVLoader
from rag_core.loaders.markdown import MarkdownLoader
from rag_core.loaders.text import TextLoader

__all__ = [
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "CSVLoader",
]

# PDFLoader requires pypdf, so import it conditionally
try:
    from rag_core.loaders.pdf import PDFLoader  # noqa: F401

    __all__.append("PDFLoader")
except ImportError:
    pass
