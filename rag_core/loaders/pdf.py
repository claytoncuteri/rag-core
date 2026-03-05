"""Loader for PDF files using pypdf."""

from __future__ import annotations

from pathlib import Path

from rag_core.loaders.base import BaseLoader
from rag_core.models import Document

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[assignment, misc]


class PDFLoader(BaseLoader):
    """Load PDF files into Document objects using pypdf.

    Requires the `pypdf` package. Install with: ``pip install rag-core[pdf]``

    Args:
        pages_as_documents: If True, each page becomes a separate Document.
            If False (default), the entire PDF is one Document.

    Example:
        >>> loader = PDFLoader(pages_as_documents=True)
        >>> docs = loader.load("report.pdf")
        >>> print(f"Loaded {len(docs)} pages")
    """

    supported_extensions: set[str] = {".pdf"}

    def __init__(self, pages_as_documents: bool = False) -> None:
        if PdfReader is None:
            raise ImportError(
                "pypdf is required for PDFLoader. "
                "Install it with: pip install rag-core[pdf]"
            )
        self.pages_as_documents = pages_as_documents

    def load(self, path: str | Path) -> list[Document]:
        """Load a PDF file as one or more Documents.

        Args:
            path: Path to the PDF file.

        Returns:
            A list of Document objects. One per file, or one per page if
            ``pages_as_documents`` is True.
        """
        path = self._validate_file(path)
        reader = PdfReader(str(path))

        if self.pages_as_documents:
            documents = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(path),
                                "file_type": "pdf",
                                "page_number": i + 1,
                                "total_pages": len(reader.pages),
                            },
                            source=str(path),
                        )
                    )
            return documents
        else:
            all_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    all_text.append(text)

            combined = "\n\n".join(all_text)
            if not combined.strip():
                combined = "(No extractable text found in PDF)"

            return [
                Document(
                    content=combined,
                    metadata={
                        "source": str(path),
                        "file_type": "pdf",
                        "total_pages": len(reader.pages),
                    },
                    source=str(path),
                )
            ]
