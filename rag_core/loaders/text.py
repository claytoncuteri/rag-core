"""Loader for plain text files."""

from __future__ import annotations

from pathlib import Path

from rag_core.loaders.base import BaseLoader
from rag_core.models import Document


class TextLoader(BaseLoader):
    """Load plain text files into Document objects.

    Args:
        encoding: Character encoding to use when reading files.

    Example:
        >>> loader = TextLoader()
        >>> docs = loader.load("notes.txt")
        >>> print(docs[0].content[:50])
    """

    supported_extensions: set[str] = {".txt", ".text"}

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        """Load a text file as a single Document.

        Args:
            path: Path to the text file.

        Returns:
            A list containing one Document with the file's content.
        """
        path = self._validate_file(path)
        content = path.read_text(encoding=self.encoding)

        return [
            Document(
                content=content,
                metadata={"source": str(path), "file_type": "text"},
                source=str(path),
            )
        ]
