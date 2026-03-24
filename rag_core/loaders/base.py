"""Abstract base class for document loaders."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

from rag_core.exceptions import DocumentLoadError
from rag_core.models import Document


class BaseLoader(ABC):
    """Abstract base class for loading documents from files.

    Subclasses must implement the `load` method to handle specific file
    formats. The `load_directory` method is provided for convenience and
    calls `load` on each matching file.

    Attributes:
        supported_extensions: Set of file extensions this loader can handle
            (e.g., {".txt", ".text"}).
    """

    supported_extensions: set[str] = set()

    @abstractmethod
    def load(self, path: str | Path) -> list[Document]:
        """Load a single file and return a list of Documents.

        Args:
            path: Path to the file to load.

        Returns:
            A list of Document objects extracted from the file. Most loaders
            return a single Document, but some (like CSVLoader) may return
            multiple Documents from one file.

        Raises:
            DocumentLoadError: If the file cannot be loaded.
        """

    def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        """Load all supported files from a directory.

        Args:
            dir_path: Path to the directory to scan.
            recursive: If True, scan subdirectories as well.

        Returns:
            A list of Document objects from all matching files.

        Raises:
            DocumentLoadError: If dir_path is not a directory.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise DocumentLoadError(f"Not a directory: {dir_path}")

        documents: list[Document] = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self.load(file_path)
                    documents.extend(docs)
                    logger.debug("Loaded %s (%d documents)", file_path.name, len(docs))
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", file_path, exc)
                    continue

        logger.info("Loaded %d documents from %s", len(documents), dir_path)
        return documents

    def _validate_file(self, path: str | Path) -> Path:
        """Validate that a file exists and has a supported extension.

        Args:
            path: Path to validate.

        Returns:
            The resolved Path object.

        Raises:
            DocumentLoadError: If the file does not exist or is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if not path.is_file():
            raise DocumentLoadError(f"Not a file: {path}")
        if self.supported_extensions and path.suffix.lower() not in self.supported_extensions:
            raise DocumentLoadError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {self.supported_extensions}"
            )
        return path
