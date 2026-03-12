"""Loader for Markdown files."""

from __future__ import annotations

import re
from pathlib import Path

from rag_core.loaders.base import BaseLoader
from rag_core.models import Document


class MarkdownLoader(BaseLoader):
    """Load Markdown files, optionally stripping formatting.

    Args:
        encoding: Character encoding to use when reading files.
        strip_formatting: If True, remove Markdown syntax (headers, bold,
            links, etc.) and return plain text.

    Example:
        >>> loader = MarkdownLoader(strip_formatting=True)
        >>> docs = loader.load("README.md")
    """

    supported_extensions: set[str] = {".md", ".markdown", ".mdown"}

    def __init__(
        self,
        encoding: str = "utf-8",
        strip_formatting: bool = False,
    ) -> None:
        self.encoding = encoding
        self.strip_formatting = strip_formatting

    def load(self, path: str | Path) -> list[Document]:
        """Load a Markdown file as a single Document.

        Args:
            path: Path to the Markdown file.

        Returns:
            A list containing one Document with the file's content.
        """
        path = self._validate_file(path)
        content = path.read_text(encoding=self.encoding)

        if self.strip_formatting:
            content = self._strip_markdown(content)

        return [
            Document(
                content=content,
                metadata={"source": str(path), "file_type": "markdown"},
                source=str(path),
            )
        ]

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove common Markdown formatting from text.

        Strips headers, bold/italic markers, links, images, inline code,
        and code blocks.

        Args:
            text: Raw Markdown text.

        Returns:
            Plain text with Markdown syntax removed.
        """
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Remove links, keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Remove blockquotes
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
        # Clean up extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
