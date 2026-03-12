"""Loader for CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from rag_core.loaders.base import BaseLoader
from rag_core.models import Document


class CSVLoader(BaseLoader):
    """Load CSV files into Document objects.

    Can produce either one Document per row or one Document for the entire file.

    Args:
        encoding: Character encoding to use when reading files.
        content_columns: Column names to concatenate as the Document content.
            If None, all columns are used.
        metadata_columns: Column names to include as Document metadata.
            If None, all columns not in content_columns are used.
        one_doc_per_row: If True (default), each row becomes a separate Document.
            If False, the entire CSV is combined into a single Document.
        delimiter: CSV field delimiter.

    Example:
        >>> loader = CSVLoader(content_columns=["title", "body"])
        >>> docs = loader.load("articles.csv")
    """

    supported_extensions: set[str] = {".csv"}

    def __init__(
        self,
        encoding: str = "utf-8",
        content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        one_doc_per_row: bool = True,
        delimiter: str = ",",
    ) -> None:
        self.encoding = encoding
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.one_doc_per_row = one_doc_per_row
        self.delimiter = delimiter

    def load(self, path: str | Path) -> list[Document]:
        """Load a CSV file as one or more Documents.

        Args:
            path: Path to the CSV file.

        Returns:
            A list of Document objects extracted from the CSV.
        """
        path = self._validate_file(path)

        with open(path, encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            rows = list(reader)

        if not rows:
            return []

        fieldnames = list(rows[0].keys())

        content_cols = self.content_columns or fieldnames
        meta_cols = self.metadata_columns or [
            c for c in fieldnames if c not in content_cols
        ]

        if self.one_doc_per_row:
            documents = []
            for i, row in enumerate(rows):
                content_parts = [
                    f"{col}: {row.get(col, '')}" for col in content_cols if row.get(col)
                ]
                content = "\n".join(content_parts)

                if not content.strip():
                    continue

                metadata = {col: row.get(col, "") for col in meta_cols}
                metadata["source"] = str(path)
                metadata["file_type"] = "csv"
                metadata["row_index"] = i

                documents.append(
                    Document(
                        content=content,
                        metadata=metadata,
                        source=str(path),
                    )
                )
            return documents
        else:
            all_parts = []
            for row in rows:
                parts = [
                    f"{col}: {row.get(col, '')}" for col in content_cols if row.get(col)
                ]
                all_parts.append(" | ".join(parts))

            content = "\n".join(all_parts)
            return [
                Document(
                    content=content,
                    metadata={
                        "source": str(path),
                        "file_type": "csv",
                        "row_count": len(rows),
                    },
                    source=str(path),
                )
            ]
