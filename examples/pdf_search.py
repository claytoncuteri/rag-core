"""PDF search example using rag-core.

This example shows how to:
1. Load PDF documents from a directory
2. Index them with embeddings
3. Search across all PDFs with natural language queries

Requirements:
    pip install rag-core[local,pdf]

Usage:
    python pdf_search.py /path/to/pdf/directory "your search query"
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag_core import RAGPipeline
from rag_core.embeddings.local_embeddings import LocalEmbeddings
from rag_core.loaders.pdf import PDFLoader
from rag_core.stores.memory import InMemoryStore


def main(pdf_dir: str, query: str) -> None:
    """Load PDFs from a directory and search them.

    Args:
        pdf_dir: Path to a directory containing PDF files.
        query: The search query string.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.is_dir():
        print(f"Error: '{pdf_dir}' is not a valid directory.")
        sys.exit(1)

    # Load PDFs
    print(f"Loading PDFs from: {pdf_dir}")
    loader = PDFLoader(pages_as_documents=True)
    documents = loader.load_directory(pdf_dir)

    if not documents:
        print("No PDF documents found in the directory.")
        sys.exit(1)

    print(f"Loaded {len(documents)} pages from PDF files.\n")

    # Set up the pipeline
    print("Building search index with local embeddings...")
    pipeline = RAGPipeline(
        embedding_provider=LocalEmbeddings(),
        store=InMemoryStore(),
        chunk_strategy="recursive",
        chunk_size=400,
        chunk_overlap=50,
    )

    chunk_count = pipeline.ingest(documents)
    print(f"Indexed {chunk_count} chunks.\n")

    # Search
    print(f"Searching for: \"{query}\"\n")
    response = pipeline.query(query, top_k=5)

    if not response.retrieved_chunks:
        print("No relevant results found.")
        return

    print(f"Found {len(response.retrieved_chunks)} results "
          f"(confidence: {response.confidence_score:.3f}):\n")

    for i, chunk in enumerate(response.retrieved_chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page_number", "?")
        preview = chunk.text[:200].replace("\n", " ")
        print(f"  [{i}] {source} (page {page})")
        print(f"      {preview}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdf_search.py <pdf_directory> <query>")
        print('Example: python pdf_search.py ./papers "neural network architecture"')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
