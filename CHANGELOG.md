# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-19

### Added

- **Document Loaders**: Text, Markdown, PDF (via pypdf), and CSV loaders with directory scanning
- **Chunking Strategies**: Fixed-size, semantic (paragraph-aware), and recursive (default) chunkers with configurable size and overlap
- **Embedding Providers**: Local embeddings via sentence-transformers (all-MiniLM-L6-v2) and OpenAI text-embedding-3-small
- **Embedding Cache**: Hash-based caching layer with optional disk persistence (.npz)
- **Vector Stores**: In-memory numpy store with cosine similarity and ChromaDB integration
- **Retrieval**: Top-k retriever with metadata-boosted re-ranking (recency, source authority, diversity)
- **Prompt Builder**: Template-based prompt assembly for QA, summarization, and comparison tasks
- **Pipeline Orchestrator**: Single-interface `RAGPipeline` class tying all components together
- **Configuration**: Dataclass-based `RAGConfig` for pipeline defaults
- **Tests**: Unit tests for chunkers, stores, and end-to-end pipeline
- **Examples**: Basic QA and PDF search examples
