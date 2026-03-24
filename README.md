# rag-core

[![CI](https://github.com/claytoncuteri/rag-core/actions/workflows/ci.yml/badge.svg)](https://github.com/claytoncuteri/rag-core/actions/workflows/ci.yml)

A lightweight, modular RAG pipeline library for Python.

## Installation

```bash
# Core library (numpy only)
pip install rag-core

# With OpenAI embeddings
pip install rag-core[openai]

# With local sentence-transformer embeddings
pip install rag-core[local]

# With ChromaDB vector store
pip install rag-core[chroma]

# With PDF loading support
pip install rag-core[pdf]

# Everything
pip install rag-core[all]
```

For development:

```bash
git clone https://github.com/claytoncuteri/rag-core.git
cd rag-core
pip install -e ".[all,dev]"
```

## Quick Start

```python
from rag_core import RAGPipeline
from rag_core.loaders import TextLoader
from rag_core.embeddings import LocalEmbeddings
from rag_core.stores import InMemoryStore

# Set up the pipeline
pipeline = RAGPipeline(
    embedding_provider=LocalEmbeddings(),
    store=InMemoryStore(),
    chunk_strategy="recursive",
    chunk_size=500,
    chunk_overlap=50,
)

# Load and ingest documents
loader = TextLoader()
docs = loader.load_directory("./my_docs")
pipeline.ingest(docs)

# Ask a question
response = pipeline.query("What is the main topic?")
print(response.answer)
print(f"Confidence: {response.confidence_score:.2f}")
for source in response.sources:
    print(f"  - {source}")
```

## Architecture

```
                        RAG Pipeline
 +------------------------------------------------------------+
 |                                                            |
 |  Documents --> Loader --> Chunker --> Embeddings --> Store  |
 |                                                            |
 |  Query --> Embed Query --> Retrieve --> Rank --> Response   |
 |                                                            |
 +------------------------------------------------------------+
        |            |            |            |
    +--------+  +--------+  +---------+  +--------+
    | Loaders|  |Chunkers|  |Embedding|  | Stores |
    |--------|  |--------|  |Providers|  |--------|
    | Text   |  | Fixed  |  |---------|  | Memory |
    | MD     |  |Semantic|  | OpenAI  |  | Chroma |
    | PDF    |  |Recursive| | Local   |  +--------+
    | CSV    |  +--------+  +---------+
    +--------+
```

Each component is pluggable. Swap out any layer without changing the rest
of the pipeline.

## Design Decisions

### Why recursive chunking as the default?

Recursive chunking tries to split on the most meaningful boundaries first
(double newlines, then single newlines, then sentence endings, then raw
character count). This preserves semantic coherence in each chunk while
still guaranteeing a maximum chunk size. Fixed-size chunking is faster
but often splits mid-sentence, and purely semantic chunking can produce
unpredictable chunk sizes.

### Why cosine similarity?

Cosine similarity measures the angle between two vectors, making it
invariant to magnitude. This is ideal for comparing embeddings because
two texts about the same topic will point in similar directions regardless
of document length. Vectors are normalized at insert time so that cosine
similarity reduces to a simple dot product, which is fast with numpy.

### Local vs. API embeddings

API embeddings (OpenAI) offer higher quality and require no GPU, but they
add latency and cost. Local embeddings (sentence-transformers) are free,
run offline, and are fast enough for most use cases. The library supports
both so you can choose the right tradeoff for your project.

### Embedding cache

Computing embeddings is the most expensive step in the pipeline. The
EmbeddingCache stores results keyed by a hash of the text and model name,
and can persist to disk as `.npz` files. This avoids redundant API calls
and speeds up re-ingestion of unchanged documents.

## Components

### Loaders

Load documents from various file formats into a standard `Document` object.

- `TextLoader` - Plain text files (.txt)
- `MarkdownLoader` - Markdown files (.md), strips formatting
- `PDFLoader` - PDF files via pypdf (requires `rag-core[pdf]`)
- `CSVLoader` - CSV files, one document per row or per file

### Chunkers

Split documents into smaller chunks for embedding and retrieval.

- `FixedSizeChunker` - Split by character count with configurable overlap
- `SemanticChunker` - Split on paragraph breaks, headers, and sentence groups
- `RecursiveChunker` - Try progressively finer separators (default)

### Embedding Providers

Convert text to vector representations.

- `OpenAIEmbeddings` - Uses text-embedding-3-small (requires `rag-core[openai]`)
- `LocalEmbeddings` - Uses all-MiniLM-L6-v2 via sentence-transformers (requires `rag-core[local]`)
- `EmbeddingCache` - Wraps any provider with hash-based caching

### Vector Stores

Store and search embedding vectors.

- `InMemoryStore` - Pure numpy, good for small to medium datasets
- `ChromaStore` - Wrapper around ChromaDB for persistence (requires `rag-core[chroma]`)

### Retrieval

- `Retriever` - Combines a store and embedding provider for top-k retrieval
- `Ranker` - Re-scores results using metadata boosters (recency, source authority, diversity)

### Prompts

- `PromptBuilder` - Assembles prompts from retrieved chunks and question
- Built-in templates for QA, summarization, and comparison tasks

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[all,dev]"`
4. Make your changes and add tests
5. Run the test suite: `pytest`
6. Run the linter: `ruff check .`
7. Submit a pull request

Please ensure all tests pass and code follows the existing style (Google-style
docstrings, type hints on all public functions).

## License

MIT License. See [LICENSE](LICENSE) for details.
