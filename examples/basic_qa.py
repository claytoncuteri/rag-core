"""Basic question-answering example using rag-core.

This example shows how to:
1. Create documents from raw text
2. Set up a pipeline with local embeddings
3. Ingest documents
4. Query the pipeline with questions

Requirements:
    pip install rag-core[local]
"""

from rag_core import Document, RAGPipeline
from rag_core.embeddings.local_embeddings import LocalEmbeddings
from rag_core.stores.memory import InMemoryStore


def main() -> None:
    """Run a basic QA pipeline on sample documents."""

    # Create some sample documents
    documents = [
        Document(
            content=(
                "Retrieval-Augmented Generation (RAG) is an approach that combines "
                "information retrieval with text generation. Instead of relying solely "
                "on a language model's training data, RAG retrieves relevant documents "
                "from a knowledge base and uses them as context for generating answers. "
                "This makes responses more accurate and grounded in source material."
            ),
            metadata={"topic": "RAG", "source": "rag_overview"},
            source="rag_overview",
        ),
        Document(
            content=(
                "Vector embeddings are numerical representations of text that capture "
                "semantic meaning. Similar texts produce similar vectors, allowing "
                "efficient similarity search. Common embedding models include OpenAI's "
                "text-embedding-3-small and the open-source all-MiniLM-L6-v2 from "
                "sentence-transformers."
            ),
            metadata={"topic": "embeddings", "source": "embeddings_guide"},
            source="embeddings_guide",
        ),
        Document(
            content=(
                "Chunking is the process of splitting large documents into smaller "
                "pieces for embedding and retrieval. Good chunking preserves semantic "
                "meaning within each chunk. Common strategies include fixed-size "
                "splitting, recursive splitting on natural boundaries, and semantic "
                "splitting based on topic changes."
            ),
            metadata={"topic": "chunking", "source": "chunking_guide"},
            source="chunking_guide",
        ),
    ]

    # Set up the pipeline
    print("Setting up RAG pipeline with local embeddings...")
    pipeline = RAGPipeline(
        embedding_provider=LocalEmbeddings(),
        store=InMemoryStore(),
        chunk_strategy="recursive",
        chunk_size=300,
        chunk_overlap=30,
    )

    # Ingest documents
    chunk_count = pipeline.ingest(documents)
    print(f"Ingested {len(documents)} documents into {chunk_count} chunks.\n")

    # Ask questions
    questions = [
        "What is RAG?",
        "How do vector embeddings work?",
        "What are the different chunking strategies?",
    ]

    for question in questions:
        print(f"Q: {question}")
        response = pipeline.query(question, top_k=2)
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Sources: {', '.join(response.sources)}")
        print(f"Chunks retrieved: {len(response.retrieved_chunks)}")
        print()


if __name__ == "__main__":
    main()
