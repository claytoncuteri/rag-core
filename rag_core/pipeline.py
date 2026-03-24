"""Main RAG pipeline orchestrator.

Provides the RAGPipeline class that ties together chunking, embedding,
storage, and retrieval into a single interface.
"""

from __future__ import annotations

from typing import Literal

from rag_core.chunkers.base import BaseChunker
from rag_core.chunkers.fixed import FixedSizeChunker
from rag_core.chunkers.recursive import RecursiveChunker
from rag_core.chunkers.semantic import SemanticChunker
from rag_core.embeddings.base import EmbeddingProvider
from rag_core.exceptions import ChunkingError, EmbeddingError, PipelineError
from rag_core.models import Chunk, Document, RAGResponse
from rag_core.prompts.builder import PromptBuilder
from rag_core.retrieval.retriever import Retriever
from rag_core.stores.base import VectorStore


class RAGPipeline:
    """End-to-end RAG pipeline for document ingestion and querying.

    Orchestrates the full retrieval-augmented generation workflow: chunking
    documents, computing embeddings, storing vectors, and retrieving relevant
    context for a given query.

    Args:
        embedding_provider: The embedding provider to use for vectorization.
        store: The vector store backend for persisting and searching embeddings.
        chunk_strategy: Chunking strategy name. One of "fixed", "semantic",
            or "recursive".
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        metric: Similarity metric for retrieval. Currently only "cosine"
            is supported.

    Example:
        >>> from rag_core.embeddings import LocalEmbeddings
        >>> from rag_core.stores import InMemoryStore
        >>> pipeline = RAGPipeline(
        ...     embedding_provider=LocalEmbeddings(),
        ...     store=InMemoryStore(),
        ... )
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: VectorStore,
        chunk_strategy: Literal["fixed", "semantic", "recursive"] = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        metric: Literal["cosine"] = "cosine",
    ) -> None:
        self.embedding_provider = embedding_provider
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metric = metric

        self._chunker = self._build_chunker(chunk_strategy, chunk_size, chunk_overlap)
        self._retriever = Retriever(
            store=store,
            embedding_provider=embedding_provider,
        )
        self._prompt_builder = PromptBuilder()
        self._chunks: list[Chunk] = []

    @staticmethod
    def _build_chunker(
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> BaseChunker:
        """Create a chunker instance from a strategy name.

        Args:
            strategy: One of "fixed", "semantic", or "recursive".
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            A configured BaseChunker subclass instance.

        Raises:
            ValueError: If the strategy name is not recognized.
        """
        chunkers: dict[str, type[BaseChunker]] = {
            "fixed": FixedSizeChunker,
            "semantic": SemanticChunker,
            "recursive": RecursiveChunker,
        }
        if strategy not in chunkers:
            raise ChunkingError(
                f"Unknown chunk strategy '{strategy}'. "
                f"Choose from: {', '.join(chunkers.keys())}"
            )
        return chunkers[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest(self, documents: list[Document]) -> int:
        """Ingest documents into the pipeline.

        Chunks each document, computes embeddings, and stores the vectors
        for later retrieval.

        Args:
            documents: List of Document objects to ingest.

        Returns:
            The total number of chunks created and stored.

        Raises:
            PipelineError: If chunking, embedding, or storage fails.
        """
        try:
            all_chunks: list[Chunk] = []
            for doc in documents:
                chunks = self._chunker.chunk(doc)
                all_chunks.extend(chunks)

            if not all_chunks:
                return 0

            texts = [c.text for c in all_chunks]
        except Exception as exc:
            raise PipelineError(f"Failed to chunk documents: {exc}") from exc

        try:
            embeddings = self.embedding_provider.embed(texts)
        except Exception as exc:
            raise EmbeddingError(f"Failed to compute embeddings: {exc}") from exc

        ids = [c.chunk_id for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]

        try:
            self.store.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
            )
        except Exception as exc:
            raise PipelineError(f"Failed to store vectors: {exc}") from exc

        self._chunks.extend(all_chunks)
        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
        template_name: str = "qa",
    ) -> RAGResponse:
        """Query the pipeline with a natural language question.

        Retrieves the most relevant chunks and assembles a RAGResponse
        containing the retrieved context, sources, and a confidence score.

        Args:
            question: The user's question or query string.
            top_k: Number of top results to retrieve.
            template_name: The prompt template to use for building context.

        Returns:
            A RAGResponse with retrieved chunks, sources, and confidence.

        Raises:
            PipelineError: If retrieval or prompt building fails.
        """
        try:
            results = self._retriever.retrieve(query=question, top_k=top_k)
        except Exception as exc:
            raise PipelineError(f"Retrieval failed for query: {exc}") from exc

        if not results:
            return RAGResponse(
                answer="",
                sources=[],
                confidence_score=0.0,
                retrieved_chunks=[],
            )

        retrieved_chunks: list[Chunk] = []
        sources: list[str] = []
        scores: list[float] = []

        chunk_lookup = {c.chunk_id: c for c in self._chunks}

        for result in results:
            chunk_id = result["id"]
            score = result["score"]
            scores.append(score)

            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                retrieved_chunks.append(chunk)
                source = chunk.metadata.get("source", chunk.doc_id)
                if source and source not in sources:
                    sources.append(source)
            else:
                # Build a chunk from the store result
                chunk = Chunk(
                    text=result.get("document", ""),
                    metadata=result.get("metadata", {}),
                    chunk_id=chunk_id,
                )
                retrieved_chunks.append(chunk)

        confidence = sum(scores) / len(scores) if scores else 0.0

        prompt = self._prompt_builder.build(
            question=question,
            chunks=retrieved_chunks,
            template_name=template_name,
        )

        return RAGResponse(
            answer=prompt,
            sources=sources,
            confidence_score=confidence,
            retrieved_chunks=retrieved_chunks,
        )

    def clear(self) -> None:
        """Remove all stored data from the pipeline."""
        self.store.clear()
        self._chunks.clear()
