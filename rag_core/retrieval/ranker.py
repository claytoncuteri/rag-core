"""Re-ranking module for boosting retrieval results with metadata signals."""

from __future__ import annotations

import time
from typing import Any


class Ranker:
    """Re-scores retrieval results using metadata-based boosters.

    Applies adjustable boosts for recency, source authority, and diversity
    to refine the ordering of retrieved results beyond raw similarity scores.

    Args:
        recency_weight: Weight for the recency boost (0.0 to 1.0).
            Documents with more recent timestamps get higher boosts.
        authority_weight: Weight for the source authority boost (0.0 to 1.0).
            Documents from authoritative sources get higher boosts.
        diversity_weight: Weight for the diversity penalty (0.0 to 1.0).
            Penalizes results from sources already represented in the
            top results.
        authority_sources: Set of source identifiers considered authoritative.

    Example:
        >>> ranker = Ranker(
        ...     recency_weight=0.1,
        ...     authority_sources={"official_docs", "api_reference"},
        ... )
        >>> reranked = ranker.rerank(results)
    """

    def __init__(
        self,
        recency_weight: float = 0.1,
        authority_weight: float = 0.1,
        diversity_weight: float = 0.05,
        authority_sources: set[str] | None = None,
    ) -> None:
        self.recency_weight = recency_weight
        self.authority_weight = authority_weight
        self.diversity_weight = diversity_weight
        self.authority_sources = authority_sources or set()

    def rerank(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-score and re-sort retrieval results.

        Applies recency, authority, and diversity adjustments to the base
        similarity scores, then returns results sorted by the adjusted score.

        Args:
            results: List of result dicts from a Retriever, each with
                "id", "score", "metadata", and "document" keys.

        Returns:
            The same results list, re-sorted by adjusted scores. Each result
            gets an additional "adjusted_score" key.
        """
        if not results:
            return results

        scored = []
        seen_sources: dict[str, int] = {}

        for result in results:
            base_score = result.get("score", 0.0)
            metadata = result.get("metadata", {})

            recency_boost = self._recency_boost(metadata)
            authority_boost = self._authority_boost(metadata)

            source = metadata.get("source", "")
            diversity_penalty = self._diversity_penalty(source, seen_sources)

            adjusted = (
                base_score
                + self.recency_weight * recency_boost
                + self.authority_weight * authority_boost
                - self.diversity_weight * diversity_penalty
            )

            result_copy = dict(result)
            result_copy["adjusted_score"] = adjusted
            scored.append(result_copy)

            seen_sources[source] = seen_sources.get(source, 0) + 1

        scored.sort(key=lambda r: r["adjusted_score"], reverse=True)
        return scored

    def _recency_boost(self, metadata: dict[str, Any]) -> float:
        """Compute a recency boost from metadata timestamps.

        Expects a "timestamp" key in metadata as a Unix epoch float.
        Returns a value between 0.0 and 1.0, where more recent documents
        score higher.

        Args:
            metadata: The chunk's metadata dict.

        Returns:
            A recency boost value between 0.0 and 1.0.
        """
        timestamp = metadata.get("timestamp")
        if timestamp is None:
            return 0.0

        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return 0.0

        now = time.time()
        age_days = (now - ts) / 86400.0

        if age_days < 0:
            return 1.0
        if age_days > 365:
            return 0.0

        # Linear decay over one year
        return max(0.0, 1.0 - (age_days / 365.0))

    def _authority_boost(self, metadata: dict[str, Any]) -> float:
        """Compute an authority boost based on the document source.

        Args:
            metadata: The chunk's metadata dict.

        Returns:
            1.0 if the source is in the authority set, 0.0 otherwise.
        """
        source = metadata.get("source", "")
        return 1.0 if source in self.authority_sources else 0.0

    @staticmethod
    def _diversity_penalty(source: str, seen_sources: dict[str, int]) -> float:
        """Compute a diversity penalty for over-represented sources.

        Args:
            source: The source identifier for the current result.
            seen_sources: Dict tracking how many times each source has
                appeared in the results so far.

        Returns:
            A penalty value. Higher values indicate the source is already
            well-represented.
        """
        count = seen_sources.get(source, 0)
        return float(count)
