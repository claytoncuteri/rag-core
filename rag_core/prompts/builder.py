"""Prompt builder that assembles context and question into a formatted prompt."""

from __future__ import annotations

from rag_core.models import Chunk
from rag_core.prompts.templates import TEMPLATES


class PromptBuilder:
    """Builds formatted prompts from retrieved chunks and a question.

    Combines a prompt template with retrieved context chunks and the user's
    question to produce a ready-to-use prompt string.

    Args:
        custom_templates: Optional dict of additional templates to register.
            Keys are template names, values are template strings with
            {context}, {question}, and {instructions} placeholders.

    Example:
        >>> builder = PromptBuilder()
        >>> prompt = builder.build(
        ...     question="What is RAG?",
        ...     chunks=my_chunks,
        ...     template_name="qa",
        ... )
    """

    def __init__(
        self,
        custom_templates: dict[str, str] | None = None,
    ) -> None:
        self._templates = dict(TEMPLATES)
        if custom_templates:
            self._templates.update(custom_templates)

    def build(
        self,
        question: str,
        chunks: list[Chunk],
        template_name: str = "qa",
        instructions: str = "",
    ) -> str:
        """Build a prompt from a question, chunks, and template.

        Args:
            question: The user's question or query.
            chunks: List of retrieved Chunk objects to use as context.
            template_name: Name of the template to use.
            instructions: Optional additional instructions to include.

        Returns:
            A formatted prompt string ready for an LLM.

        Raises:
            ValueError: If the template name is not recognized.
        """
        if template_name not in self._templates:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {', '.join(self._templates.keys())}"
            )

        context = self._format_context(chunks)
        template = self._templates[template_name]

        return template.format(
            context=context,
            question=question,
            instructions=instructions,
        ).strip()

    @staticmethod
    def _format_context(chunks: list[Chunk]) -> str:
        """Format a list of chunks into a numbered context string.

        Args:
            chunks: List of Chunk objects.

        Returns:
            A formatted string with each chunk numbered and separated
            by blank lines.
        """
        if not chunks:
            return "(No context available)"

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[{i}] (Source: {source})\n{chunk.text}")

        return "\n\n".join(parts)

    def register_template(self, name: str, template: str) -> None:
        """Register a new prompt template.

        Args:
            name: The name to register the template under.
            template: The template string with {context}, {question},
                and {instructions} placeholders.
        """
        self._templates[name] = template

    def list_templates(self) -> list[str]:
        """Return the names of all registered templates.

        Returns:
            A sorted list of template names.
        """
        return sorted(self._templates.keys())
