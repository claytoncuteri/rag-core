"""Built-in prompt templates for common RAG tasks.

Each template is a string with placeholders for {context}, {question},
and optionally {instructions}. These are filled in by the PromptBuilder.
"""

QA_TEMPLATE = """Answer the question based on the provided context. If the context
does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {question}

{instructions}

Answer:"""

SUMMARIZATION_TEMPLATE = """Summarize the following content. Focus on the key points
and main ideas. Be concise but thorough.

Content:
{context}

{instructions}

Summary:"""

COMPARISON_TEMPLATE = """Compare and contrast the information found in the following
context passages. Highlight similarities, differences, and any notable patterns.

Context:
{context}

Question: {question}

{instructions}

Comparison:"""

TEMPLATES: dict[str, str] = {
    "qa": QA_TEMPLATE,
    "summarization": SUMMARIZATION_TEMPLATE,
    "comparison": COMPARISON_TEMPLATE,
}
