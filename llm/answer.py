from typing import Any, Dict, List

from llm.gemini_embedder import get_gemini_client


DEFAULT_ANSWER_MODEL = "gemini-2.5-flash"


def explain(query: str, chunks: List[Dict[str, Any]], *, model: str = DEFAULT_ANSWER_MODEL) -> str:
    if not chunks:
        return "No relevant context found in codebase."

    context = "\n\n".join([c.get("snippet", "") for c in chunks if c.get("snippet")])
    prompt = f"""You are a code analysis assistant.
Answer strictly using the provided code context.
Do NOT use external knowledge.
If the answer is not clearly present in the code, say: "Not found in codebase."
When answering:
- Refer to function names or logic if visible
- Keep answer concise and grounded

Context:
{context}

Question:
{query}"""

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text or "Not found in codebase."
    except Exception as e:
        return f"LLM Error: {str(e)}"

