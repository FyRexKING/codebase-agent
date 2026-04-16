import os
import time
from typing import Any, Dict, List, Tuple

from llm.gemini_embedder import get_gemini_client
from llm.errors import CodebaseAgentError, is_gemini_auth_error, is_gemini_quota_error


DEFAULT_ANSWER_MODEL = "gemini-2.5-flash"
DEFAULT_VERIFY_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_ANSWER_MODEL = "gemini-2.5-pro"
DEFAULT_FALLBACK_VERIFY_MODEL = "gemini-2.5-pro"


def _format_sources(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build structured context blocks and return:
    - context_text: for the LLM
    - sources: normalized sources with stable ids (S1, S2, ...)
    """
    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []

    # Prefer longer excerpts for grounding, but keep bounded.
    for idx, c in enumerate(chunks, start=1):
        source_id = f"S{idx}"
        path = c.get("path", "unknown")
        symbol_id = c.get("symbol_id", "")
        symbol_type = c.get("symbol_type", "")
        start_line = c.get("start_line")
        end_line = c.get("end_line")
        full_text = c.get("full_text") or c.get("snippet") or ""

        excerpt = full_text[:1200]
        blocks.append(
            "\n".join(
                [
                    f"[{source_id}]",
                    f"path: {path}",
                    f"symbol: {symbol_id} ({symbol_type})",
                    f"lines: {start_line}-{end_line}",
                    "code:",
                    excerpt,
                ]
            )
        )

        sources.append(
            {
                "source_id": source_id,
                "path": path,
                "symbol_id": symbol_id,
                "symbol_type": symbol_type,
                "start_line": start_line,
                "end_line": end_line,
                "excerpt": excerpt,
            }
        )

    return ("\n\n---\n\n".join(blocks), sources)


def _generate_answer(query: str, *, context: str, model: str) -> str:
    prompt = f"""You are a codebase question-answering assistant.

Rules (non-negotiable):
- Use ONLY the sources provided below. Do not rely on external knowledge.
- Every factual claim MUST end with at least one citation like [S1] or [S2].
- If you cannot answer from the sources, reply exactly: Not found in codebase.
- If the sources are contradictory or insufficient, say Not found in codebase.
- Keep the explanation easy to read for a normal developer.
- Do not paste large code blocks inline with the explanation. Put code into the dedicated section below.

Output format (exact headings):
## Summary
1-3 sentences.

## StepByStep
Numbered steps. Each step must end with citations.

## WrappedStreams
One line stating which streams are wrapped/replaced, with citations.

## KeyCode
Up to 3 short excerpts max. Each excerpt must be in a fenced code block and must start with a one-line label like \"[S2] path: ... lines: a-b\".

Sources:
{context}

Question:
{query}

Answer (with citations):"""

    client = get_gemini_client()

    # Retry transient Gemini overloads (503) with exponential backoff.
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return (response.text or "").strip() or "Not found in codebase."
        except Exception as e:
            last_err = e
            msg = str(e)
            if is_gemini_quota_error(msg):
                raise CodebaseAgentError(
                    code="GEMINI_429_QUOTA",
                    user_message="Gemini quota/rate limit hit while generating the answer (429).",
                    hint="Wait 30–60 seconds and retry. If this keeps happening, ask shorter questions or reduce load, or upgrade quota.",
                    detail=msg,
                )
            if is_gemini_auth_error(msg):
                raise CodebaseAgentError(
                    code="GEMINI_AUTH_FAILED",
                    user_message="Gemini authentication failed while generating the answer.",
                    hint="Check that `GEMINI_API_KEY` is valid and the backend was restarted after updating `.env`.",
                    detail=msg,
                )
            if "503" not in msg and "UNAVAILABLE" not in msg:
                raise
            if attempt < 3:
                time.sleep(1.5 * (2**attempt))

    # Optional fallback model if the primary is overloaded.
    fallback_model = os.environ.get("ANSWER_MODEL_FALLBACK", "").strip() or DEFAULT_FALLBACK_ANSWER_MODEL
    if fallback_model and fallback_model != model:
        response = client.models.generate_content(model=fallback_model, contents=prompt)
        return (response.text or "").strip() or "Not found in codebase."

    raise last_err or RuntimeError("LLM call failed")


def _verify_answer(answer: str, *, context: str, model: str) -> bool:
    """
    Lightweight verifier: returns True only if the answer is fully supported by sources.
    """
    prompt = f"""You are a strict verifier.

Given:
- Sources
- A proposed answer

Task:
- If ANY sentence in the answer is not directly supported by the sources, output exactly: FAIL
- Otherwise output exactly: PASS
Sources:
{context}
Answer:
{answer}
"""
    client = get_gemini_client()
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            verdict = (response.text or "").strip().upper()
            return verdict == "PASS"
        except Exception as e:
            last_err = e
            msg = str(e)
            if is_gemini_quota_error(msg):
                raise CodebaseAgentError(
                    code="GEMINI_429_QUOTA",
                    user_message="Gemini quota/rate limit hit while verifying the answer (429).",
                    hint="Disable verification (`VERIFY_ANSWER=0`) or wait and retry.",
                    detail=msg,
                )
            if is_gemini_auth_error(msg):
                raise CodebaseAgentError(
                    code="GEMINI_AUTH_FAILED",
                    user_message="Gemini authentication failed while verifying the answer.",
                    hint="Check that `GEMINI_API_KEY` is valid and the backend was restarted after updating `.env`.",
                    detail=msg,
                )
            if "503" not in msg and "UNAVAILABLE" not in msg:
                raise
            if attempt < 3:
                time.sleep(1.0 * (2**attempt))
    fallback_model = os.environ.get("VERIFY_MODEL_FALLBACK", "").strip() or DEFAULT_FALLBACK_VERIFY_MODEL
    if fallback_model and fallback_model != model:
        response = client.models.generate_content(model=fallback_model, contents=prompt)
        verdict = (response.text or "").strip().upper()
        return verdict == "PASS"
    raise last_err or RuntimeError("Verifier call failed")
def explain(query: str, chunks: List[Dict[str, Any]], *, model: str = DEFAULT_ANSWER_MODEL) -> str:
    if not chunks:
        return "No relevant context found in codebase."

    try:
        context, _sources = _format_sources(chunks)
        answer = _generate_answer(query, context=context, model=model)

        if os.environ.get("VERIFY_ANSWER", "").strip().lower() in {"1", "true", "yes"}:
            ok = _verify_answer(answer, context=context, model=DEFAULT_VERIFY_MODEL)
            if not ok:
                return "Not found in codebase."
        return answer
    except Exception as e:
        # Keep raw errors out of the "explanation" channel when possible;
        # the API layer can catch CodebaseAgentError and render a friendly message.
        if isinstance(e, CodebaseAgentError):
            raise
        return f"LLM Error: {str(e)}"

