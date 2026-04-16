import os
from typing import List

import google.genai as genai

from llm.errors import CodebaseAgentError, is_gemini_auth_error, is_gemini_quota_error

VECTOR_SIZE = 3072
DEFAULT_EMBED_MODEL = "gemini-embedding-001"

_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise CodebaseAgentError(
            code="GEMINI_API_KEY_MISSING",
            user_message="Gemini API key is not configured.",
            hint="Add `GEMINI_API_KEY=...` to your `.env` (and restart the backend).",
        )

    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _to_float_vector(raw_embedding) -> List[float]:
    value = raw_embedding

    if hasattr(value, "values"):
        value = value.values

    if isinstance(value, dict):
        if "values" in value:
            value = value["values"]
        elif "embedding" in value:
            value = value["embedding"]

    if not isinstance(value, (list, tuple)):
        value = list(value)

    if value and isinstance(value[0], (list, tuple)):
        if len(value) == 1:
            value = value[0]
        else:
            raise ValueError("Embedding vector is nested; expected a flat vector")

    return [float(x) for x in value]


def _extract_vectors(response) -> List[List[float]]:
    items = getattr(response, "embeddings", None)
    if items is None and isinstance(response, dict):
        items = response.get("embeddings")

    if not items:
        raise ValueError("Embedding response missing 'embeddings'")

    return [_to_float_vector(item) for item in items]


def embed_query(text: str, *, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    client = get_gemini_client()
    try:
        response = client.models.embed_content(
            model=model,
            contents=[text],
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        vectors = _extract_vectors(response)
        return vectors[0]
    except Exception as e:
        msg = str(e)
        if is_gemini_quota_error(msg):
            raise CodebaseAgentError(
                code="GEMINI_429_QUOTA",
                user_message="Gemini quota/rate limit hit while embedding the query (429).",
                hint="Wait 30–60 seconds and retry. If this keeps happening, reduce requests or upgrade your Gemini quota.",
                detail=msg,
            )
        if is_gemini_auth_error(msg):
            raise CodebaseAgentError(
                code="GEMINI_AUTH_FAILED",
                user_message="Gemini authentication failed while embedding the query.",
                hint="Check that `GEMINI_API_KEY` is valid and the backend was restarted after updating `.env`.",
                detail=msg,
            )
        raise


def embed_documents(
    texts: List[str], *, model: str = DEFAULT_EMBED_MODEL, batch_size: int = 10
) -> List[List[float]]:
    client = get_gemini_client()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.models.embed_content(
                model=model,
                contents=batch,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            vectors.extend(_extract_vectors(response))
        except Exception as e:
            msg = str(e)
            if is_gemini_quota_error(msg):
                raise CodebaseAgentError(
                    code="GEMINI_429_QUOTA",
                    user_message="Gemini quota/rate limit hit while embedding repository chunks (429).",
                    hint="Try a smaller repository (or index fewer files), wait and retry, or increase your Gemini quota. If you have a hard limit like 100 chunks, keep repos very small.",
                    detail=msg,
                )
            if is_gemini_auth_error(msg):
                raise CodebaseAgentError(
                    code="GEMINI_AUTH_FAILED",
                    user_message="Gemini authentication failed while embedding repository chunks.",
                    hint="Check that `GEMINI_API_KEY` is valid and the backend was restarted after updating `.env`.",
                    detail=msg,
                )
            raise

    if len(vectors) != len(texts):
        raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")

    return vectors

