import os
from typing import List

import google.genai as genai

VECTOR_SIZE = 3072
DEFAULT_EMBED_MODEL = "gemini-embedding-001"

_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Set it in your environment (or .env) before calling /ask."
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
    response = client.models.embed_content(
        model=model,
        contents=[text],
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    vectors = _extract_vectors(response)
    return vectors[0]


def embed_documents(
    texts: List[str], *, model: str = DEFAULT_EMBED_MODEL, batch_size: int = 10
) -> List[List[float]]:
    client = get_gemini_client()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        vectors.extend(_extract_vectors(response))

    if len(vectors) != len(texts):
        raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")

    return vectors

