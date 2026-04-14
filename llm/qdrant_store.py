import hashlib
import uuid
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from llm.config import QDRANT_HOST, QDRANT_PORT
from llm.gemini_embedder import VECTOR_SIZE

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def collection_name_for_repo(repo_url: str) -> str:
    normalized = (repo_url or "").strip().lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"repo_{digest}"


def point_id_for_chunk(chunk: Dict[str, Any]) -> str:
    repo_url = (chunk.get("repo_url") or "").strip().lower()
    path = (chunk.get("path") or "").replace("\\", "/")
    symbol_id = (chunk.get("symbol_id") or "").strip()
    start_line = chunk.get("start_line")
    end_line = chunk.get("end_line")

    key = f"{repo_url}|{path}|{symbol_id}|{start_line}|{end_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def ensure_collection_exists(collection_name: str) -> None:
    qdrant = get_qdrant_client()
    existing = {c.name for c in qdrant.get_collections().collections}
    if collection_name in existing:
        return
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def reset_collection(collection_name: str) -> None:
    qdrant = get_qdrant_client()
    existing = {c.name for c in qdrant.get_collections().collections}
    if collection_name in existing:
        qdrant.delete_collection(collection_name=collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def upsert_points(collection_name: str, points: Iterable[PointStruct]) -> None:
    qdrant = get_qdrant_client()
    qdrant.upsert(collection_name=collection_name, points=list(points))


def search_points(
    collection_name: str, *, query_vector: List[float], limit: int
) -> List[Any]:
    qdrant = get_qdrant_client()
    return qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
    )

