import uuid
import os
import re
import hashlib
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from llm.config import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT
import google.genai as genai
from rank_bm25 import BM25Okapi

SEMANTIC_WEIGHT = 0.55
BM25_WEIGHT = 0.25
GRAPH_WEIGHT = 0.20
SEMANTIC_CANDIDATE_K = 30
FINAL_CONTEXT_K = 8
HYBRID_SCORE_THRESHOLD = 0.2
GRAPH_SEED_K = 6

_gemini_client = None
_qdrant_client = None

VECTOR_SIZE = 3072


def collection_name_for_repo(repo_url: str) -> str:
    """
    Derive a stable, Qdrant-safe collection name from a repo URL.

    We keep it short and avoid special characters to be safe across Qdrant versions.
    """
    normalized = (repo_url or "").strip().lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"repo_{digest}"

def point_id_for_chunk(chunk: dict) -> str:
    """
    Deterministic point ID for Qdrant upserts.

    This makes re-ingesting the *same repo* idempotent: the same chunk key maps
    to the same Qdrant point id, so upsert updates instead of duplicating.
    """
    repo_url = (chunk.get("repo_url") or "").strip().lower()
    path = (chunk.get("path") or "").replace("\\", "/")
    symbol_id = (chunk.get("symbol_id") or "").strip()
    start_line = chunk.get("start_line")
    end_line = chunk.get("end_line")

    key = f"{repo_url}|{path}|{symbol_id}|{start_line}|{end_line}"
    # Use UUIDv5 so Qdrant ids stay UUID-shaped while remaining deterministic.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _get_gemini_client() -> genai.Client:
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


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client

def _ensure_collection_exists(collection_name: str):
    qdrant = _get_qdrant_client()
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in existing_collections:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

 # Note: do not connect to Qdrant at import time.
 # Collection checks happen inside index/search paths.


def _to_float_vector(raw_embedding):
    """Normalize an embedding payload into a flat list[float]."""
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


def _extract_vectors(response):
    """Extract one or more vectors from google.genai embed response."""
    items = getattr(response, "embeddings", None)
    if items is None and isinstance(response, dict):
        items = response.get("embeddings")

    if not items:
        raise ValueError("Embedding response missing 'embeddings'")

    return [_to_float_vector(item) for item in items]


def _tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _min_max_normalize(scores):
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


def _graph_proximity_scores(candidates):
    if not candidates:
        return []

    seeds = sorted(candidates, key=lambda c: c["semantic_score"], reverse=True)[:GRAPH_SEED_K]

    seed_paths = {s.get("path") for s in seeds if s.get("path")}
    seed_modules = {s.get("module") for s in seeds if s.get("module")}
    seed_symbol_names = {s.get("symbol_name") for s in seeds if s.get("symbol_name")}
    seed_dependencies = Counter()
    seed_imports = Counter()
    for s in seeds:
        for dep in s.get("dependencies", []):
            seed_dependencies[dep] += 1
        for imp in s.get("imports", []):
            seed_imports[imp] += 1

    scores = []
    for c in candidates:
        score = 0.0
        if c.get("path") in seed_paths:
            score += 0.8
        if c.get("module") in seed_modules:
            score += 0.7

        deps = set(c.get("dependencies", []))
        imports = set(c.get("imports", []))
        symbol_name = c.get("symbol_name")

        if deps & seed_symbol_names:
            score += 1.2
        if symbol_name and seed_dependencies.get(symbol_name, 0) > 0:
            score += 1.0

        if imports and seed_imports:
            overlap = sum(1 for i in imports if i in seed_imports)
            score += min(1.0, overlap / 3.0)

        scores.append(score)

    return scores

def embed(text):
    """Embed using Google Gemini 3072-dimensional"""
    gemini_client = _get_gemini_client()
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    vectors = _extract_vectors(response)
    return vectors[0]

def batch_embed(texts, batch_size=10):
    """Embed texts in batches using Google Gemini 3072-dimensional"""
    vectors = []
    gemini_client = _get_gemini_client()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )

        vectors.extend(_extract_vectors(response))

    if len(vectors) != len(texts):
        raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
    
    return vectors

def index_chunks(chunks, *, collection_name: str | None = None):
    """Index code chunks into Qdrant vector database"""
    if not chunks:
        return

    resolved_collection = collection_name or COLLECTION_NAME
    _ensure_collection_exists(resolved_collection)
    qdrant = _get_qdrant_client()
    
    max_len = 800
    texts = [chunk["text"][:max_len] for chunk in chunks]
    vectors = batch_embed(texts)
    
    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "path": chunk.get("path"),
            "text": chunk.get("text"),
            "repo_url": chunk.get("repo_url"),
            "repo_name": chunk.get("repo_name"),
            "module": chunk.get("module"),
            "symbol_name": chunk.get("symbol_name"),
            "symbol_type": chunk.get("symbol_type"),
            "symbol_id": chunk.get("symbol_id"),
            "parent_symbol": chunk.get("parent_symbol"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "imports": chunk.get("imports", []),
            "dependencies": chunk.get("dependencies", []),
        }
        points.append(
            PointStruct(
                id=point_id_for_chunk(chunk),
                vector=vector,
                payload=payload,
            )
        )
    
    qdrant.upsert(
        collection_name=resolved_collection,
        points=points,
    )

def explain(query, chunks):
    """Generate an answer using Gemini based on retrieved chunks"""
    if not chunks:
        return "No relevant context found in codebase."
    
    context = "\n\n".join([c["snippet"] for c in chunks])
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
        gemini_client = _get_gemini_client()
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or "Not found in codebase."
    except Exception as e:
        return f"LLM Error: {str(e)}"

def search(query, *, collection_name: str | None = None):
    """Search using hybrid retrieval (semantic + BM25 + graph proximity)."""
    try:
        resolved_collection = collection_name or COLLECTION_NAME
        _ensure_collection_exists(resolved_collection)
        qdrant = _get_qdrant_client()

        vector = embed(query)
        results = qdrant.search(
            collection_name=resolved_collection,
            query_vector=vector,
            limit=SEMANTIC_CANDIDATE_K,
        )

        candidates = []
        for point in results:
            payload = point.payload
            text = payload.get("text", "")
            candidates.append(
                {
                    "path": payload.get("path", "unknown"),
                    "repo_name": payload.get("repo_name", "unknown"),
                    "module": payload.get("module", ""),
                    "symbol_name": payload.get("symbol_name", ""),
                    "symbol_type": payload.get("symbol_type", ""),
                    "symbol_id": payload.get("symbol_id", ""),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "imports": payload.get("imports", []) or [],
                    "dependencies": payload.get("dependencies", []) or [],
                    "full_text": text,
                    "snippet": text[:300],
                    "semantic_score": float(getattr(point, "score", 0.0) or 0.0),
                }
            )

        if not candidates:
            return {"explanation": "No relevant context found in codebase.", "sources": []}

        tokenized_docs = [_tokenize(c["full_text"]) for c in candidates]
        tokenized_query = _tokenize(query)

        if tokenized_query and any(tokenized_docs):
            bm25 = BM25Okapi(tokenized_docs)
            bm25_scores = list(bm25.get_scores(tokenized_query))
        else:
            bm25_scores = [0.0] * len(candidates)

        semantic_scores = [c["semantic_score"] for c in candidates]
        graph_scores = _graph_proximity_scores(candidates)
        semantic_norm = _min_max_normalize(semantic_scores)
        bm25_norm = _min_max_normalize(bm25_scores)
        graph_norm = _min_max_normalize(graph_scores)

        for idx, candidate in enumerate(candidates):
            hybrid = (
                (SEMANTIC_WEIGHT * semantic_norm[idx])
                + (BM25_WEIGHT * bm25_norm[idx])
                + (GRAPH_WEIGHT * graph_norm[idx])
            )
            candidate["bm25_score"] = float(bm25_scores[idx])
            candidate["graph_score"] = float(graph_scores[idx])
            candidate["hybrid_score"] = float(hybrid)

        ranked = sorted(candidates, key=lambda c: c["hybrid_score"], reverse=True)
        filtered = [c for c in ranked if c["hybrid_score"] >= HYBRID_SCORE_THRESHOLD]
        final_chunks = (filtered or ranked)[:FINAL_CONTEXT_K]

        explanation = explain(query, final_chunks)
        return {
            "explanation": explanation,
            "sources": [
                {
                    "path": c["path"],
                    "repo_name": c["repo_name"],
                    "module": c["module"],
                    "symbol_name": c["symbol_name"],
                    "symbol_type": c["symbol_type"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "snippet": c["snippet"],
                    "semantic_score": round(c["semantic_score"], 4),
                    "bm25_score": round(c["bm25_score"], 4),
                    "graph_score": round(c["graph_score"], 4),
                    "hybrid_score": round(c["hybrid_score"], 4),
                }
                for c in final_chunks
            ],
            "retrieval": {
                "strategy": "hybrid_semantic_bm25_graph",
                "weights": {
                    "semantic": SEMANTIC_WEIGHT,
                    "bm25": BM25_WEIGHT,
                    "graph": GRAPH_WEIGHT,
                },
                "candidate_count": len(candidates),
                "selected_count": len(final_chunks),
                "threshold": HYBRID_SCORE_THRESHOLD,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Search failed: {str(e)}"
        }
