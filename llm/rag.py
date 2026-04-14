from typing import Any, Dict, List

from qdrant_client.models import PointStruct

from llm.answer import explain
from llm.config import COLLECTION_NAME
from llm.gemini_embedder import embed_documents, embed_query
from llm.hybrid_rank import hybrid_rerank
from llm.qdrant_store import (
    collection_name_for_repo,
    ensure_collection_exists,
    point_id_for_chunk,
    reset_collection,
    search_points,
    upsert_points,
)

SEMANTIC_WEIGHT = 0.55
BM25_WEIGHT = 0.25
GRAPH_WEIGHT = 0.20
SEMANTIC_CANDIDATE_K = 30
FINAL_CONTEXT_K = 8
HYBRID_SCORE_THRESHOLD = 0.2
GRAPH_SEED_K = 6


def _is_not_found_explanation(explanation: str) -> bool:
    """
    The answer model is instructed to reply exactly "Not found in codebase."
    when it cannot answer from sources, but we also handle minor variants and
    the local "no context" string for robustness.
    """
    t = (explanation or "").strip().lower()
    return t in {
        "not found in codebase.",
        "not found in codebase",
        "no relevant context found in codebase.",
        "no relevant context found in codebase",
    }

def index_chunks(chunks, *, collection_name: str | None = None):
    """Index code chunks into Qdrant vector database"""
    if not chunks:
        return

    resolved_collection = collection_name or COLLECTION_NAME
    ensure_collection_exists(resolved_collection)
    
    max_len = 800
    texts = [chunk["text"][:max_len] for chunk in chunks]
    vectors = embed_documents(texts)
    
    points: List[PointStruct] = []
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
    
    upsert_points(resolved_collection, points)

def search(query, *, collection_name: str | None = None):
    """Search using hybrid retrieval (semantic + BM25 + graph proximity)."""
    try:
        resolved_collection = collection_name or COLLECTION_NAME
        ensure_collection_exists(resolved_collection)

        vector = embed_query(query)
        results = search_points(
            resolved_collection, query_vector=vector, limit=SEMANTIC_CANDIDATE_K
        )

        candidates: List[Dict[str, Any]] = []
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

        final_chunks = hybrid_rerank(
            candidates,
            query=query,
            semantic_weight=SEMANTIC_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            graph_weight=GRAPH_WEIGHT,
            graph_seed_k=GRAPH_SEED_K,
            threshold=HYBRID_SCORE_THRESHOLD,
            final_k=FINAL_CONTEXT_K,
        )

        explanation = explain(query, final_chunks)
        if _is_not_found_explanation(explanation):
            return {
                "explanation": explanation,
                "sources": [],
                "retrieval": {
                    "strategy": "hybrid_semantic_bm25_graph",
                    "weights": {
                        "semantic": SEMANTIC_WEIGHT,
                        "bm25": BM25_WEIGHT,
                        "graph": GRAPH_WEIGHT,
                    },
                    "candidate_count": len(candidates),
                    "selected_count": 0,
                    "threshold": HYBRID_SCORE_THRESHOLD,
                },
            }
        return {
            "explanation": explanation,
            "sources": [
                {
                    "source_id": f"S{idx + 1}",
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
                for idx, c in enumerate(final_chunks)
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
