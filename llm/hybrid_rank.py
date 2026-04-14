import re
from collections import Counter
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


def _graph_proximity_scores(candidates: List[Dict[str, Any]], *, seed_k: int) -> List[float]:
    if not candidates:
        return []

    seeds = sorted(candidates, key=lambda c: c["semantic_score"], reverse=True)[:seed_k]

    seed_paths = {s.get("path") for s in seeds if s.get("path")}
    seed_modules = {s.get("module") for s in seeds if s.get("module")}
    seed_symbol_names = {s.get("symbol_name") for s in seeds if s.get("symbol_name")}
    seed_dependencies: Counter[str] = Counter()
    seed_imports: Counter[str] = Counter()
    for s in seeds:
        for dep in s.get("dependencies", []):
            seed_dependencies[dep] += 1
        for imp in s.get("imports", []):
            seed_imports[imp] += 1

    scores: List[float] = []
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


def hybrid_rerank(
    candidates: List[Dict[str, Any]],
    *,
    query: str,
    semantic_weight: float,
    bm25_weight: float,
    graph_weight: float,
    graph_seed_k: int,
    threshold: float,
    final_k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    tokenized_docs = [_tokenize(c.get("full_text", "")) for c in candidates]
    tokenized_query = _tokenize(query)

    if tokenized_query and any(tokenized_docs):
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = list(bm25.get_scores(tokenized_query))
    else:
        bm25_scores = [0.0] * len(candidates)

    semantic_scores = [float(c.get("semantic_score", 0.0) or 0.0) for c in candidates]
    graph_scores = _graph_proximity_scores(candidates, seed_k=graph_seed_k)

    semantic_norm = _min_max_normalize(semantic_scores)
    bm25_norm = _min_max_normalize(bm25_scores)
    graph_norm = _min_max_normalize(graph_scores)

    for idx, candidate in enumerate(candidates):
        hybrid = (
            (semantic_weight * semantic_norm[idx])
            + (bm25_weight * bm25_norm[idx])
            + (graph_weight * graph_norm[idx])
        )
        candidate["bm25_score"] = float(bm25_scores[idx])
        candidate["graph_score"] = float(graph_scores[idx])
        candidate["hybrid_score"] = float(hybrid)

    ranked = sorted(candidates, key=lambda c: c["hybrid_score"], reverse=True)
    filtered = [c for c in ranked if c["hybrid_score"] >= threshold]
    return (filtered or ranked)[:final_k]

