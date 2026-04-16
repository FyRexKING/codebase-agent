# Codebase RAG Agent

Codebase RAG Agent is a lightweight **repository question-answering** service built around Retrieval-Augmented Generation (RAG). It clones a GitHub repository, chunks Python code by AST symbols, embeds the chunks, stores them in Qdrant, and answers questions using Gemini with grounded citations.

## What it supports

- **Python-only indexing** (current scope)
- **AST-aware chunking** (module/class/function/method) with line ranges and symbol metadata
- **Vector search** (Qdrant) with **hybrid re-ranking** (semantic + BM25 + lightweight import/dependency proximity)
- **Incremental indexing** on re-ingest using `git diff` (only changed files are re-embedded)
- **Deterministic chunk IDs** so re-indexing performs stable upserts
- **Grounded answers**: responses are instructed to cite sources and refuse when evidence is insufficient
- **User-friendly errors** for common failures (Gemini 429 quota, GitHub repo not found/private)

## Architecture (high level)

- **Ingest**: clone → load `.py` files → AST chunking → Gemini embeddings → Qdrant upsert
- **Query**: embed query → Qdrant semantic candidates → hybrid re-rank → Gemini answer from curated sources

## Quickstart

See `SETUP.md` for a complete, reproducible setup.

At minimum:

1. Start Qdrant (Docker) and keep it running.
2. Create `.env` with `GEMINI_API_KEY=...`.
3. Run the backend:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

4. Open the UI:
- `http://127.0.0.1:8000/static/index.html`

## API

- `POST /ingest?repo_url=<url>`
  - Performs **incremental indexing** if the repo was indexed before.
- `POST /ingest?repo_url=<url>&force_full=true`
  - Forces a **full rebuild** (drops the repo collection and re-indexes everything).
- `GET /ask?repo_url=<url>&query=<question>`
  - Answers a question using retrieved sources from that repo’s collection.

Ingestion responses include:
- `mode`: `initial_full` | `incremental` | `full`
- `chunks_indexed`
- `head_commit`, `base_commit`
- `changed_files`, `deleted_files`

## Index state (incremental indexing)

The backend stores per-repo index state at:
- `data/index_state.json`

This is used to compute diffs between the last indexed commit and the current HEAD on re-ingest.

## Configuration

Environment variables (via `.env`):
- `GEMINI_API_KEY` (required)
- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `DATA_DIR` (default `data`)

## Limitations

- Indexer currently processes **only Python** (`.py`) files.
- Repository workspace is currently **single-directory** (`data/repo`), so concurrent ingests are not supported.
- Gemini quota limits may require using smaller repos and/or fewer chunks (errors are surfaced with actionable hints).

## Troubleshooting

- **Gemini 429 / quota exceeded**
  - Use smaller repos, retry after a short wait, or reduce repeated ingests.
- **GitHub repo not found / auth failed**
  - This demo clones anonymously; private repos will fail unless you add authentication to the clone step.

## Project layout

- `backend/`: FastAPI service and static file serving
- `llm/`: ingestion, chunking, embeddings, retrieval, ranking, and answer generation
- `frontend/`: simple HTML/CSS/JS UI
- `data/`: local working directory and index state
