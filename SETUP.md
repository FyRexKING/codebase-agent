# Setup Guide (Local)

This guide describes a clean local setup for the Codebase RAG Agent on Linux/macOS/Windows (WSL).

## Prerequisites

- Python 3.10+ (recommended)
- Docker (for Qdrant)
- A Gemini API key

## 1) Clone and create a virtual environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Start Qdrant (with persistent storage)

From any directory, start Qdrant with a mounted volume so data survives container recreation:

```bash
docker run --rm \
  -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Verify Qdrant is up:

```bash
curl -sS http://localhost:6333/collections
```

## 3) Configure environment variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=YOUR_KEY_HERE
QDRANT_HOST=localhost
QDRANT_PORT=6333
DATA_DIR=data
```

Notes:
- You must **restart** the backend after changing `.env`.
- `DATA_DIR` is used for the cloned repo workspace and the incremental index state file.

## 4) Run the backend

From the project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:
- UI: `http://127.0.0.1:8000/static/index.html`
- API root: `http://127.0.0.1:8000/`

## 5) Ingest a repository

Use the UI, or call the API directly.

Incremental ingest (default):

```bash
curl -X POST "http://127.0.0.1:8000/ingest?repo_url=https://github.com/tartley/colorama"
```

Force a full rebuild:

```bash
curl -X POST "http://127.0.0.1:8000/ingest?repo_url=https://github.com/tartley/colorama&force_full=true"
```

After ingest, the backend persists per-repo commit state at:
- `data/index_state.json`

## 6) Ask questions

```bash
curl -G "http://127.0.0.1:8000/ask" \
  --data-urlencode "repo_url=https://github.com/tartley/colorama" \
  --data-urlencode "query=What does colorama.init() do, step by step, and which streams does it wrap?"
```

## Common issues

## Gemini 429 (quota / rate limit)

Symptoms:
- Ingest or ask returns an error with `error_type=GEMINI_429_QUOTA`.

Fixes:
- Use smaller repos (fewer chunks).
- Wait 30–60 seconds and retry.
- Reduce repeated ingests.

## GitHub repo not found / private repo

Symptoms:
- Ingest returns `GITHUB_REPO_NOT_FOUND` or `GITHUB_AUTH_FAILED`.

Fixes:
- Use a public repository URL for the demo.
- If you need private repos, add authentication to the clone step (PAT/SSH).

## Qdrant connection errors

Symptoms:
- Ingest fails with connection errors to Qdrant.

Fixes:
- Confirm Qdrant is running on `localhost:6333`.
- Ensure `QDRANT_HOST`/`QDRANT_PORT` match your environment.

