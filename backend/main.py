from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from llm.errors import CodebaseAgentError
from llm.ingest import clone_repo, compute_repo_diff, get_head_commit, load_python_files, chunk_files
from llm.index_state import get_repo_state, set_repo_state
from llm.rag import incremental_index_chunks, index_chunks, search, collection_name_for_repo, reset_collection

# Create API app
app = FastAPI(title="Codebase RAG Agent", description="Semantic search over codebases using RAG")

load_dotenv()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if they exist
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Root endpoint
@app.get("/")
def root():
    return {
        "name": "Codebase RAG Agent",
        "description": "Semantic search over your codebase using RAG",
        "endpoints": {
            "GET /": "This message",
            "POST /ingest": "Ingest a GitHub repo (query params: repo_url, force_full=false)",
            "GET /ask": "Query indexed codebase (query param: query)",
            "GET /static/": "Frontend interface"
        }
    }

# Endpoint to ingest a GitHub repo
@app.post("/ingest")
def ingest(repo_url: str, force_full: bool = False):
    """
    Ingest a GitHub repository and index its Python code.

    - By default, uses incremental indexing (git-diff based) if the repo was indexed before.
    - Set force_full=true to reset and rebuild the index from scratch.
    """
    try:
        # Clone repo
        repo_path = clone_repo(repo_url)
        collection_name = collection_name_for_repo(repo_url)

        # Full rebuild path.
        if force_full:
            files = load_python_files(repo_path)
            chunks = chunk_files(files, repo_url=repo_url)
            reset_collection(collection_name)
            index_chunks(chunks, collection_name=collection_name)
            head = get_head_commit(repo_path)
            set_repo_state(repo_url, last_indexed_commit=head)
            return {
                "status": "success",
                "mode": "full",
                "message": "Repository indexed successfully (full rebuild)",
                "chunks_indexed": len(chunks),
                "head_commit": head,
            }

        # Incremental path.
        prev = get_repo_state(repo_url)
        head = get_head_commit(repo_path)

        # If nothing changed since last time, skip indexing entirely.
        if prev is not None and prev.last_indexed_commit == head:
            return {
                "status": "success",
                "mode": "incremental",
                "message": "Repository already up to date; no indexing needed",
                "chunks_indexed": 0,
                "head_commit": head,
                "base_commit": prev.last_indexed_commit,
                "changed_files": [],
                "deleted_files": [],
            }

        base_commit = prev.last_indexed_commit if prev is not None else None
        diff = compute_repo_diff(repo_path, base_commit=base_commit)

        # Load only changed/added Python files for chunking.
        changed_set = set(diff.changed_or_added_py_abs)
        files = load_python_files(repo_path, only_paths=changed_set) if changed_set else []
        chunks = chunk_files(files, repo_url=repo_url) if files else []

        # For first index of this repo (no base), build a clean collection.
        if base_commit is None:
            reset_collection(collection_name)
            index_chunks(chunks, collection_name=collection_name)
        else:
            incremental_index_chunks(
                repo_url=repo_url,
                collection_name=collection_name,
                chunks=chunks,
                changed_or_added_paths=diff.changed_or_added_py_abs,
                deleted_paths=diff.deleted_py_abs,
            )

        set_repo_state(repo_url, last_indexed_commit=diff.head_commit)
        return {
            "status": "success",
            "mode": "incremental" if base_commit is not None else "initial_full",
            "message": "Repository indexed successfully",
            "chunks_indexed": len(chunks),
            "head_commit": diff.head_commit,
            "base_commit": base_commit,
            "changed_files": [os.path.relpath(p, repo_path) for p in diff.changed_or_added_py_abs],
            "deleted_files": [os.path.relpath(p, repo_path) for p in diff.deleted_py_abs],
        }
    except CodebaseAgentError as e:
        return {
            "status": "error",
            "error_type": e.code,
            "message": e.user_message,
            "hint": e.hint,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Endpoint to query codebase
@app.get("/ask")
def ask(query: str, repo_url: str):
    """Query the indexed codebase"""
    if not query or query.strip() == "":
        return {
            "status": "error",
            "message": "Query cannot be empty"
        }
    try:
        results = search(query, collection_name=collection_name_for_repo(repo_url))
        return {
            "status": "success",
            "results": results
        }
    except CodebaseAgentError as e:
        return {
            "status": "error",
            "error_type": e.code,
            "message": e.user_message,
            "hint": e.hint,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
