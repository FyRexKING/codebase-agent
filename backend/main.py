from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from llm.ingest import clone_repo, load_python_files, chunk_files
from llm.rag import index_chunks, search, collection_name_for_repo, reset_collection

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
            "POST /ingest": "Ingest a GitHub repo (query param: repo_url)",
            "GET /ask": "Query indexed codebase (query param: query)",
            "GET /static/": "Frontend interface"
        }
    }

# Endpoint to ingest a GitHub repo
@app.post("/ingest")
def ingest(repo_url: str):
    """Ingest a GitHub repository and index its Python code"""
    try:
        # Clone repo
        repo_path = clone_repo(repo_url)
        # Load Python files
        files = load_python_files(repo_path)
        # Convert into chunks
        chunks = chunk_files(files, repo_url=repo_url)
        # Index into vector DB
        collection_name = collection_name_for_repo(repo_url)
        reset_collection(collection_name)
        index_chunks(chunks, collection_name=collection_name)
        return {
            "status": "success",
            "message": "Repository indexed successfully",
            "chunks_indexed": len(chunks)
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
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
