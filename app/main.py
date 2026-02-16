from fastapi import FastAPI
from app.ingest import clone_repo, load_python_files, chunk_files
from app.rag import index_chunks, search
# Create API app
app = FastAPI()
# Endpoint to ingest a GitHub repo
@app.post("/ingest")
def ingest(repo_url: str):
    # Clone repo
    repo_path = clone_repo(repo_url)
    # Load Python files
    files = load_python_files(repo_path)
    # Convert into chunks
    chunks = chunk_files(files)
    # Index into vector DB
    index_chunks(chunks)
    return {"status": "indexed"}
# Endpoint to query codebase
@app.get("/ask")
def ask(query: str):
    # Perform semantic search
    results = search(query)
    return {"results": results}
