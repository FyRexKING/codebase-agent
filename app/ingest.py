import os
from git import Repo
from app.config import REPO_DIR
# Clone GitHub repository locally
def clone_repo(repo_url):
    # If repo already exists, reuse it
    if os.path.exists(REPO_DIR):
        return REPO_DIR
    # Clone repo into data/repo
    Repo.clone_from(repo_url, REPO_DIR)
    return REPO_DIR
# Load all Python files from repo
def load_python_files(repo_path):
    files = []
    # Walk through directory tree
    for root, _, filenames in os.walk(repo_path):
        for name in filenames:
            # Only index Python files for now
            if name.endswith(".py"):
                full_path = os.path.join(root, name)
                try:
                    # Read file content
                    with open(full_path, "r", errors="ignore") as f:
                        content = f.read()
                    files.append({
                        "path": full_path,
                        "content": content
                    })
                except Exception:
                    # Skip unreadable files
                    continue
    return files
# Convert files into chunks for embedding
def chunk_files(files):
    chunks = []
    # For now each file is one chunk
    # Later we will split by function/class
    for file in files:
        chunks.append({
            "text": file["content"],
            "path": file["path"]
        })
    return chunks
