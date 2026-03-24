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
    # 🚫 Directories to ignore
    IGNORE_DIRS = {"venv", ".git", "__pycache__", "tests", "docs", "node_modules"}
    # 🚫 File patterns to ignore
    IGNORE_FILES = {".pyc"}
    for root, dirs, filenames in os.walk(repo_path):
        # 🔥 Remove ignored directories (IMPORTANT)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in filenames:
            # Only Python files
            if not name.endswith(".py"):
                continue
            # Skip unwanted file types
            if any(name.endswith(ext) for ext in IGNORE_FILES):
                continue
            full_path = os.path.join(root, name)
            try:
                with open(full_path, "r", errors="ignore") as f:
                    content = f.read()
                if not content.strip() or len(content) < 50:
                    continue
                files.append({
                    "path": full_path,
                    "content": content
                })
            except Exception:
                continue
    return files
# def chunk_files(files):
#     chunks=[]
#     for file in files:
#         chunks.append({
#             "text": file["content"],
#             "path": file["path"]
#         })
#     return chunks
def chunk_files(files):
    chunks = []
    CHUNK_SIZE = 400
    for file in files:
        content = file["content"]
        for i in range(0, len(content), CHUNK_SIZE):
            chunk_text = content[i:i+CHUNK_SIZE]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "path": file["path"]
                })
    return chunks
