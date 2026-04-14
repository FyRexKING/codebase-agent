import os

# Directory where we store downloaded repos
DATA_DIR = os.environ.get("DATA_DIR", "data")

# Folder where repo will be cloned
REPO_DIR = os.path.join(DATA_DIR, "repo")

# Qdrant connection
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# Default collection name (kept for backward compatibility; new code should prefer
# per-repo collection naming).
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "code_chunk")
