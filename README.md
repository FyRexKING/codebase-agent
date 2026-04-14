# Codebase RAG Agent

A semantic search system for codebases using Retrieval-Augmented Generation (RAG).

## 📁 Project Structure

```
codebase-agent/
├── backend/              # FastAPI backend layer
│   ├── __init__.py      
│   └── main.py          # FastAPI endpoints + static file serving
├── llm/                 # LLM & RAG layer
│   ├── __init__.py      
│   ├── config.py        # Configuration
│   ├── ingest.py        # Git cloning + file loading + chunking
│   └── rag.py           # Vector DB + Gemini integration
├── frontend/            # Frontend UI (no backend needed)
│   ├── index.html       # Main page
│   ├── style.css        # Styling
│   └── script.js        # JavaScript logic
├── data/                # Data storage
│   └── repo/            # Cloned repositories
└── requirements.txt     # Python dependencies
```

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Qdrant Vector DB (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Set Environment Variables
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Backend
```bash
cd /home/beluga/codebase-agent
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## 🚀 Usage

1. Open your browser: `http://localhost:8000/static/index.html`
2. Enter a GitHub repository URL (e.g., `https://github.com/tiangolo/fastapi`)
3. Click "Load Repo" to index the codebase
4. Ask questions about the code
5. View AI-generated answers with source code references

## 📚 API Endpoints

- `POST /ingest?repo_url=<url>` - Ingest and index a GitHub repository
- `GET /ask?query=<query>` - Query the indexed codebase
- `GET /` - API documentation

## 🔧 Architecture

### Backend Flow
1. **Ingest**: Clone repo → Load Python files → Chunk by Python AST symbols (module / class / function) → Embed with Gemini (`gemini-embedding-001`) → Store in Qdrant
2. **Search**: Embed query → Retrieve semantic candidates from Qdrant → Hybrid rank (semantic + BM25 + lightweight dependency/import proximity) → Send selected snippets to Gemini (`gemini-2.5-flash`) to generate the final answer

### Frontend
- Simple HTML/CSS/JS (no frameworks)
- Clean UI with two main steps: Load Repo + Ask Questions
- Real-time status messages and results display

## ⚠️ Current Limitations

- Only indexes Python files
- Qdrant point IDs are currently random UUIDs on every ingest, so re-ingesting will append duplicates unless you manually clear Qdrant
- Single working directory (`data/repo`) (no concurrent/multi-workspace ingest story yet)
- Qdrant host/port and `DATA_DIR` are documented in `.env.example` but currently hard-coded in `llm/config.py` and `llm/rag.py`
- Retrieved context sent to the LLM is snippet-only (not full structured citations/locations), which can reduce answer grounding

## 🎯 Next Steps

Planned improvements (high-signal):

- Deterministic `chunk_id` (stable IDs) so re-ingest becomes true upsert (update changed code, delete removed code)
- SQL metadata store (e.g. Postgres) for chunk metadata + dependency edges, enabling incremental re-index from `git diff` / file hashes
- Stronger grounding in `/ask`: better context formatting, explicit citations, and refusal when evidence is insufficient
