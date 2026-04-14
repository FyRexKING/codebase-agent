# 🚀 Codebase Agent — Engineering Guidelines

## 🎯 Objective
Build a scalable, modular, and production-grade codebase understanding agent that:
- Parses repositories
- Generates embeddings
- Stores them in a vector DB (Qdrant)
- Retrieves relevant context
- Answers developer queries with high accuracy

This system must prioritize:
- Correctness over speed
- Modularity over shortcuts
- Observability over assumptions

---

# 🧱 1. Architecture Overview

## High-Level Flow

User Query
   ↓
Query Processor
   ↓
Retriever (Vector DB + Filters)
   ↓
Context Builder
   ↓
LLM Reasoning Layer
   ↓
Response Generator

---

## Core Modules


/src
/ingestion
/parser
/chunking
/embedding
/vectorstore
/retrieval
/context
/llm
/evaluation
/utils
/config


Each module must be:
- Independently testable
- Loosely coupled
- Clearly documented

---

# ⚙️ 2. Strict Coding Principles

## 2.1 Modularity (NON-NEGOTIABLE)
- No file > 300 lines
- Each function has ONE responsibility
- No business logic in controllers

BAD:
- One file doing parsing + embedding + storage

GOOD:
- parser → chunker → embedder → store (separate layers)

---

## 2.2 Type Safety
- Use type hints everywhere
- Enforce with `mypy`

Example:
```python
def chunk_file(content: str) -> list[str]:
2.3 No Hidden State
Avoid globals
Pass dependencies explicitly
2.4 Config Driven

All configs must be externalized:

/config/settings.yaml

Never hardcode:

Model names
Chunk sizes
DB paths
2.5 Logging (MANDATORY)

Use structured logging.

logger.info("Chunking started", file=file_path)
🧩 3. Module Design Rules
📥 Ingestion Module

Responsibility:

Clone repo / load files

Rules:

No parsing here
Only file discovery
🌳 Parser Module

Responsibility:

Convert code → structured representation (AST if possible)

Rules:

Language-specific parsers separated
No embedding logic
✂️ Chunking Module

Responsibility:

Split code intelligently

Strategies:

Function-level
Class-level
AST-aware chunks

DO NOT:

Use naive fixed chunking only
🧠 Embedding Module

Responsibility:

Convert chunks → vectors

Rules:

Pluggable models
Batch processing
🗃️ Vector Store Module

Responsibility:

Store & retrieve embeddings

Rules:

Abstract interface
class VectorStore:
    def upsert(...)
    def query(...)
🔍 Retrieval Module

Responsibility:

Fetch relevant chunks

Must Support:

Top-k similarity
Metadata filtering (file, function, commit)
🧱 Context Builder

Responsibility:

Assemble final prompt context

Rules:

Deduplicate chunks
Maintain token limits
🤖 LLM Module

Responsibility:

Generate responses

Rules:

No direct DB calls
Only consumes context
📊 Evaluation Module

Responsibility:

Measure correctness

Metrics:

Retrieval accuracy
Answer faithfulness
Latency
🔁 4. Incremental Indexing (CRITICAL FEATURE)

System must support:

Initial Ingestion
Full repo → embeddings
Incremental Updates
Detect changed files via git diff
Re-embed ONLY changed files
Required Design
Repo Metadata:
- repo_url
- last_commit_hash
- indexed_files
🧠 5. Query Handling Pipeline

Steps:

Normalize query
Expand query (optional)
Retrieve top-k chunks
Re-rank (optional)
Build context
Call LLM
Post-process response
⚠️ 6. Anti-Patterns (STRICTLY FORBIDDEN)

❌ Mixing parsing + embedding
❌ Calling LLM without retrieval
❌ Large monolithic files
❌ Hardcoded configs
❌ Ignoring token limits
❌ No logging
❌ No evaluation

🧪 7. Testing Requirements

Each module must have:

Unit tests
Edge case tests
Failure handling

Example:

Empty file
Large file
Invalid syntax
Retrieval < 200ms
Embedding batch optimized
Avoid redundant DB calls
🧭 10. Development Workflow
Build ingestion
Add parser
Add chunking
Add embeddings
Add retrieval
Add LLM layer
Add evaluation

DO NOT jump steps.

🧠 11. Design Philosophy
Think like a compiler, not a chatbot
Structure > brute force
Retrieval quality > LLM power
🏁 12. Final Goal

The agent should:

✔ Trace execution across files
✔ Answer "why" questions
✔ Handle large repos
✔ Update incrementally

🔥 Final Rule

If a human developer wouldn’t understand your system design in 10 minutes,
your design is bad.

Simplify.
Modularize.
Ship.


---

## Straight talk

If you actually follow this:
- You’ll build something **way above average student projects**
- This starts looking like **DeepWiki / Sourcegraph-level thinking**

If you don’t:
- You’ll end up with a **Frankenstein pipeline that barely works**

---

If you want next step, I can:
- Turn this into a **Cursor rule file (`.cursorrules`)**
- Or design your **exact class structure + interfaces (real code-level blueprint)**