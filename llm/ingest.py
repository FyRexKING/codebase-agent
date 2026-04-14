import ast
import os
import re
import shutil
from git import Repo
from llm.config import REPO_DIR

def is_valid_github_url(url):
    """Validate GitHub repository URL"""
    pattern = r'^https?://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$'
    return re.match(pattern, url) is not None

# Clone GitHub repository locally
def clone_repo(repo_url):
    """Clone a GitHub repo, replacing any existing repo"""
    # Validate URL format
    if not is_valid_github_url(repo_url):
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")
    
    # Clean up existing repo directory to avoid stale cache
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    
    # Clone repo into data/repo
    try:
        Repo.clone_from(repo_url, REPO_DIR)
    except Exception as e:
        raise ValueError(f"Failed to clone repository: {str(e)}")
    
    return REPO_DIR

def load_python_files(repo_path):
    files = []
    IGNORE_DIRS = {"venv", ".git", "__pycache__", "tests", "docs", "node_modules"}
    IGNORE_FILES = {".pyc"}
    
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in filenames:
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

def _repo_name_from_url(repo_url):
    cleaned = (repo_url or "").rstrip("/")
    name = cleaned.split("/")[-1] if cleaned else "unknown"
    return name[:-4] if name.endswith(".git") else name


def _module_name_from_path(rel_path):
    normalized = rel_path.replace("\\", "/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    if normalized.endswith("/__init__"):
        normalized = normalized[:-9]
    return normalized.replace("/", ".") if normalized else "root"


def _extract_imports(tree):
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return sorted(imports)


def _extract_calls(node):
    calls = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                calls.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                calls.add(sub.func.attr)
    return sorted(calls)


def _slice_source(lines, start_line, end_line):
    start = max(1, int(start_line or 1))
    end = max(start, int(end_line or start))
    return "\n".join(lines[start - 1:end])


def _append_symbol_chunk(chunks, *, text, path, repo_url, repo_name, module_name, symbol_name,
                         symbol_type, start_line, end_line, imports, dependencies,
                         parent_symbol=None):
    if not text.strip():
        return
    symbol_id = f"{module_name}.{symbol_name}" if module_name else symbol_name
    chunks.append(
        {
            "text": text,
            "path": path,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "module": module_name,
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "symbol_id": symbol_id,
            "parent_symbol": parent_symbol,
            "start_line": start_line,
            "end_line": end_line,
            "imports": imports,
            "dependencies": dependencies,
        }
    )


def chunk_files(files, repo_url=""):
    chunks = []
    repo_name = _repo_name_from_url(repo_url)

    for file in files:
        path = file["path"]
        content = file["content"]
        rel_path = os.path.relpath(path, REPO_DIR)
        module_name = _module_name_from_path(rel_path)
        lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None

        if tree is None:
            chunk_size = 600
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i:i + chunk_size]
                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "path": path,
                            "repo_url": repo_url,
                            "repo_name": repo_name,
                            "module": module_name,
                            "symbol_name": f"fragment_{i // chunk_size}",
                            "symbol_type": "module_fragment",
                            "symbol_id": f"{module_name}.fragment_{i // chunk_size}",
                            "parent_symbol": None,
                            "start_line": None,
                            "end_line": None,
                            "imports": [],
                            "dependencies": [],
                        }
                    )
            continue

        imports = _extract_imports(tree)
        had_symbol_chunks = False

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                text = _slice_source(lines, node.lineno, node.end_lineno)
                _append_symbol_chunk(
                    chunks,
                    text=text,
                    path=path,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    module_name=module_name,
                    symbol_name=node.name,
                    symbol_type="function",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    imports=imports,
                    dependencies=_extract_calls(node),
                )
                had_symbol_chunks = True

            elif isinstance(node, ast.ClassDef):
                class_text = _slice_source(lines, node.lineno, node.end_lineno)
                _append_symbol_chunk(
                    chunks,
                    text=class_text,
                    path=path,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    module_name=module_name,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    imports=imports,
                    dependencies=_extract_calls(node),
                )
                had_symbol_chunks = True

                parent_symbol = f"{module_name}.{node.name}"
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_text = _slice_source(lines, child.lineno, child.end_lineno)
                        method_name = f"{node.name}.{child.name}"
                        _append_symbol_chunk(
                            chunks,
                            text=method_text,
                            path=path,
                            repo_url=repo_url,
                            repo_name=repo_name,
                            module_name=module_name,
                            symbol_name=method_name,
                            symbol_type="method",
                            start_line=child.lineno,
                            end_line=child.end_lineno,
                            imports=imports,
                            dependencies=_extract_calls(child),
                            parent_symbol=parent_symbol,
                        )

        if not had_symbol_chunks:
            chunks.append(
                {
                    "text": content,
                    "path": path,
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "module": module_name,
                    "symbol_name": "module",
                    "symbol_type": "module",
                    "symbol_id": f"{module_name}.module",
                    "parent_symbol": None,
                    "start_line": 1,
                    "end_line": len(lines),
                    "imports": imports,
                    "dependencies": _extract_calls(tree),
                }
            )

    return chunks
