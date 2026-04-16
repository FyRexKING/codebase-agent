import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm.config import DATA_DIR


STATE_PATH = os.path.join(DATA_DIR, "index_state.json")


@dataclass(frozen=True)
class RepoIndexState:
    repo_url: str
    last_indexed_commit: str


def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def get_repo_state(repo_url: str) -> Optional[RepoIndexState]:
    normalized = (repo_url or "").strip().lower()
    if not normalized:
        return None
    state = _load_state()
    entry = state.get(normalized)
    if not isinstance(entry, dict):
        return None
    commit = (entry.get("last_indexed_commit") or "").strip()
    if not commit:
        return None
    return RepoIndexState(repo_url=normalized, last_indexed_commit=commit)


def set_repo_state(repo_url: str, *, last_indexed_commit: str) -> None:
    normalized = (repo_url or "").strip().lower()
    if not normalized:
        return
    commit = (last_indexed_commit or "").strip()
    if not commit:
        return
    state = _load_state()
    state[normalized] = {"last_indexed_commit": commit}
    _save_state(state)

