from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CodebaseAgentError(Exception):
    code: str
    user_message: str
    hint: Optional[str] = None
    detail: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover (presentation only)
        base = f"{self.code}: {self.user_message}"
        if self.detail:
            return f"{base} ({self.detail})"
        return base


def is_gemini_quota_error(msg: str) -> bool:
    t = (msg or "").lower()
    return (
        "429" in t
        or "resource_exhausted" in t
        or "quota" in t
        or "rate limit" in t
        or "too many requests" in t
    )


def is_gemini_auth_error(msg: str) -> bool:
    t = (msg or "").lower()
    return "api key" in t and ("invalid" in t or "unauthorized" in t or "permission" in t)


def classify_git_clone_error(msg: str) -> CodebaseAgentError:
    """
    Turn common Git/GitHub clone failures into user-friendly messages.
    We intentionally match on substrings since GitPython wraps libgit2/git output.
    """
    t = (msg or "").lower()

    # Common GitHub/HTTP errors.
    if "repository not found" in t or "not found" in t and "github.com" in t:
        return CodebaseAgentError(
            code="GITHUB_REPO_NOT_FOUND",
            user_message="GitHub repository not found.",
            hint="Check the URL is correct. If the repo is private, you must provide credentials (this demo currently clones anonymously).",
            detail=msg,
        )

    if "authentication failed" in t or "could not read username" in t or "permission denied" in t:
        return CodebaseAgentError(
            code="GITHUB_AUTH_FAILED",
            user_message="GitHub authentication failed (repo may be private).",
            hint="Use a public repo URL for the demo, or add an auth method (PAT/SSH) to the ingestion step.",
            detail=msg,
        )

    if "rate limit" in t:
        return CodebaseAgentError(
            code="GITHUB_RATE_LIMITED",
            user_message="GitHub rate limit hit while cloning the repository.",
            hint="Wait a bit and retry, or use a smaller repo / authenticated clone to increase limits.",
            detail=msg,
        )

    return CodebaseAgentError(
        code="GITHUB_CLONE_FAILED",
        user_message="Failed to clone the GitHub repository.",
        hint="Verify the repo URL is valid and publicly accessible.",
        detail=msg,
    )

