import sys
from contextvars import ContextVar
from pathlib import Path
from functools import lru_cache
from langchain_openai import ChatOpenAI

# Ensure backend root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.state import RepoState
from onboard_github import fetch_specific_file
from utils.vector_db import read_file_chunks

# Per-request model API key, set by the endpoints in main.py and read by _llm().
# Deliberately kept OUT of graph state: anything in state gets serialized into
# LangSmith traces and the persisted session snapshot, leaking the user's key.
_request_api_key: ContextVar[str] = ContextVar("model_api_key", default="")


def set_request_api_key(key: str) -> None:
    """Bind the user's model API key to the current request context."""
    _request_api_key.set(key or "")


def get_request_api_key() -> str:
    return _request_api_key.get()


def _llm(api_key: str = "") -> ChatOpenAI:
    """Lazy-initialize the LLM so load_dotenv() in main.py runs first.

    Resolution order: explicit arg > request context > backend .env key.
    """
    from config import settings
    key = api_key or _request_api_key.get() or settings.openai_api_key
    return _cached_llm(key)


@lru_cache(maxsize=16)
def _cached_llm(key: str) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.0, streaming=True, api_key=key)


def _build_file_contents(files: dict[str, str], char_limit: int = 12_000) -> str:
    """Concatenate fetched files into a single string, respecting a char limit."""
    parts = []
    total = 0
    for filename, content in files.items():
        block = f"\n=== {filename} ===\n{content}"
        if total + len(block) > char_limit:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


async def _read_codebase_file(filepath: str, state: RepoState) -> str:
    """Helper to read files from fetched files, vector DB chunks, or GitHub API."""
    # 1. Check fetched_files in state
    if filepath in state.get("fetched_files", {}):
        return state["fetched_files"][filepath]

    # 2. Check Qdrant chunks for this session
    try:
        joined = read_file_chunks(state["session_id"], filepath)
        if joined:
            return joined
    except Exception:
        pass

    # 3. Fallback: Fetch directly from GitHub API
    content = await fetch_specific_file(state["repo_name"], filepath)
    if content:
        return content

    return f"Error: Could not read file '{filepath}'. Make sure the path is correct and exists."
