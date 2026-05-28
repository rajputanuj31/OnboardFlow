import sys
import sqlite3
from pathlib import Path
from functools import lru_cache
from langchain_openai import ChatOpenAI

# Ensure backend root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.state import RepoState
from onboard_github import fetch_specific_file
from utils.vector_db import DB_PATH

@lru_cache(maxsize=16)
def _llm(api_key: str = "") -> ChatOpenAI:
    """Lazy-initialize the LLM so load_dotenv() in main.py runs first."""
    from config import settings
    key = api_key or settings.openai_api_key
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
    """Helper to read files from fetched files, local cache, or GitHub API."""
    # 1. Check fetched_files in state
    if filepath in state.get("fetched_files", {}):
        return state["fetched_files"][filepath]

    # 2. Check local SQLite vector DB cache
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT content FROM code_chunks WHERE session_id = ? AND filepath = ? ORDER BY chunk_index",
                (state["session_id"], filepath)
            )
            rows = cursor.fetchall()
            if rows:
                return "".join(row[0] for row in rows)
    except Exception:
        pass

    # 3. Fallback: Fetch directly from GitHub API
    content = await fetch_specific_file(state["repo_name"], filepath)
    if content:
        return content
        
    return f"Error: Could not read file '{filepath}'. Make sure the path is correct and exists."
