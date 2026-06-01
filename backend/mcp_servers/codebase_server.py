"""Custom Codebase MCP server.

Exposes tools that operate over the locally-ingested repo state:
- `read_codebase_file(session_id, filepath)` — fetched_files → Qdrant chunks → GitHub API fallback
- `semantic_search(session_id, query, top_k)` — Qdrant ANN over the session's chunks
- `get_repo_structure(session_id)` — cleaned tree built during /ingest
- `get_file_summary(session_id)` — LLM-generated repo summary + architecture notes

Launched as a stdio subprocess by the FastAPI backend at startup.
Run standalone for debugging: `python -m mcp_servers codebase`.
"""
import sys
from pathlib import Path

# Ensure backend root is on sys.path so we can import siblings (utils, graph, config)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from utils.session_db import get_session
from utils.vector_db import query_vector_db
from graph.helpers import _read_codebase_file


mcp = FastMCP("onboardflow-codebase")


@mcp.tool()
async def read_codebase_file(session_id: str, filepath: str) -> str:
    """Read a specific file from the ingested repository.

    Resolution order: in-memory fetched_files → Qdrant chunks (assembled) →
    live GitHub API fetch. Returns the full file contents as text.
    """
    state = get_session(session_id)
    if not state:
        return f"Error: session '{session_id}' not found. Has /ingest been called?"
    return await _read_codebase_file(filepath, state)


@mcp.tool()
async def semantic_search(session_id: str, query: str, top_k: int = 4) -> list[dict]:
    """Semantic search across the ingested repo's code chunks.

    Returns a list of {filepath, chunk_index, content, similarity} dicts,
    ordered by descending similarity. Limited to chunks belonging to `session_id`.
    """
    return await query_vector_db(session_id, query, top_k=top_k)


@mcp.tool()
def get_repo_structure(session_id: str) -> list[str]:
    """Return the cleaned, depth-limited tree built during ingestion.

    The tree has noise (node_modules, lockfiles, binaries, etc.) pruned and
    is formatted as Markdown lines with item counts.
    """
    state = get_session(session_id)
    if not state:
        return []
    return list(state.get("repo_structure") or [])


@mcp.tool()
def get_file_summary(session_id: str) -> dict:
    """Return the LLM-generated repository summary and Mermaid architecture notes."""
    state = get_session(session_id)
    if not state:
        return {"summary": "", "architecture": "", "error": f"session '{session_id}' not found"}
    return {
        "summary": state.get("repo_summary", ""),
        "architecture": state.get("architecture_notes", ""),
        "repo_name": state.get("repo_name", ""),
        "repo_description": state.get("repo_description", ""),
        "repo_language": state.get("repo_language", ""),
    }


def run_stdio() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_stdio()
