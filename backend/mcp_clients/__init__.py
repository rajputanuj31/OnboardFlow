from __future__ import annotations

import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings

logger = logging.getLogger(__name__)

GITHUB_MCP_BINARY = "github-mcp-server"

# Module-level singleton populated by MCPRegistry.start(). Graph nodes import the
# accessor functions below to find loaded tools without needing FastAPI app state.
_registry: "MCPRegistry | None" = None


def get_registry() -> "MCPRegistry":
    if _registry is None:
        raise RuntimeError("MCPRegistry not started. Has FastAPI lifespan run?")
    return _registry


def get_codebase_tools() -> list[BaseTool]:
    return get_registry().codebase_tools


def get_github_tools() -> list[BaseTool]:
    return get_registry().github_tools


def get_tool_by_name(name: str) -> BaseTool | None:
    if name == "semantic_search":
        from langchain_core.tools import StructuredTool
        from utils.vector_db import query_vector_db
        from graph.helpers import get_request_api_key

        async def _local_semantic_search(session_id: str, query: str, top_k: int = 4) -> list[dict]:
            key = get_request_api_key()
            return await query_vector_db(session_id, query, top_k=top_k, model_api_key=key)

        return StructuredTool.from_function(
            coroutine=_local_semantic_search,
            name="semantic_search",
            description=(
                "Semantic search across the ingested repo's code chunks. "
                "Returns a list of {filepath, chunk_index, content, similarity} dicts, "
                "ordered by descending similarity. Limited to chunks belonging to session_id."
            )
        )

    reg = get_registry()
    for t in (*reg.codebase_tools, *reg.github_tools):
        if t.name == name:
            return t
    return None


def is_github_mcp_available() -> bool:
    return _registry is not None and _registry.github_available


class MCPRegistry:
    """Holds open MCP client sessions + their loaded LangChain tools.

    One instance lives on `app.state.mcp` for the lifetime of the FastAPI app.
    """

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self.codebase_session: ClientSession | None = None
        self.github_session: ClientSession | None = None
        self.codebase_tools: list[BaseTool] = []
        self.github_tools: list[BaseTool] = []
        self.github_available: bool = False

    async def start(self) -> None:
        global _registry
        _registry = self
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        # Codebase MCP server: launched as `python -m mcp_servers codebase`
        # using the same interpreter we're running under.
        backend_root = Path(__file__).resolve().parent.parent
        codebase_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_servers", "codebase"],
            env={
                "PYTHONPATH": str(backend_root),
                **_passthrough_env(),
            },
        )
        try:
            self.codebase_session = await self._enter_session(codebase_params)
            self.codebase_tools = await load_mcp_tools(self.codebase_session)
            logger.info("Codebase MCP: loaded %d tools", len(self.codebase_tools))
        except Exception as e:
            logger.exception("Failed to start Codebase MCP server: %s", e)
            raise  # Codebase MCP is critical; surface failure at boot.

        # GitHub MCP server (optional): only attempt if binary is on PATH.
        binary_path = shutil.which(GITHUB_MCP_BINARY)
        if binary_path is None:
            fallback_paths = [
                backend_root / "bin" / GITHUB_MCP_BINARY,
                Path.home() / "go" / "bin" / GITHUB_MCP_BINARY,
            ]
            for path in fallback_paths:
                if path.is_file() and os.access(path, os.X_OK):
                    binary_path = str(path)
                    break

        if binary_path is None:
            logger.warning(
                "GitHub MCP binary %r not found on PATH; PR-review tooling disabled. "
                "Install with: go install github.com/github/github-mcp-server@latest",
                GITHUB_MCP_BINARY,
            )
            return

        github_params = StdioServerParameters(
            command=binary_path,
            args=["stdio"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token,
                **_passthrough_env(),
            },
        )
        try:
            self.github_session = await self._enter_session(github_params)
            self.github_tools = await load_mcp_tools(self.github_session)
            self.github_available = True
            logger.info("GitHub MCP: loaded %d tools", len(self.github_tools))
        except Exception as e:
            logger.warning("GitHub MCP server failed to start: %s", e)

    async def _enter_session(self, params: StdioServerParameters) -> ClientSession:
        assert self._stack is not None
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def stop(self) -> None:
        global _registry
        if self._stack is not None:
            try:
                await self._stack.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error tearing down MCP registry: %s", e)
            self._stack = None
        self.codebase_session = None
        self.github_session = None
        self.codebase_tools = []
        self.github_tools = []
        self.github_available = False
        if _registry is self:
            _registry = None

    def status(self) -> dict[str, Any]:
        return {
            "codebase": {
                "connected": self.codebase_session is not None,
                "tool_count": len(self.codebase_tools),
            },
            "github": {
                "connected": self.github_session is not None,
                "available": self.github_available,
                "tool_count": len(self.github_tools),
            },
        }

    def all_tools(self) -> list[BaseTool]:
        return [*self.codebase_tools, *self.github_tools]


def _passthrough_env() -> dict[str, str]:
    """Env vars worth forwarding to MCP subprocesses (PATH, HOME, etc.)."""
    keep = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
            "OPENAI_API_KEY", "QDRANT_URL", "GITHUB_TOKEN",
            "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
    return {k: os.environ[k] for k in keep if k in os.environ}
