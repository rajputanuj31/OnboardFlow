"""Codebase MCP server boots over stdio and exposes the expected tools."""
import os
import sys
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_codebase_mcp_lists_four_tools():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    backend_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(backend_root)}

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers", "codebase"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)

    assert names == ["get_file_summary", "get_repo_structure", "read_codebase_file", "semantic_search"]
