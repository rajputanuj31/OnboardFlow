"""PR review nodes — Phase 4 multi-agent flow.

Topology (see graph.py::build_pr_review_graph):
  fetch_pr ─┬─> code_explainer ─┐
            ├─> impact_analyst ─┼─> synthesizer (streams) ──> END
            └─> risk_assessor ──┘

`fetch_pr` is a plain data-fetching node (no LLM): it uses the GitHub MCP server
tools to pull PR metadata, the per-file changes, and the unified diff. Each of
the three specialists is a `create_react_agent` bound to a curated subset of the
Codebase MCP tools (read_codebase_file, semantic_search, get_repo_structure).
The synthesizer streams the final markdown review tagged "final".
"""
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.helpers import _llm
from graph.state import PRReviewState
from mcp_clients import get_tool_by_name, is_github_mcp_available
from prompts import (
    PR_CODE_EXPLAINER_PROMPT,
    PR_IMPACT_ANALYST_PROMPT,
    PR_RISK_ASSESSOR_PROMPT,
    PR_SYNTHESIZER_PROMPT,
)

PR_SPECIALIST_NAMES = ("code_explainer", "impact_analyst", "risk_assessor")

DIFF_TRUNCATE_CHARS = 12_000


def _tools_for(names: list[str]) -> list:
    out = []
    for n in names:
        t = get_tool_by_name(n)
        if t is not None:
            out.append(t)
    return out


async def _call_github_tool(tool_name: str, args: dict[str, Any]) -> Any:
    """Invoke a GitHub MCP tool by name. Returns parsed JSON when possible, else raw text.

    Tool names follow the github/github-mcp-server contract: `pull_request_read`, etc.
    """
    tool = get_tool_by_name(tool_name)
    if tool is None:
        raise RuntimeError(
            f"GitHub MCP tool '{tool_name}' not available. "
            "Install github-mcp-server and ensure it is on PATH."
        )
    result = await tool.ainvoke(args)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result
    return result


def _files_summary(pr_files: list[dict]) -> str:
    if not pr_files:
        return "(no file list)"
    lines = []
    for f in pr_files[:30]:
        name = f.get("filename") or f.get("path") or "?"
        add = f.get("additions", 0)
        rem = f.get("deletions", 0)
        status = f.get("status", "")
        lines.append(f"- {name} ({status}, +{add}/-{rem})")
    if len(pr_files) > 30:
        lines.append(f"...and {len(pr_files) - 30} more files")
    return "\n".join(lines)


def _truncate(text: str, limit: int = DIFF_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [diff truncated at {limit} chars] ..."


# ── fetch_pr ──────────────────────────────────────────────────────────────────

async def fetch_pr(state: PRReviewState) -> dict:
    """Pull PR metadata, file list, and unified diff via the GitHub MCP server."""
    if not is_github_mcp_available():
        raise RuntimeError(
            "GitHub MCP server is not available. Install github-mcp-server and restart the backend."
        )

    owner = state["pr_owner"]
    repo = state["pr_repo"]
    pull_number = state["pr_number"]

    pr_meta_raw = await _call_github_tool(
        "pull_request_read",
        {"owner": owner, "repo": repo, "pullNumber": pull_number, "method": "get"},
    )
    pr_files_raw = await _call_github_tool(
        "pull_request_read",
        {"owner": owner, "repo": repo, "pullNumber": pull_number, "method": "get_files"},
    )
    pr_diff_raw = await _call_github_tool(
        "pull_request_read",
        {"owner": owner, "repo": repo, "pullNumber": pull_number, "method": "get_diff"},
    )

    pr_metadata: dict[str, Any] = {}
    if isinstance(pr_meta_raw, dict):
        pr_metadata = {
            "title": pr_meta_raw.get("title", ""),
            "description": pr_meta_raw.get("body", "") or "",
            "author": (pr_meta_raw.get("user") or {}).get("login", ""),
            "base": (pr_meta_raw.get("base") or {}).get("ref", ""),
            "head": (pr_meta_raw.get("head") or {}).get("ref", ""),
            "state": pr_meta_raw.get("state", ""),
        }

    pr_files: list[dict] = pr_files_raw if isinstance(pr_files_raw, list) else []
    pr_diff: str = pr_diff_raw if isinstance(pr_diff_raw, str) else json.dumps(pr_diff_raw)

    return {
        "pr_metadata": pr_metadata,
        "pr_files": pr_files,
        "pr_diff": pr_diff,
    }


# ── specialists ───────────────────────────────────────────────────────────────

def _prompt_args(state: PRReviewState) -> dict[str, Any]:
    meta = state.get("pr_metadata") or {}
    return {
        "pr_title": meta.get("title", ""),
        "pr_description": meta.get("description", "") or "(none)",
        "pr_author": meta.get("author", ""),
        "pr_base": meta.get("base", ""),
        "pr_head": meta.get("head", ""),
        "pr_files_summary": _files_summary(state.get("pr_files") or []),
        "pr_diff": _truncate(state.get("pr_diff") or ""),
        "session_id": state.get("session_id") or "",
    }


async def _run_specialist(
    state: PRReviewState,
    *,
    name: str,
    prompt_template: str,
    tool_names: list[str],
) -> dict:
    args = _prompt_args(state)
    system_prompt = prompt_template.format(**args)

    tools = _tools_for(tool_names) if state.get("session_id") else []
    llm = _llm()
    agent = create_react_agent(model=llm, tools=tools, prompt=SystemMessage(content=system_prompt))

    config: RunnableConfig = {
        "tags": [f"agent:{name}"],
        "run_name": name,
        "recursion_limit": 12,
    }
    user_kick = HumanMessage(
        content=f"Review PR #{state['pr_number']} in {state['pr_owner']}/{state['pr_repo']}."
    )
    result = await agent.ainvoke({"messages": [user_kick]}, config=config)
    answer = result["messages"][-1].content if result.get("messages") else ""
    return {"specialist_outputs": {name: answer}}


async def code_explainer(state: PRReviewState) -> dict:
    return await _run_specialist(
        state,
        name="code_explainer",
        prompt_template=PR_CODE_EXPLAINER_PROMPT,
        tool_names=["read_codebase_file", "semantic_search"],
    )


async def impact_analyst(state: PRReviewState) -> dict:
    return await _run_specialist(
        state,
        name="impact_analyst",
        prompt_template=PR_IMPACT_ANALYST_PROMPT,
        tool_names=["semantic_search", "get_repo_structure", "read_codebase_file"],
    )


async def risk_assessor(state: PRReviewState) -> dict:
    return await _run_specialist(
        state,
        name="risk_assessor",
        prompt_template=PR_RISK_ASSESSOR_PROMPT,
        tool_names=["read_codebase_file"],
    )


# ── synthesizer ───────────────────────────────────────────────────────────────

async def synthesizer(state: PRReviewState) -> dict:
    outputs = state.get("specialist_outputs") or {}
    meta = state.get("pr_metadata") or {}
    prompt = PR_SYNTHESIZER_PROMPT.format(
        code_explainer_output=outputs.get("code_explainer", "(missing)"),
        impact_analyst_output=outputs.get("impact_analyst", "(missing)"),
        risk_assessor_output=outputs.get("risk_assessor", "(missing)"),
        pr_title=meta.get("title", ""),
        pr_author=meta.get("author", ""),
    )

    llm = _llm().with_config(
        tags=["final", "agent:synthesizer"], run_name="synthesizer"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"final_review": response.content}
