"""Multi-agent Q&A nodes.

Topology (see graph.py::build_qa_graph):
  router ─┬─> code_reader ──────┐
          ├─> arch_explainer ───┼─> synthesizer ──> update_history ──> END
          └─> dep_tracer ───────┘

The router picks 1-3 specialists; the conditional edge fan-outs via the `Send`
API. Each specialist is a `create_react_agent` bound to a curated subset of the
Codebase MCP tools. Specialists write into `specialist_outputs` (a dict reduced
by union — see state.py). The synthesizer runs a streaming LLM tagged "final"
so the SSE consumer can show its tokens to the user.

For per-agent status pills in the UI, each specialist invocation is configured
with a tag `agent:<name>` that propagates through LangGraph events.
"""
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.helpers import _llm
from graph.state import QAState
from mcp_clients import get_tool_by_name
from prompts import (
    ARCH_EXPLAINER_PROMPT,
    CODE_READER_PROMPT,
    DEP_TRACER_PROMPT,
    QA_SYNTHESIZER_PROMPT,
    ROUTER_PROMPT,
)

SPECIALIST_NAMES = ("code_reader", "arch_explainer", "dep_tracer")


class _RouterDecision(BaseModel):
    specialists: list[str] = Field(
        description="Specialist names to dispatch the question to. Subset of "
                    "['code_reader', 'arch_explainer', 'dep_tracer']."
    )


def _format_history(chat_history: list[dict], limit: int = 6) -> str:
    if not chat_history:
        return "(none)"
    recent = chat_history[-limit:]
    lines = []
    for msg in recent:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _structure_preview(structure: list[str], n: int = 30) -> str:
    if not structure:
        return "(none)"
    return "\n".join(structure[:n])


def _tools_for(names: list[str]) -> list:
    out = []
    for n in names:
        t = get_tool_by_name(n)
        if t is not None:
            out.append(t)
    return out


# ── router ────────────────────────────────────────────────────────────────────

async def router(state: QAState) -> dict:
    """Pick 1-3 specialists based on the user's question + repo context."""
    prompt = ROUTER_PROMPT.format(
        repo_name=state.get("repo_name", ""),
        repo_summary=(state.get("repo_summary", "") or "")[:1500],
        structure_preview=_structure_preview(state.get("repo_structure", [])),
        chat_history=_format_history(state.get("chat_history", [])),
        question=state["current_question"],
    )

    llm = _llm().with_structured_output(_RouterDecision)
    config: RunnableConfig = {"tags": ["agent:router"], "run_name": "router"}
    try:
        decision: _RouterDecision = await llm.ainvoke([HumanMessage(content=prompt)], config=config)
        picked = [s for s in decision.specialists if s in SPECIALIST_NAMES]
    except Exception:
        picked = []

    if not picked:
        picked = ["code_reader"]  # safe default

    return {"selected_specialists": picked}


# ── specialists ───────────────────────────────────────────────────────────────

async def _run_specialist(
    state: QAState,
    *,
    name: str,
    system_prompt: str,
    tool_names: list[str],
) -> dict:
    tools = _tools_for(tool_names)
    llm = _llm()
    agent = create_react_agent(model=llm, tools=tools, prompt=SystemMessage(content=system_prompt))

    config: RunnableConfig = {
        "tags": [f"agent:{name}"],
        "run_name": name,
        "recursion_limit": 12,  # bounds the ReAct loop inside create_react_agent
    }
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=state["current_question"])]},
        config=config,
    )
    answer = result["messages"][-1].content if result.get("messages") else ""
    return {"specialist_outputs": {name: answer}}


async def code_reader(state: QAState) -> dict:
    prompt = CODE_READER_PROMPT.format(
        repo_name=state.get("repo_name", ""),
        repo_summary=(state.get("repo_summary", "") or "")[:1500],
        session_id=state["session_id"],
        question=state["current_question"],
    )
    return await _run_specialist(
        state,
        name="code_reader",
        system_prompt=prompt,
        tool_names=["read_codebase_file", "semantic_search"],
    )


async def arch_explainer(state: QAState) -> dict:
    prompt = ARCH_EXPLAINER_PROMPT.format(
        repo_name=state.get("repo_name", ""),
        session_id=state["session_id"],
        question=state["current_question"],
    )
    return await _run_specialist(
        state,
        name="arch_explainer",
        system_prompt=prompt,
        tool_names=["get_file_summary", "get_repo_structure"],
    )


async def dep_tracer(state: QAState) -> dict:
    prompt = DEP_TRACER_PROMPT.format(
        repo_name=state.get("repo_name", ""),
        session_id=state["session_id"],
        question=state["current_question"],
    )
    return await _run_specialist(
        state,
        name="dep_tracer",
        system_prompt=prompt,
        tool_names=["semantic_search", "read_codebase_file"],
    )


# ── synthesizer ───────────────────────────────────────────────────────────────

async def synthesizer(state: QAState) -> dict:
    """Streaming LLM call that fuses specialist outputs into one final answer.

    Tagged "final" so the SSE generator forwards only its tokens to the client.
    """
    outputs = state.get("specialist_outputs") or {}
    if not outputs:
        # Nothing came back from specialists — degrade gracefully.
        specialist_block = "(no specialist findings)"
    else:
        specialist_block = "\n\n".join(
            f"### {name}\n{content}" for name, content in outputs.items()
        )

    prompt = QA_SYNTHESIZER_PROMPT.format(
        repo_name=state.get("repo_name", ""),
        specialist_outputs=specialist_block,
        chat_history=_format_history(state.get("chat_history", [])),
        question=state["current_question"],
    )

    llm = _llm().with_config(
        tags=["final", "agent:synthesizer"], run_name="synthesizer"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"current_answer": response.content}


def update_history(state: QAState) -> dict:
    history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": state["current_question"]},
        {"role": "assistant", "content": state.get("current_answer", "")},
    ]
    return {"chat_history": history}


# ── routing (conditional edge for Send fan-out) ───────────────────────────────

def fan_out_to_specialists(state: QAState) -> list[Any]:
    """Conditional edge: emit one Send per selected specialist.

    Each Send carries the same state forward — LangGraph copies it so parallel
    branches don't mutate each other's view.
    """
    from langgraph.types import Send
    selected = state.get("selected_specialists") or ["code_reader"]
    # Defensive de-dup while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name in SPECIALIST_NAMES and name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        ordered = ["code_reader"]
    return [Send(name, state) for name in ordered]
