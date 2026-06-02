import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from graph.graph import ingest_graph, pr_review_graph, qa_graph
from graph.state import PRReviewState, QAState, RepoState
from mcp_clients import MCPRegistry, is_github_mcp_available
from models import (
    IngestRequest, IngestResponse,
    PRReviewRequest,
    QuestionRequest, QuestionResponse,
    SessionResponse,
)
from onboard_github import parse_github_pr_url
from utils.session_db import init_db, save_session, get_session, get_sessions_count
from config import settings

load_dotenv()
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.mcp = MCPRegistry()
    await app.state.mcp.start()
    try:
        yield
    finally:
        await app.state.mcp.stop()


app = FastAPI(title="OnboardFlow", version="2.0.0", lifespan=lifespan)

frontend_url = os.getenv("FRONTEND_URL")
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if frontend_url:
    # Support multiple origins if needed by splitting on comma, or just append
    if "," in frontend_url:
        allowed_origins.extend([o.strip() for o in frontend_url.split(",") if o.strip()])
    else:
        allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage is handled persistently via SQLite database file (.cache/sessions.db)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest, x_model_api_key: str | None = Header(None)):
    """
    Fetch a GitHub repo, run LLM summarization, and store the result in session.
    Call this once per repo. Re-calling with the same session_id overwrites it.
    """
    api_key = x_model_api_key or settings.openai_api_key
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Model API Key is required. Please enter it in the Ingest screen."
        )

    initial_state: RepoState = {
        "session_id": req.session_id,
        "repo_url": req.repo_url,
        "model_api_key": api_key,
        # Fetch fields (filled by ingest node)
        "repo_name": "",
        "repo_description": "",
        "repo_language": "",
        "repo_stars": 0,
        "repo_license": "",
        "repo_topics": [],
        "repo_structure": [],
        "readme": "",
        "contributing": "",
        "fetched_files": {},
        # LLM fields (filled by summarize node)
        "repo_summary": "",
        "architecture_notes": "",
        # Conversation
        "chat_history": [],
        "current_question": "",
        "current_answer": "",
        # ReAct loop state
        "needs_tool_call": False,
        "tool_calls_made": 0,
        "messages": [],
        "rag_context": "",
    }

    try:
        final_state = await asyncio.wait_for(
            ingest_graph.ainvoke(initial_state),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timed out fetching/summarizing repo. Try a smaller repo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    save_session(req.session_id, final_state)

    return IngestResponse(
        status="ready",
        repo_name=final_state["repo_name"],
        repo_description=final_state["repo_description"],
        summary=final_state["repo_summary"],
        files_fetched=list(final_state["fetched_files"].keys()),
        repo_language=final_state["repo_language"],
        repo_stars=final_state["repo_stars"],
        architecture=final_state["architecture_notes"],
    )


@app.post("/ask")
async def ask_endpoint(req: QuestionRequest, x_model_api_key: str | None = Header(None)):
    """Answer a question about the ingested repository (multi-agent) and stream SSE."""
    persisted = get_session(req.session_id)
    if not persisted:
        raise HTTPException(status_code=404, detail="Session not found. Call /ingest first.")

    api_key = x_model_api_key or persisted.get("model_api_key") or settings.openai_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="Model API Key is required. Please configure it.")

    chat_history = req.chat_history if req.chat_history is not None else persisted.get("chat_history", [])

    qa_state: QAState = {
        "session_id": req.session_id,
        "model_api_key": api_key,
        "repo_name": persisted.get("repo_name", ""),
        "repo_summary": persisted.get("repo_summary", ""),
        "architecture_notes": persisted.get("architecture_notes", ""),
        "repo_structure": persisted.get("repo_structure", []) or [],
        "chat_history": chat_history,
        "current_question": req.question,
        "current_answer": "",
        "selected_specialists": [],
        "specialist_outputs": {},
    }

    return StreamingResponse(
        _stream_multi_agent(
            graph=qa_graph,
            initial_state=qa_state,
            persisted_state=persisted,
            persist_key="chat_history",
            user_text=req.question,
            done_payload_builder=lambda chunks: {
                "chat_history": list(chat_history) + [
                    {"role": "user", "content": req.question},
                    {"role": "assistant", "content": "".join(chunks)},
                ]
            },
            persist_done_history=True,
            session_id=req.session_id,
        ),
        media_type="text/event-stream",
    )


@app.post("/review-pr")
async def review_pr_endpoint(req: PRReviewRequest, x_model_api_key: str | None = Header(None)):
    """Multi-agent PR review. Streams SSE: per-agent status pills + synthesizer tokens."""
    if not is_github_mcp_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "GitHub MCP server is not available. Install github-mcp-server "
                "(go install github.com/github/github-mcp-server@latest) and restart the backend."
            ),
        )

    try:
        owner, repo, pr_number = parse_github_pr_url(req.pr_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_key = x_model_api_key or settings.openai_api_key
    if req.session_id:
        s = get_session(req.session_id)
        if s and not api_key:
            api_key = s.get("model_api_key", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Model API Key is required.")

    pr_state: PRReviewState = {
        "session_id": req.session_id,
        "model_api_key": api_key,
        "pr_url": req.pr_url,
        "pr_owner": owner,
        "pr_repo": repo,
        "pr_number": pr_number,
        "pr_metadata": {},
        "pr_files": [],
        "pr_diff": "",
        "specialist_outputs": {},
        "final_review": "",
    }

    return StreamingResponse(
        _stream_multi_agent(
            graph=pr_review_graph,
            initial_state=pr_state,
            persisted_state=None,
            persist_key=None,
            user_text=req.pr_url,
            done_payload_builder=lambda chunks: {"review": "".join(chunks)},
            persist_done_history=False,
            session_id=None,
        ),
        media_type="text/event-stream",
    )


def _agent_from_tags(tags: list[str] | None) -> str | None:
    for t in tags or []:
        if isinstance(t, str) and t.startswith("agent:"):
            return t.split(":", 1)[1]
    return None


async def _stream_multi_agent(
    *,
    graph,
    initial_state: dict,
    persisted_state: dict | None,
    persist_key: str | None,
    user_text: str,
    done_payload_builder,
    persist_done_history: bool,
    session_id: str | None,
):
    final_chunks: list[str] = []
    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name")
            tags = event.get("tags") or []
            agent = _agent_from_tags(tags)

            # 1. Stream tokens ONLY from the synthesizer (tagged "final").
            if kind == "on_chat_model_stream" and "final" in tags:
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    final_chunks.append(chunk.content)
                    yield f"data: {json.dumps({'event': 'token', 'text': chunk.content})}\n\n"
                continue

            # 2. Per-agent activity → status pills.
            if agent and agent not in ("synthesizer", "router") and kind == "on_chain_start":
                yield f"data: {json.dumps({'event': 'status', 'agent': agent, 'text': f'{agent}: thinking...'})}\n\n"
            elif agent and agent not in ("synthesizer", "router") and kind == "on_chain_end":
                yield f"data: {json.dumps({'event': 'status', 'agent': agent, 'text': ''})}\n\n"
            elif kind == "on_tool_start" and agent:
                tool_input = event["data"].get("input", {})
                hint = ""
                if isinstance(tool_input, dict):
                    hint = tool_input.get("filepath") or tool_input.get("query") or ""
                label = f"{agent}: {name}({hint})" if hint else f"{agent}: {name}"
                yield f"data: {json.dumps({'event': 'status', 'agent': agent, 'text': label})}\n\n"
            elif kind == "on_tool_end" and agent:
                yield f"data: {json.dumps({'event': 'status', 'agent': agent, 'text': ''})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'text': str(e)})}\n\n"
        return

    payload = {"event": "done", **done_payload_builder(final_chunks)}
    yield f"data: {json.dumps(payload)}\n\n"

    if persist_done_history and session_id and persisted_state is not None:
        # Persist updated chat_history back to the session DB
        persisted_state["chat_history"] = payload.get("chat_history", persisted_state.get("chat_history", []))
        save_session(session_id, persisted_state)


@app.get("/session/{session_id}", response_model=SessionResponse)
async def session_endpoint(session_id: str):
    """Check whether a session exists and what repo it has loaded."""
    state = get_session(session_id)
    if not state:
        return SessionResponse(exists=False)
    return SessionResponse(
        exists=True,
        repo_name=state["repo_name"],
        files_fetched=list(state["fetched_files"].keys()),
        repo_language=state["repo_language"],
        repo_stars=state["repo_stars"],
        architecture=state["architecture_notes"],
        chat_history=state.get("chat_history", []),
    )


@app.get("/health")
def health():
    mcp_status: dict = {}
    mcp = getattr(app.state, "mcp", None)
    if mcp is not None:
        mcp_status = mcp.status()
    return {
        "status": "ok",
        "sessions": get_sessions_count(),
        "mcp": mcp_status,
    }


# ── Dev entrypoint (triggered reload V2) ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
