import asyncio
import json
import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from graph.graph import ingest_graph, qa_graph
from graph.state import RepoState
from models import (
    IngestRequest, IngestResponse,
    QuestionRequest, QuestionResponse,
    SessionResponse,
)
from utils.session_db import init_db, save_session, get_session, get_sessions_count
from config import settings

load_dotenv()


app = FastAPI(title="OnboardFlow", version="2.0.0")


@app.on_event("startup")
def on_startup():
    init_db()

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
    """
    Answer a question about the ingested repository and stream response SSE.
    """
    state = get_session(req.session_id)
    if not state:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /ingest first.",
        )
    
    api_key = x_model_api_key or state.get("model_api_key") or settings.openai_api_key
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Model API Key is required. Please configure it."
        )
    
    state["current_question"] = req.question
    state["model_api_key"] = api_key
    if req.chat_history is not None:
        state["chat_history"] = req.chat_history
        
    # Reset loop/turn state for the new question
    state["needs_tool_call"] = False
    state["tool_calls_made"] = 0
    state["messages"] = []
    state["rag_context"] = ""

    async def event_generator():
        final_answer_chunks = []
        try:
            async for event in qa_graph.astream_events(state, version="v2"):
                kind = event.get("event")
                name = event.get("name")
                
                # 1. Output LLM Tokens
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        final_answer_chunks.append(chunk.content)
                        yield f"data: {json.dumps({'event': 'token', 'text': chunk.content})}\n\n"
                
                # 2. Output Tool Executions
                elif kind == "on_tool_start" and name == "read_codebase_file":
                    filepath = event["data"].get("input", {}).get("filepath", "")
                    yield f"data: {json.dumps({'event': 'status', 'text': f'Reading file: {filepath}'})}\n\n"
                    
                elif kind == "on_tool_end" and name == "read_codebase_file":
                    yield f"data: {json.dumps({'event': 'status', 'text': ''})}\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'text': str(e)})}\n\n"
            return

        final_answer = "".join(final_answer_chunks)
        updated_history = list(state.get("chat_history", [])) + [
            {"role": "user", "content": req.question},
            {"role": "assistant", "content": final_answer},
        ]
        
        yield f"data: {json.dumps({'event': 'done', 'chat_history': updated_history})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    return {"status": "ok", "sessions": get_sessions_count()}


# ── Dev entrypoint (triggered reload V2) ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
