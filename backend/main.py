import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from graph.graph import ingest_graph, qa_graph
from graph.state import RepoState
from models import (
    IngestRequest, IngestResponse,
    QuestionRequest, QuestionResponse,
    SessionResponse,
)

load_dotenv()

app = FastAPI(title="OSS Onboarding Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Key: session_id (str)  →  Value: fully populated RepoState dict
sessions: dict[str, RepoState] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest):
    """
    Fetch a GitHub repo, run LLM summarization, and store the result in session.
    Call this once per repo. Re-calling with the same session_id overwrites it.
    """
    initial_state: RepoState = {
        "repo_url": req.repo_url,
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

    sessions[req.session_id] = final_state

    return IngestResponse(
        status="ready",
        repo_name=final_state["repo_name"],
        repo_description=final_state["repo_description"],
        summary=final_state["repo_summary"],
        files_fetched=list(final_state["fetched_files"].keys()),
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_endpoint(req: QuestionRequest):
    """
    Answer a question about the ingested repository.
    Session must exist (call /ingest first).
    """
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /ingest first.",
        )

    state = dict(sessions[req.session_id])   # shallow copy — avoid mutating stored state
    state["current_question"] = req.question

    try:
        updated_state = await asyncio.wait_for(
            qa_graph.ainvoke(state),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Answer timed out. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sessions[req.session_id] = updated_state

    return QuestionResponse(
        answer=updated_state["current_answer"],
        chat_history=updated_state["chat_history"],
    )


@app.get("/session/{session_id}", response_model=SessionResponse)
async def session_endpoint(session_id: str):
    """Check whether a session exists and what repo it has loaded."""
    if session_id not in sessions:
        return SessionResponse(exists=False)
    state = sessions[session_id]
    return SessionResponse(
        exists=True,
        repo_name=state["repo_name"],
        files_fetched=list(state["fetched_files"].keys()),
    )


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
