from typing import TypedDict

class RepoState(TypedDict):
    session_id: str
    repo_url: str
    model_api_key: str

    # ── Fetched from GitHub (filled by ingest node) ──────────────────────────
    repo_name: str
    repo_description: str
    repo_language: str
    repo_stars: int
    repo_license: str
    repo_topics: list[str]
    repo_structure: list[str]
    readme: str
    contributing: str
    fetched_files: dict[str, str]

    # ── LLM-generated (filled by summarize node) ──────────────────────────────
    repo_summary: str
    architecture_notes: str

    # ── Conversation (filled by answer node) ──────────────────────────────────
    chat_history: list[dict]
    current_question: str
    current_answer: str

    # ── ReAct loop state (managed by answer node) ─────────────────────────────
    needs_tool_call: bool
    tool_calls_made: int
    messages: list
    rag_context: str
