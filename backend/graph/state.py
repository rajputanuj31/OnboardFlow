from typing import Annotated, TypedDict


def _merge_dict(a: dict, b: dict) -> dict:
    """Reducer for parallel LangGraph branches that all write to the same dict field."""
    return {**(a or {}), **(b or {})}


class RepoState(TypedDict):
    """State for the ingest graph + the long-lived session snapshot persisted in SQLite.

    The Q&A and PR-review flows use their own narrower TypedDicts (QAState,
    PRReviewState) — `RepoState` is intentionally limited to what /ingest fills
    and what later flows hydrate from.

    NOTE: the user's model API key must NEVER be added to any graph state —
    state is serialized into LangSmith traces and the persisted session DB.
    It travels via a request-scoped ContextVar instead (see graph/helpers.py).
    """
    session_id: str
    repo_url: str

    # Fetched from GitHub (filled by ingest_repo)
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

    # LLM-generated (filled by summarize_* nodes)
    repo_summary: str
    architecture_notes: str

    # Conversation history (kept on RepoState so /session can return it)
    chat_history: list[dict]


class QAState(TypedDict):
    """State for the multi-agent Q&A graph.

    Hydrated from a persisted RepoState (see main.py::/ask) plus the user's
    current question. Specialist nodes write into `specialist_outputs` in
    parallel — the Annotated reducer merges their per-key writes.
    """
    session_id: str

    # Repo context (read-only during Q&A)
    repo_name: str
    repo_summary: str
    architecture_notes: str
    repo_structure: list[str]

    # Conversation
    chat_history: list[dict]
    current_question: str
    current_answer: str

    # Router output
    selected_specialists: list[str]

    # Parallel specialist results: keys are specialist node names.
    specialist_outputs: Annotated[dict[str, str], _merge_dict]


class PRReviewState(TypedDict):
    """State for the PR review graph.

    `session_id` is optional — when present, agents get richer codebase context
    via the Codebase MCP tools. Without it, agents work from the diff alone.
    """
    session_id: str | None

    # Input
    pr_url: str
    pr_owner: str
    pr_repo: str
    pr_number: int

    # Populated by fetch_pr node (via GitHub MCP)
    pr_metadata: dict
    pr_files: list[dict]
    pr_diff: str

    # Parallel specialist results: code_explainer, impact_analyst, risk_assessor.
    specialist_outputs: Annotated[dict[str, str], _merge_dict]

    # Final synthesized review
    final_review: str
