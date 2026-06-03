# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


# ── Internal fetch models (used by onboard_github.py) ────────────────────────

class RepoDetails(BaseModel):
    repo: str
    readme: str
    contributing: str
    repo_stars: int
    repo_language: str
    license: str
    description: str
    topics: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    structure: list[str] = Field(default_factory=list)


# ── API request / response models ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    repo_url: str = Field(description="e.g. https://github.com/psf/requests")
    session_id: str = Field(description="Client-generated unique session identifier")


class IngestResponse(BaseModel):
    status: str                  # "ready"
    repo_name: str
    repo_description: str
    summary: str
    files_fetched: list[str]
    repo_language: str
    repo_stars: int
    architecture: str


class QuestionRequest(BaseModel):
    session_id: str
    question: str
    chat_history: list[dict] | None = None


class SessionResponse(BaseModel):
    exists: bool
    repo_name: str = ""
    files_fetched: list[str] = Field(default_factory=list)
    repo_language: str = ""
    repo_stars: int = 0
    architecture: str = ""
    chat_history: list[dict] = Field(default_factory=list)


class PRReviewRequest(BaseModel):
    pr_url: str = Field(description="e.g. https://github.com/owner/repo/pull/123")
    session_id: str | None = Field(
        default=None,
        description="Optional: provide if the repo has been ingested. Lets agents read codebase context."
    )
