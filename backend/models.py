# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


class RepoOnboardRequest(BaseModel):
    repo_url: str = Field(description="e.g. https://github.com/Significant-Gravitas/AutoGPT")


class RepoDetails(BaseModel):
    repo: str
    readme: str
    contributing: str
    repo_stars: int
    repo_language: str
    license: str
