from pydantic import BaseModel, Field


class UserInput(BaseModel):
    username: str
    stack: list[str] = Field(description="e.g. ['TypeScript', 'React', 'LangChain']")
    interests: list[str] = Field(description="e.g. ['frontend', 'ai/ml']")


class MatchedIssue(BaseModel):
    repo: str
    issue_number: int
    title: str
    url: str
    labels: list[str]
    match_reason: str


class IssueSearchResult(BaseModel):
    issues: list[MatchedIssue]
    searched_repos: list[str]
