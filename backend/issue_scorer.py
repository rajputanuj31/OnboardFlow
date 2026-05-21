from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from models import MatchedIssue


MODEL_NAME = "gpt-4o-mini"


PROMPT = ChatPromptTemplate.from_template(
    """
You are ranking GitHub issues for a developer
looking to contribute to open source.

Developer profile:
- Stack: {stack}
- Interests: {interests}

Issues to rank:
{issues}

For each issue:
- assign a relevance_score (1-10)
- write a one-sentence match_reason

Scoring criteria:
- 8-10 → directly uses their stack, clear scope
- 5-7 → related stack or interest area
- 1-4 → weak match, vague, or overly complex

Rules:
- match_reason must be specific
- mention technologies/domains explicitly
- avoid generic statements like:
  "matches your stack"

Example:
"Uses TypeScript and React patterns
you already work with."
"""
)


class ScoredIssue(BaseModel):
    repo: str
    issue_number: int

    relevance_score: int = Field(
        description="1-10 issue relevance score"
    )

    match_reason: str = Field(
        description="Why this issue matches the developer"
    )


class ScoringOutput(BaseModel):
    scored: list[ScoredIssue]


class IssueScorer:
    def __init__(self):
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
            api_key=settings.openai_api_key
        )

        self.chain = (
            PROMPT
            | llm.with_structured_output(ScoringOutput)
        )

    async def score(
        self,
        issues: list[MatchedIssue],
        stack: list[str],
        interests: list[str],
    ) -> list[MatchedIssue]:

        if not issues:
            return []

        issues_text = "\n".join([
            (
                f"- repo:{issue.repo} "
                f"#{issue.issue_number} | "
                f"{issue.title} | "
                f"labels:{','.join(issue.labels)}"
            )
            for issue in issues
        ])

        result = await self.chain.ainvoke({
            "stack": stack,
            "interests": interests,
            "issues": issues_text,
        })

        output = (
            ScoringOutput(**result)
            if isinstance(result, dict)
            else result
        )

        score_map = {
            (s.repo, s.issue_number): s
            for s in output.scored
        }

        scored_issues = []

        for issue in issues:
            key = (issue.repo, issue.issue_number)

            if key in score_map:
                scored = score_map[key]

                scored_issues.append(
                    issue.model_copy(update={
                        "match_reason": scored.match_reason,
                        "_score": scored.relevance_score,
                    })
                )
            else:
                scored_issues.append(issue)

        scored_issues.sort(
            key=lambda issue: score_map.get(
                (issue.repo, issue.issue_number),
                ScoredIssue(
                    repo="",
                    issue_number=0,
                    relevance_score=0,
                    match_reason=""
                )
            ).relevance_score,
            reverse=True,
        )

        return scored_issues