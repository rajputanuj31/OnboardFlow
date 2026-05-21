from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings


MODEL_NAME = "gpt-4o-mini"

class GitHubSearchTerms(BaseModel):
    languages: list[str] = Field(
        description="Valid GitHub language names"
    )

    topics: list[str] = Field(
        description="GitHub repo topic names"
    )

PROMPT = ChatPromptTemplate.from_template(
    """
You are helping find open-source repos on GitHub
that match a developer's profile.

The developer provided:
- Stack: {stack}
- Interests: {interests}

Your job is to translate this into GitHub search terms.

Rules:
- languages: only valid GitHub language filter values
- topics: lowercase-hyphenated GitHub topic strings
- max 5 topics total
- Return 1-3 languages and 2-5 topics
"""
)

class QueryBuilder:
    def __init__(self):
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
            api_key=settings.openai_api_key
        )

        self.chain = (
            PROMPT
            | llm.with_structured_output(GitHubSearchTerms)
        )

    async def build(
        self,
        stack: list[str],
        interests: list[str]
    ) -> GitHubSearchTerms:
        result = await self.chain.ainvoke({
            "stack": stack,
            "interests": interests,
        })
        return GitHubSearchTerms(**result) if isinstance(result, dict) else result