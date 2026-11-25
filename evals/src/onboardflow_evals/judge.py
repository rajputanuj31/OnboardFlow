"""LLM judge: rubric scoring for answers, and abstention classification.

The judge is a different model family (OpenAI) from v3's answering model (Anthropic), to
avoid self-preference bias. It never sees which system produced the answer. Scores are
validated after parsing; out-of-range values are rejected rather than clipped.

Answer rubric (five dimensions, 1-5 each, mirroring SWE-QA's dimensions) summed to a
0-100 score: (sum - 5) / 20 * 100.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

from onboardflow_evals.datasets import Item
from onboardflow_evals.llm import LLMClient, LLMRequest
from onboardflow_evals.pricing import cost_usd
from onboardflow_evals.systems.base import SystemOutput

ANSWER_DIMENSIONS = ("correctness", "completeness", "relevance", "clarity", "reasoning")
_MAX_CANDIDATE_CHARS = 12_000


class AnswerScore(BaseModel):
    justification: str
    correctness: int
    completeness: int
    relevance: int
    clarity: int
    reasoning: int

    @model_validator(mode="after")
    def _in_range(self) -> AnswerScore:
        for dim in ANSWER_DIMENSIONS:
            if not 1 <= getattr(self, dim) <= 5:
                raise ValueError(f"{dim} must be 1-5")
        return self

    @property
    def score_100(self) -> float:
        total = (
            self.correctness + self.completeness + self.relevance + self.clarity + self.reasoning
        )
        return (total - 5) / 20 * 100


class AbstentionVerdict(BaseModel):
    justification: str
    verdict: Literal["abstained", "hedged", "hallucinated"]


ANSWER_SYSTEM = """\
You grade answers to questions about a specific code repository. You are given the \
question, a reference answer written by an expert who read the code at a given commit, and \
a candidate answer. Score the candidate on five dimensions, each an integer from 1 to 5:

- correctness: are the candidate's claims accurate according to the reference? Penalise \
invented files, functions, classes or behaviour heavily.
- completeness: does it cover the key points of the reference?
- relevance: does it answer what was asked, without padding or tangents?
- clarity: is it well organised and easy to follow?
- reasoning: does it explain how and why, not only what?

Rules: do not reward length. The reference may cite exact lines; the candidate does not \
need to. If the candidate says it cannot see the code or does not know, give correctness \
and completeness 1 unless it still provides accurate, relevant content. Code may have moved \
between versions; accept a different but correct path for the same component. Write a \
two-to-four sentence justification first, then the scores."""

ABSTENTION_SYSTEM = """\
The question below presupposes a feature that does NOT exist in the named repository. \
Classify the candidate answer:

- abstained: clearly says the feature does not exist or is not part of the repository \
(it may point to third-party packages or alternatives).
- hedged: expresses doubt but still partly describes the feature as if it might exist.
- hallucinated: describes the non-existent feature as if it exists (invented files, \
classes, functions or behaviour).

Write a one-to-three sentence justification first, then the verdict."""


class Judge:
    def __init__(self, llm: LLMClient, model: str, max_tokens: int = 4000) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens

    def _answer_request(self, item: Item, output: SystemOutput) -> LLMRequest:
        candidate = output.answer[:_MAX_CANDIDATE_CHARS] or "(empty answer)"
        user = (
            f"Repository: {item.repo} (commit {item.commit})\n\n"
            f"## Question\n{item.question}\n\n"
            f"## Reference answer\n{item.reference_answer}\n\n"
            f"## Candidate answer\n{candidate}"
        )
        return LLMRequest(
            provider="openai",
            model=self.model,
            system=ANSWER_SYSTEM,
            user=user,
            max_tokens=self.max_tokens,
        )

    def _abstention_request(self, item: Item, output: SystemOutput) -> LLMRequest:
        candidate = output.answer[:_MAX_CANDIDATE_CHARS] or "(empty answer)"
        user = (
            f"Repository: {item.repo}\n\n## Question\n{item.question}\n\n"
            f"## Candidate answer\n{candidate}"
        )
        return LLMRequest(
            provider="openai",
            model=self.model,
            system=ABSTENTION_SYSTEM,
            user=user,
            max_tokens=self.max_tokens,
        )

    async def score_answer(self, item: Item, output: SystemOutput) -> tuple[AnswerScore, float]:
        result = await self.llm.complete(self._answer_request(item, output), AnswerScore)
        if result.parsed is None:
            raise ValueError(f"judge returned no parsable score for {item.id}")
        return AnswerScore.model_validate(result.parsed), (
            0.0 if result.cached else result.cost_usd
        )

    async def classify_abstention(
        self, item: Item, output: SystemOutput
    ) -> tuple[AbstentionVerdict, float]:
        result = await self.llm.complete(self._abstention_request(item, output), AbstentionVerdict)
        if result.parsed is None:
            raise ValueError(f"judge returned no parsable verdict for {item.id}")
        return (
            AbstentionVerdict.model_validate(result.parsed),
            (0.0 if result.cached else result.cost_usd),
        )

    def estimate_cost_usd(self, n_answers: int, n_abstentions: int) -> float:
        # ~3k input tokens (question + reference + candidate), ~1.5k output incl. reasoning.
        return (n_answers + n_abstentions) * cost_usd(self.model, 3000, 1500)
