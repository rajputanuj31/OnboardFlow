"""Baseline: the answering model with no access to the repository.

Measures what the model already knows from training. On popular repos this is high
(memorisation); the gap between this and a retrieval system on *fresh* repos is the part
retrieval actually earns.
"""

from __future__ import annotations

from typing import Literal

from onboardflow_evals.datasets import Item, render_question
from onboardflow_evals.llm import Effort, LLMClient, LLMRequest
from onboardflow_evals.paths import extract_paths
from onboardflow_evals.pricing import cost_usd
from onboardflow_evals.systems.base import SystemOutput

SYSTEM_PROMPT = (
    "You are an expert software engineer answering questions about the {repo} GitHub "
    "repository. You do not have access to the repository's code in this conversation: "
    "answer only from what you already know. Be specific (file paths, classes, functions). "
    "If you are not confident that something exists in the repository, say so plainly "
    "instead of guessing."
)

# Rough per-item token assumptions for the pre-run spend estimate (deliberately high).
_EST_OUTPUT_TOKENS = 3000


class NoContextSystem:
    name = "no-context"

    def __init__(
        self,
        llm: LLMClient,
        provider: Literal["openai", "anthropic"],
        model: str,
        effort: Effort,
        max_tokens: int,
        concurrency: int,
    ) -> None:
        self.llm = llm
        self.provider: Literal["openai", "anthropic"] = provider
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.concurrency = concurrency

    def _request(self, item: Item) -> LLMRequest:
        return LLMRequest(
            provider=self.provider,
            model=self.model,
            system=SYSTEM_PROMPT.format(repo=item.repo),
            user=render_question(item),
            max_tokens=self.max_tokens,
            effort=self.effort,
        )

    async def answer(self, item: Item) -> SystemOutput:
        result = await self.llm.complete(self._request(item))
        return SystemOutput(
            item_id=item.id,
            system=self.name,
            answer=result.text,
            files=extract_paths(result.text),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            stop_reason=result.stop_reason,
            cached=result.cached,
        )

    def estimate_cost_usd(self, items: list[Item]) -> float:
        total = 0.0
        for item in items:
            request = self._request(item)
            input_tokens = (len(request.system) + len(request.user)) // 3
            total += cost_usd(self.model, input_tokens, _EST_OUTPUT_TOKENS)
        return total
