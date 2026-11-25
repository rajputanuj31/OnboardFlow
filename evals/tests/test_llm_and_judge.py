from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from onboardflow_evals.cache import DiskCache
from onboardflow_evals.datasets import Item
from onboardflow_evals.judge import AnswerScore, Judge
from onboardflow_evals.llm import LLMClient, LLMRequest, LLMResult, MissingCredentialError
from onboardflow_evals.pricing import UnknownPriceError, cost_usd
from onboardflow_evals.systems.base import SystemOutput


class FakeAdapter:
    def __init__(self, parsed: dict[str, object] | None = None, text: str = "answer") -> None:
        self.calls = 0
        self.parsed = parsed
        self.text = text

    async def complete(self, request: LLMRequest, schema: type[BaseModel] | None) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text=self.text,
            parsed=self.parsed,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            latency_ms=5.0,
        )


def _request(user: str = "q") -> LLMRequest:
    return LLMRequest(
        provider="anthropic", model="claude-sonnet-5-5", system="s", user=user, max_tokens=10
    )


async def test_repeat_call_is_served_from_cache(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    client = LLMClient(DiskCache(tmp_path), {"anthropic": adapter})

    first = await client.complete(_request())
    second = await client.complete(_request())

    assert adapter.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text


async def test_any_prompt_change_misses_the_cache(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    client = LLMClient(DiskCache(tmp_path), {"anthropic": adapter})
    await client.complete(_request("q1"))
    await client.complete(_request("q2"))
    assert adapter.calls == 2


async def test_missing_provider_fails_loudly(tmp_path: Path) -> None:
    client = LLMClient(DiskCache(tmp_path), {})
    with pytest.raises(MissingCredentialError):
        await client.complete(_request())


def test_unknown_model_price_is_an_error() -> None:
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    with pytest.raises(UnknownPriceError):
        cost_usd("some-unpriced-model", 1, 1)


def test_answer_score_range_and_scale() -> None:
    perfect = AnswerScore(
        justification="", correctness=5, completeness=5, relevance=5, clarity=5, reasoning=5
    )
    worst = AnswerScore(
        justification="", correctness=1, completeness=1, relevance=1, clarity=1, reasoning=1
    )
    assert perfect.score_100 == 100
    assert worst.score_100 == 0
    with pytest.raises(ValidationError):
        AnswerScore(
            justification="", correctness=6, completeness=1, relevance=1, clarity=1, reasoning=1
        )


async def test_judge_sees_reference_but_not_system_name(tmp_path: Path) -> None:
    seen: list[LLMRequest] = []
    scores = {
        "justification": "ok",
        "correctness": 4,
        "completeness": 3,
        "relevance": 5,
        "clarity": 4,
        "reasoning": 3,
    }

    class Capturing(FakeAdapter):
        async def complete(self, request: LLMRequest, schema: type[BaseModel] | None) -> LLMResult:
            seen.append(request)
            return await super().complete(request, schema)

    judge = Judge(
        LLMClient(DiskCache(tmp_path), {"openai": Capturing(parsed=scores)}), "gpt-4o-mini"
    )
    item = Item(
        id="x",
        suite="answer",
        slice="fresh",
        repo="o/r",
        commit="abc",
        question="How?",
        reference_answer="REFERENCE TEXT",
        source="test",
    )
    output = SystemOutput(item_id="x", system="secret-system-name", answer="CANDIDATE TEXT")

    score, cost = await judge.score_answer(item, output)

    assert score.score_100 == pytest.approx((19 - 5) / 20 * 100)
    assert cost == pytest.approx(0.01)
    assert "REFERENCE TEXT" in seen[0].user
    assert "CANDIDATE TEXT" in seen[0].user
    assert "secret-system-name" not in seen[0].user + seen[0].system


async def test_no_context_uses_the_configured_provider(tmp_path: Path) -> None:
    from onboardflow_evals.systems.no_context import NoContextSystem

    seen: list[LLMRequest] = []

    class Capturing(FakeAdapter):
        async def complete(self, request: LLMRequest, schema: type[BaseModel] | None) -> LLMResult:
            seen.append(request)
            return await super().complete(request, schema)

    client = LLMClient(DiskCache(tmp_path), {"openai": Capturing(text="see src/flask/app.py")})
    system = NoContextSystem(client, "openai", "gpt-6.1-terra", "medium", 1000, 1)
    item = Item(
        id="x",
        suite="answer",
        slice="popular",
        repo="pallets/flask",
        commit="c",
        question="Where is the app object?",
        reference_answer="app.py",
        source="t",
    )

    output = await system.answer(item)

    assert seen[0].provider == "openai"
    assert seen[0].model == "gpt-6.1-terra"
    assert "pallets/flask" in seen[0].system
    assert output.files == ["src/flask/app.py"]
