"""One cached interface over the model providers the harness uses.

- OpenAI: the answering model (no-context baseline, and v3) and the judge. The judge is a
  different, stronger model than the answerer; same-provider bias is checked against human
  labels in build step 11.
- Anthropic: optional, only as an extra model in the step 7 model comparison.

Every call goes through `LLMClient.complete`, which serves repeats from the disk cache
and records tokens, cost and latency, so reports can show $/question for every system.
"""

from __future__ import annotations

import time
from typing import Literal, Protocol

import anthropic
import openai
from anthropic.types import MessageParam, OutputConfigParam
from openai.types.shared_params import Reasoning
from pydantic import BaseModel

from onboardflow_evals.cache import DiskCache, cache_key
from onboardflow_evals.pricing import cost_usd

Effort = Literal["low", "medium", "high", "xhigh", "max"]
# OpenAI reasoning effort has no "max"; the closest level is used.
_OPENAI_EFFORT: dict[str, Literal["low", "medium", "high", "xhigh"]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
}
CACHE_VERSION = 1  # bump to invalidate every cached call (e.g. after a parsing fix)


class LLMRequest(BaseModel):
    provider: Literal["anthropic", "openai"]
    model: str
    system: str
    user: str
    max_tokens: int
    effort: Effort | None = None


class LLMResult(BaseModel):
    text: str
    parsed: dict[str, object] | None = None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    stop_reason: str | None = None
    cached: bool = False


class MissingCredentialError(RuntimeError):
    pass


class Adapter(Protocol):
    async def complete(self, request: LLMRequest, schema: type[BaseModel] | None) -> LLMResult: ...


class LLMClient:
    def __init__(self, cache: DiskCache, adapters: dict[str, Adapter]) -> None:
        self.cache = cache
        self.adapters = adapters

    async def complete(
        self, request: LLMRequest, schema: type[BaseModel] | None = None
    ) -> LLMResult:
        key = cache_key(
            {
                "v": CACHE_VERSION,
                "request": request.model_dump(),
                "schema": schema.model_json_schema() if schema else None,
            }
        )
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResult.model_validate({**hit, "cached": True})
        adapter = self.adapters.get(request.provider)
        if adapter is None:
            raise MissingCredentialError(
                f"No {request.provider} client configured. Set the API key in the repo-root .env."
            )
        result = await adapter.complete(request, schema)
        self.cache.set(key, result.model_dump(exclude={"cached"}))
        return result


class AnthropicAdapter:
    def __init__(self, api_key: str) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=4)

    async def complete(self, request: LLMRequest, schema: type[BaseModel] | None) -> LLMResult:
        if schema is not None:
            raise NotImplementedError("Structured output via Anthropic is not used by the harness")
        started = time.perf_counter()
        # No refusal fallbacks on purpose: a fallback silently swaps the model under test.
        # A refusal is recorded as such (stop_reason) and scored as a failed answer.
        messages: list[MessageParam] = [{"role": "user", "content": request.user}]
        output_config: OutputConfigParam | anthropic.Omit = (
            {"effort": request.effort} if request.effort else anthropic.omit
        )
        response = await self.client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=messages,
            output_config=output_config,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost_usd(
                request.model, response.usage.input_tokens, response.usage.output_tokens
            ),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            stop_reason=response.stop_reason,
        )


class OpenAIAdapter:
    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key, max_retries=4)

    async def complete(self, request: LLMRequest, schema: type[BaseModel] | None) -> LLMResult:
        started = time.perf_counter()
        reasoning: Reasoning | openai.Omit = (
            {"effort": _OPENAI_EFFORT[request.effort]} if request.effort else openai.omit
        )
        parsed_model: BaseModel | None = None
        if schema is not None:
            parsed_response = await self.client.responses.parse(
                model=request.model,
                instructions=request.system,
                input=request.user,
                text_format=schema,
                max_output_tokens=request.max_tokens,
                reasoning=reasoning,
                store=False,
            )
            parsed_model = parsed_response.output_parsed
            usage, text, status = (
                parsed_response.usage,
                parsed_response.output_text,
                parsed_response.status,
            )
        else:
            response = await self.client.responses.create(
                model=request.model,
                instructions=request.system,
                input=request.user,
                max_output_tokens=request.max_tokens,
                reasoning=reasoning,
                store=False,
            )
            usage, text, status = response.usage, response.output_text, response.status
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        return LLMResult(
            text=text,
            parsed=parsed_model.model_dump() if parsed_model is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(request.model, input_tokens, output_tokens),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            stop_reason=status,
        )


def build_client(cache: DiskCache, anthropic_key: str | None, openai_key: str | None) -> LLMClient:
    adapters: dict[str, Adapter] = {}
    if anthropic_key:
        adapters["anthropic"] = AnthropicAdapter(anthropic_key)
    if openai_key:
        adapters["openai"] = OpenAIAdapter(openai_key)
    return LLMClient(cache, adapters)
