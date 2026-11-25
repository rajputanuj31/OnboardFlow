"""A tiny OpenAI-compatible pass-through proxy that records token usage.

v2 calls OpenAI through LangChain and never reports tokens. Pointing its OpenAI client
at this proxy (OPENAI_BASE_URL) lets the harness measure v2's real cost per question
without changing v2's code. For streaming chat completions the proxy asks OpenAI to
include usage in the final chunk (`stream_options.include_usage`), which does not change
the content v2 receives.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from onboardflow_evals.pricing import cost_usd

UPSTREAM = "https://api.openai.com/v1"
_FORWARD_HEADERS = ("authorization", "content-type", "openai-organization", "openai-project")


@dataclass
class UsageLedger:
    entries: list[tuple[str, int, int]] = field(default_factory=list)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.entries.append((model, input_tokens, output_tokens))

    def reset(self) -> None:
        self.entries.clear()

    def totals(self) -> tuple[int, int, float]:
        input_tokens = sum(e[1] for e in self.entries)
        output_tokens = sum(e[2] for e in self.entries)
        cost = sum(cost_usd(e[0], e[1], e[2]) for e in self.entries)
        return input_tokens, output_tokens, cost


def _record(ledger: UsageLedger, model: str, usage: object) -> None:
    if not isinstance(usage, dict):
        return
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    if input_tokens or output_tokens:
        ledger.add(model, input_tokens, output_tokens)


def create_proxy_app(
    ledger: UsageLedger, upstream: str = UPSTREAM, client: httpx.AsyncClient | None = None
) -> Starlette:
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def forward(request: Request) -> Response:
        body = await request.body()
        payload = json.loads(body) if body else {}
        model = str(payload.get("model", "unknown")) if isinstance(payload, dict) else "unknown"
        if isinstance(payload, dict) and payload.get("stream"):
            payload.setdefault("stream_options", {})["include_usage"] = True
            body = json.dumps(payload).encode()

        headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARD_HEADERS}
        headers["accept-encoding"] = "identity"  # plain bytes so usage can be parsed
        upstream_request = http.build_request(
            request.method,
            f"{upstream}/{request.path_params['path']}",
            params=request.query_params,
            headers=headers,
            content=body,
        )
        upstream_response = await http.send(upstream_request, stream=True)
        content_type = upstream_response.headers.get("content-type", "")

        if "text/event-stream" in content_type:

            async def relay() -> AsyncIterator[bytes]:
                buffer = ""
                try:
                    async for chunk in upstream_response.aiter_bytes():
                        yield chunk
                        buffer += chunk.decode("utf-8", errors="replace")
                        *lines, buffer = buffer.split("\n")
                        for line in lines:
                            if line.startswith("data: ") and '"usage"' in line:
                                try:
                                    data = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    continue
                                _record(ledger, model, data.get("usage"))
                finally:
                    await upstream_response.aclose()

            return StreamingResponse(
                relay(), status_code=upstream_response.status_code, media_type=content_type
            )

        content = await upstream_response.aread()
        await upstream_response.aclose()
        if upstream_response.status_code == 200 and "json" in content_type:
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                _record(ledger, model, json.loads(content).get("usage"))
        return Response(content, status_code=upstream_response.status_code, media_type=content_type)

    return Starlette(routes=[Route("/v1/{path:path}", forward, methods=["GET", "POST"])])
