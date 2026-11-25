from __future__ import annotations

import json

import httpx
import pytest

from onboardflow_evals.systems.v2 import files_read, parse_sse
from onboardflow_evals.usage_proxy import UsageLedger, create_proxy_app


def _sse(event: dict[str, object]) -> str:
    return f"data: {json.dumps(event)}"


def test_parse_sse_collects_tokens_trace_and_error() -> None:
    lines = [
        _sse({"event": "status", "agent": "code_reader", "text": "code_reader: thinking..."}),
        _sse(
            {
                "event": "status",
                "agent": "code_reader",
                "text": "code_reader: read_codebase_file(src/flask/app.py)",
            }
        ),
        _sse({"event": "status", "agent": "code_reader", "text": ""}),
        "",
        _sse({"event": "token", "text": "Look at "}),
        _sse({"event": "token", "text": "`src/flask/app.py`."}),
        "data: not-json",
        _sse({"event": "done", "chat_history": []}),
    ]
    answer, trace, error = parse_sse(lines)
    assert answer == "Look at `src/flask/app.py`."
    assert error is None
    assert files_read(trace) == ["src/flask/app.py"]


def test_parse_sse_reports_backend_error() -> None:
    _, _, error = parse_sse([_sse({"event": "error", "text": "boom"})])
    assert error == "boom"


async def test_proxy_requests_streaming_usage_and_records_it() -> None:
    sent: list[dict[str, object]] = []
    chunks = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":200}}\n\n',
        "data: [DONE]\n\n",
    ]

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content="".join(chunks).encode()
        )

    ledger = UsageLedger()
    app = create_proxy_app(
        ledger,
        upstream="https://upstream.test/v1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        response = await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "stream": True, "messages": []},
        )

    assert response.text == "".join(chunks)  # content passes through unchanged
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert ledger.totals()[:2] == (1000, 200)
    assert ledger.totals()[2] == pytest.approx((1000 * 0.15 + 200 * 0.60) / 1e6)


async def test_proxy_records_non_streaming_usage() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "usage": {"prompt_tokens": 500}})

    ledger = UsageLedger()
    app = create_proxy_app(
        ledger,
        upstream="https://upstream.test/v1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        await c.post("/v1/embeddings", json={"model": "text-embedding-3-small", "input": ["x"]})

    assert ledger.entries == [("text-embedding-3-small", 500, 0)]
