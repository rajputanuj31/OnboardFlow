"""Baseline: OnboardFlow v2 exactly as shipped (git tag v2.0.0), measured over its HTTP API.

The harness extracts v2's backend from the tag into evals/.cache/v2 (git archive, so the
working tree is untouched), installs its pinned requirements into a Python 3.12 venv, runs
it against a separate Qdrant (compose profile "baseline", so v3's data is untouched), and
routes its OpenAI traffic through the usage proxy to measure tokens and cost.

Known limitation, reported with the results: v2 can only ingest a repo's latest default
branch, not the commit a benchmark question refers to.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Self

import httpx
import uvicorn

from onboardflow_evals.datasets import Item, render_question
from onboardflow_evals.paths import extract_paths, normalize
from onboardflow_evals.systems.base import SystemOutput
from onboardflow_evals.usage_proxy import UsageLedger, create_proxy_app

V2_TAG = "v2.0.0"
V2_PYTHON = "3.12"
_STATUS_RE = re.compile(r"^(?P<agent>[\w-]+): (?P<tool>[\w-]+)\((?P<hint>.*)\)$")
_EST_COST_PER_QUESTION = 0.01  # gpt-4o-mini multi-agent run, deliberately high
_EST_COST_PER_INGEST = 0.05


def parse_sse(lines: Iterable[str]) -> tuple[str, list[str], str | None]:
    """Return (answer, tool_trace, error) from v2's SSE `data:` lines."""
    tokens: list[str] = []
    trace: list[str] = []
    error: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        kind = event.get("event")
        if kind == "token":
            tokens.append(str(event.get("text", "")))
        elif kind == "status" and event.get("text"):
            trace.append(str(event["text"]))
        elif kind == "error":
            error = str(event.get("text", "unknown error"))
    return "".join(tokens), trace, error


def files_read(trace: list[str]) -> list[str]:
    """Files v2's agents opened with read_codebase_file, in order."""
    seen: list[str] = []
    for entry in trace:
        match = _STATUS_RE.match(entry)
        if match and match["tool"] == "read_codebase_file" and match["hint"]:
            path = normalize(match["hint"])
            if path not in seen:
                seen.append(path)
    return seen


async def _run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    text = output.decode(errors="replace")
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(args)}\n{text}")
    return text


async def prepare_v2(cache_dir: Path, repo_root: Path) -> Path:
    """Extract v2's backend at the tag and install its pinned deps. Idempotent."""
    root = cache_dir / "v2"
    backend = root / "backend"
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists() and (backend / "main.py").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "v2.tar"
    await _run("git", "-C", str(repo_root), "archive", "-o", str(archive), V2_TAG, "backend")
    await _run("tar", "-xf", str(archive), "-C", str(root))
    archive.unlink()
    await _run("uv", "venv", "--python", V2_PYTHON, str(root / ".venv"))
    await _run(
        "uv",
        "pip",
        "install",
        "--python",
        str(venv_python),
        "-r",
        str(backend / "requirements.txt"),
    )
    return root


class V2System:
    name = "v2"
    concurrency = 1  # sequential: per-question cost comes from the shared usage ledger

    def __init__(
        self,
        root: Path,
        openai_key: str,
        github_token: str,
        qdrant_url: str,
        port: int,
        proxy_port: int,
    ) -> None:
        self.root = root
        self.openai_key = openai_key
        self.github_token = github_token
        self.qdrant_url = qdrant_url
        self.base_url = f"http://127.0.0.1:{port}"
        self.port = port
        self.proxy_port = proxy_port
        self.ledger = UsageLedger()
        self.ingest_costs: dict[str, float] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._proxy: uvicorn.Server | None = None
        self._proxy_task: asyncio.Task[None] | None = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        self._sessions: set[str] = set()

    async def __aenter__(self) -> Self:
        self._proxy = uvicorn.Server(
            uvicorn.Config(
                create_proxy_app(self.ledger),
                host="127.0.0.1",
                port=self.proxy_port,
                log_level="warning",
            )
        )
        self._proxy_task = asyncio.create_task(self._proxy.serve())
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "OPENAI_API_KEY": self.openai_key,
            "OPENAI_BASE_URL": f"http://127.0.0.1:{self.proxy_port}/v1",
            "GITHUB_TOKEN": self.github_token,
            "QDRANT_URL": self.qdrant_url,
            "EVAL_ENABLED": "false",  # v2's own online judge would add cost and noise
            "LANGCHAIN_TRACING_V2": "false",
        }
        log = (self.root / "v2-server.log").open("ab")
        self._process = await asyncio.create_subprocess_exec(
            str(self.root / ".venv" / "bin" / "python"),
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            cwd=self.root / "backend",
            env=env,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        await self._wait_healthy()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=10)
        if self._proxy:
            self._proxy.should_exit = True
        if self._proxy_task:
            await self._proxy_task
        await self._http.aclose()

    async def _wait_healthy(self, timeout_s: float = 120) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process and self._process.returncode is not None:
                raise RuntimeError(f"v2 server exited early; see {self.root / 'v2-server.log'}")
            with contextlib.suppress(httpx.TransportError):
                if (await self._http.get(f"{self.base_url}/health")).status_code == 200:
                    return
            await asyncio.sleep(1)
        raise TimeoutError("v2 server did not become healthy")

    @staticmethod
    def session_id(repo: str) -> str:
        return "eval-" + repo.replace("/", "-").lower()

    async def ensure_ingested(self, repo: str) -> None:
        session = self.session_id(repo)
        if session in self._sessions:
            return
        existing = (await self._http.get(f"{self.base_url}/session/{session}")).json()
        if not existing.get("exists"):
            self.ledger.reset()
            response = await self._http.post(
                f"{self.base_url}/ingest",
                json={"repo_url": f"https://github.com/{repo}", "session_id": session},
            )
            response.raise_for_status()
            self.ingest_costs[repo] = self.ledger.totals()[2]
        self._sessions.add(session)

    async def _stream_lines(self, payload: dict[str, object]) -> AsyncIterator[str]:
        async with self._http.stream("POST", f"{self.base_url}/ask", json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                yield f"data: {json.dumps({'event': 'error', 'text': body[:500]})}"
                return
            async for line in response.aiter_lines():
                yield line

    async def answer(self, item: Item) -> SystemOutput:
        started = time.perf_counter()
        try:
            await self.ensure_ingested(item.repo)
            self.ledger.reset()
            payload: dict[str, object] = {
                "session_id": self.session_id(item.repo),
                "question": render_question(item),
                "chat_history": [],  # every eval question is independent
            }
            lines = [line async for line in self._stream_lines(payload)]
        except (httpx.HTTPError, RuntimeError) as exc:
            return SystemOutput(item_id=item.id, system=self.name, error=type(exc).__name__)
        answer, trace, error = parse_sse(lines)
        input_tokens, output_tokens, cost = self.ledger.totals()
        return SystemOutput(
            item_id=item.id,
            system=self.name,
            answer=answer,
            files=extract_paths(answer),
            tool_trace=trace,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            error=error,
        )

    def estimate_cost_usd(self, items: list[Item]) -> float:
        repos = {item.repo for item in items}
        return len(items) * _EST_COST_PER_QUESTION + len(repos) * _EST_COST_PER_INGEST
