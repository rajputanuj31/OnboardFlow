"""End to end: POST /repos -> worker ingests -> snapshot ready, via the real API, queue and
database. GitHub is replaced by a fake that points at a local git repository."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow.db.models import Job, Repo, SnapshotFile
from onboardflow.github import RepoInfo, RepoNotFoundError, RepoRef
from onboardflow.ingest.pipeline import INGEST_JOB, IngestContext, mark_failed, run_ingest
from onboardflow.main import create_app
from onboardflow.settings import Settings
from onboardflow.worker import Registered, Worker
from tests.gitrepo import make_repo

pytestmark = pytest.mark.integration


@dataclass
class FakeGitHub:
    heads: dict[str, str] = field(default_factory=dict)  # "owner/name" -> sha
    size_kb: int = 10

    async def repo_info(self, ref: RepoRef) -> RepoInfo:
        sha = self.heads.get(ref.full_name.lower())
        if sha is None:
            raise RepoNotFoundError("Repository not found or not public.")
        return RepoInfo(ref=ref, default_branch="main", size_kb=self.size_kb, head_sha=sha)


@dataclass
class Env:
    app: FastAPI
    client: httpx.AsyncClient
    github: FakeGitHub
    worker: Worker
    sessions: async_sessionmaker[AsyncSession]
    origins: dict[str, Path]


@pytest.fixture
async def env(settings: Settings, tmp_path: Path) -> AsyncIterator[Env]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        github = FakeGitHub()
        app.state.github = github
        sessions = app.state.sessions
        async with sessions.begin() as session:
            await session.execute(
                text("TRUNCATE jobs, snapshot_files, blobs, snapshots, repos CASCADE")
            )
        origins: dict[str, Path] = {}
        ctx = IngestContext(
            sessions=sessions,
            settings=settings.ingest,
            remote_url=lambda repo: f"file://{origins[f'{repo.owner}/{repo.name}'.lower()]}",
            allow_protocols="file",
        )

        async def run(payload: dict[str, object]) -> None:
            await run_ingest(ctx, payload)

        async def on_failure(payload: dict[str, object], error: str, final: bool) -> None:
            await mark_failed(ctx, payload, error, final=final)

        worker = Worker(sessions, settings.worker, {INGEST_JOB: Registered(run, on_failure)})
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Env(app, client, github, worker, sessions, origins)


def _add_origin(env: Env, tmp_path: Path, full_name: str, files: dict[str, str]) -> str:
    path = tmp_path / "origins" / full_name.replace("/", "__")
    sha = make_repo(path, files)
    env.origins[full_name.lower()] = path
    env.github.heads[full_name.lower()] = sha
    return sha


async def _count(sessions: async_sessionmaker[AsyncSession], model: type) -> int:
    async with sessions() as session:
        return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_add_ingest_and_reuse(env: Env, tmp_path: Path) -> None:
    sha = _add_origin(
        env,
        tmp_path,
        "acme/widgets",
        {
            "src/widgets/core.py": "def spin():\n    return 1\n",
            "README.md": "# Widgets\n",
            "package-lock.json": "{}",
            "node_modules/x/index.js": "ignored",
        },
    )

    created = await env.client.post("/repos", json={"url": "https://github.com/acme/widgets"})
    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert body["commit_sha"] == sha

    # The same commit requested again (even concurrently) is one snapshot and one job.
    again = await asyncio.gather(
        *(
            env.client.post("/repos", json={"url": "https://github.com/ACME/widgets.git"})
            for _ in range(5)
        )
    )
    assert {r.json()["id"] for r in again} == {body["id"]}
    assert await _count(env.sessions, Job) == 1
    assert await _count(env.sessions, Repo) == 1

    assert await env.worker.run_once() is True
    snapshot = (await env.client.get(f"/snapshots/{body['id']}")).json()
    assert snapshot["status"] == "ready"
    assert snapshot["file_count"] == 3  # node_modules ignored
    assert snapshot["indexed_file_count"] == 2  # lockfile kept for grep, not indexed
    assert snapshot["progress_done"] == snapshot["progress_total"] == 3

    # Re-adding an unchanged commit is instant: 200, no new job, nothing re-cloned.
    reused = await env.client.post("/repos", json={"url": "https://github.com/acme/widgets"})
    assert reused.status_code == 200
    assert reused.json()["id"] == body["id"]
    assert await _count(env.sessions, Job) == 1


async def test_progress_stream_reports_stages_and_ends(env: Env, tmp_path: Path) -> None:
    _add_origin(env, tmp_path, "acme/stream", {"a.py": "x = 1\n"})
    snapshot_id = (
        await env.client.post("/repos", json={"url": "https://github.com/acme/stream"})
    ).json()["id"]

    async def consume() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        async with env.client.stream("GET", f"/snapshots/{snapshot_id}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    reader = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    await env.worker.run_once()
    events = await asyncio.wait_for(reader, timeout=10)

    statuses = [e["status"] for e in events]
    assert statuses[0] == "queued"
    assert statuses[-1] == "ready"  # stream ends on the terminal state


async def test_unknown_or_private_repo_is_a_404(env: Env) -> None:
    response = await env.client.post("/repos", json={"url": "https://github.com/acme/secret"})
    assert response.status_code == 404
    assert "not public" in response.json()["detail"]


async def test_invalid_url_is_a_400(env: Env) -> None:
    response = await env.client.post("/repos", json={"url": "file:///etc/passwd"})
    assert response.status_code == 400


async def test_oversized_repo_is_rejected_before_cloning(env: Env, tmp_path: Path) -> None:
    _add_origin(env, tmp_path, "acme/huge", {"a.py": "1\n"})
    env.github.size_kb = 10_000_000
    response = await env.client.post("/repos", json={"url": "https://github.com/acme/huge"})
    assert response.status_code == 413
    assert await _count(env.sessions, Job) == 0


async def test_failure_is_visible_and_retry_works(env: Env, tmp_path: Path) -> None:
    _add_origin(env, tmp_path, "acme/flaky", {"a.py": "1\n"})
    env.github.heads["acme/flaky"] = "f" * 40  # a commit that does not exist upstream
    snapshot_id = (
        await env.client.post("/repos", json={"url": "https://github.com/acme/flaky"})
    ).json()["id"]

    for _ in range(3):  # max_attempts; backoff is 0 in tests
        await env.worker.run_once()
    failed = (await env.client.get(f"/snapshots/{snapshot_id}")).json()
    assert failed["status"] == "failed"
    assert "git fetch failed" in failed["error"]

    # Fix the "upstream" and retry from the API.
    real_sha = make_repo(tmp_path / "fixed", {"a.py": "2\n"})
    env.origins["acme/flaky"] = tmp_path / "fixed"
    async with env.sessions.begin() as session:
        await session.execute(text("UPDATE snapshots SET commit_sha = :s"), {"s": real_sha})
    retried = await env.client.post(f"/snapshots/{snapshot_id}/retry")
    assert retried.status_code == 202
    assert retried.json()["status"] == "queued"
    assert (await env.client.post(f"/snapshots/{snapshot_id}/retry")).status_code == 409

    await env.worker.run_once()
    assert (await env.client.get(f"/snapshots/{snapshot_id}")).json()["status"] == "ready"


async def test_too_many_files_fails_permanently_without_retries(
    env: Env, tmp_path: Path, settings: Settings
) -> None:
    _add_origin(env, tmp_path, "acme/many", {f"f{i}.py": f"{i}\n" for i in range(12)})
    small = settings.ingest.model_copy(update={"max_files": 5})
    ctx = IngestContext(
        sessions=env.sessions,
        settings=small,
        remote_url=lambda repo: f"file://{env.origins['acme/many']}",
        allow_protocols="file",
    )

    async def run(payload: dict[str, object]) -> None:
        await run_ingest(ctx, payload)

    async def on_failure(payload: dict[str, object], error: str, final: bool) -> None:
        await mark_failed(ctx, payload, error, final=final)

    env.worker.handlers[INGEST_JOB] = Registered(run, on_failure)
    snapshot_id = (
        await env.client.post("/repos", json={"url": "https://github.com/acme/many"})
    ).json()["id"]

    await env.worker.run_once()  # one attempt is enough: retrying cannot help
    snapshot = (await env.client.get(f"/snapshots/{snapshot_id}")).json()
    assert snapshot["status"] == "failed"
    assert "too large" in snapshot["error"]
    assert await env.worker.run_once() is False  # nothing left to retry


async def test_files_are_recorded_by_content_hash(env: Env, tmp_path: Path) -> None:
    _add_origin(
        env, tmp_path, "acme/dupes", {"a.py": "same\n", "b.py": "same\n", "c.py": "other\n"}
    )
    await env.client.post("/repos", json={"url": "https://github.com/acme/dupes"})
    await env.worker.run_once()
    async with env.sessions() as session:
        rows = (await session.execute(select(SnapshotFile.path, SnapshotFile.blob_sha256))).all()
    hashes = dict(rows)
    assert hashes["a.py"] == hashes["b.py"] != hashes["c.py"]
