"""Snapshot status, live progress (SSE) and retry.

The SSE stream polls the snapshot row (cheap: primary-key lookup) and emits an event only
when something changed, plus a heartbeat comment so proxies keep the connection open. It
ends on `ready` or `failed`. Clients that reconnect simply get the current state again.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow.db.models import Repo, Snapshot, SnapshotStatus
from onboardflow.ingest.pipeline import INGEST_JOB
from onboardflow.jobs import new_job
from onboardflow.settings import Settings

router = APIRouter(tags=["snapshots"])

TERMINAL = {SnapshotStatus.READY, SnapshotStatus.FAILED}
POLL_S = 0.5
HEARTBEAT_S = 15.0


class SnapshotOut(BaseModel):
    id: uuid.UUID
    repo: str
    commit_sha: str
    status: SnapshotStatus
    progress_done: int
    progress_total: int
    file_count: int | None
    indexed_file_count: int | None
    total_bytes: int | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


def snapshot_out(snapshot: Snapshot, owner: str, name: str) -> SnapshotOut:
    return SnapshotOut(
        id=snapshot.id,
        repo=f"{owner}/{name}",
        commit_sha=snapshot.commit_sha,
        status=snapshot.status,
        progress_done=snapshot.progress_done,
        progress_total=snapshot.progress_total,
        file_count=snapshot.file_count,
        indexed_file_count=snapshot.indexed_file_count,
        total_bytes=snapshot.total_bytes,
        error=snapshot.error,
        created_at=snapshot.created_at,
        finished_at=snapshot.finished_at,
    )


async def load_snapshot(
    sessions: async_sessionmaker[AsyncSession], snapshot_id: uuid.UUID
) -> SnapshotOut | None:
    async with sessions() as session:
        row = (
            await session.execute(
                select(Snapshot, Repo.owner, Repo.name)
                .join(Repo, Repo.id == Snapshot.repo_id)
                .where(Snapshot.id == snapshot_id)
            )
        ).first()
    return None if row is None else snapshot_out(row[0], row[1], row[2])


@router.get("/snapshots/{snapshot_id}", responses={404: {}})
async def get_snapshot(snapshot_id: uuid.UUID, request: Request) -> SnapshotOut:
    snapshot = await load_snapshot(request.app.state.sessions, snapshot_id)
    if snapshot is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Snapshot not found.")
    return snapshot


@router.get("/snapshots/{snapshot_id}/events", responses={404: {}})
async def snapshot_events(snapshot_id: uuid.UUID, request: Request) -> StreamingResponse:
    sessions: async_sessionmaker[AsyncSession] = request.app.state.sessions
    if await load_snapshot(sessions, snapshot_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Snapshot not found.")

    async def stream() -> AsyncIterator[str]:
        last: str | None = None
        idle = 0.0
        while not await request.is_disconnected():
            snapshot = await load_snapshot(sessions, snapshot_id)
            if snapshot is None:
                yield _event("error", {"message": "Snapshot was deleted."})
                return
            payload = snapshot.model_dump_json()
            if payload != last:
                yield f"event: snapshot\ndata: {payload}\n\n"
                last, idle = payload, 0.0
            if snapshot.status in TERMINAL:
                return
            await asyncio.sleep(POLL_S)
            idle += POLL_S
            if idle >= HEARTBEAT_S:
                yield ": heartbeat\n\n"
                idle = 0.0

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/snapshots/{snapshot_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {}, 409: {}},
)
async def retry_snapshot(snapshot_id: uuid.UUID, request: Request) -> SnapshotOut:
    settings: Settings = request.app.state.settings
    async with request.app.state.sessions.begin() as session:
        result = await session.execute(
            update(Snapshot)
            .where(Snapshot.id == snapshot_id, Snapshot.status == SnapshotStatus.FAILED)
            .values(
                status=SnapshotStatus.QUEUED,
                error=None,
                progress_done=0,
                progress_total=0,
                finished_at=None,
            )
            .returning(Snapshot.id)
        )
        if result.scalar_one_or_none() is None:
            exists = await session.get(Snapshot, snapshot_id)
            if exists is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Snapshot not found.")
            raise HTTPException(status.HTTP_409_CONFLICT, "Only failed snapshots can be retried.")
        session.add(
            new_job(INGEST_JOB, {"snapshot_id": str(snapshot_id)}, settings.worker.max_attempts)
        )
    snapshot = await load_snapshot(request.app.state.sessions, snapshot_id)
    if snapshot is None:  # deleted between the update and this read
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Snapshot not found.")
    return snapshot


def _event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"
