"""The ingest job: clone one commit, scan it, record files, mark the snapshot ready.

Every stage writes its status and progress to the snapshot row, which is what the
`/snapshots/{id}/events` SSE stream reports. The job is safe to retry from any point:
the checkout is materialised atomically, blobs are inserted with ON CONFLICT DO NOTHING,
and a snapshot's file list is replaced as a whole.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow.db.models import Blob, Repo, Snapshot, SnapshotFile, SnapshotStatus
from onboardflow.ingest import PermanentIngestError
from onboardflow.ingest.clone import fetch_commit
from onboardflow.ingest.scan import ScannedFile, SnapshotTooLargeError, list_files, scan_batches
from onboardflow.settings import IngestSettings

INGEST_JOB = "ingest_snapshot"


def checkout_dir(data_dir: Path, repo_id: uuid.UUID, commit_sha: str) -> Path:
    return data_dir / str(repo_id) / commit_sha


@dataclass
class IngestContext:
    sessions: async_sessionmaker[AsyncSession]
    settings: IngestSettings
    # Where to fetch from. Production: the repo's GitHub https URL. Tests: a local repo.
    remote_url: Callable[[Repo], str] = lambda repo: (
        f"https://github.com/{repo.owner}/{repo.name}.git"
    )
    allow_protocols: str = "https"


async def _set(ctx: IngestContext, snapshot_id: uuid.UUID, **values: object) -> None:
    async with ctx.sessions.begin() as session:
        await session.execute(update(Snapshot).where(Snapshot.id == snapshot_id).values(**values))


async def run_ingest(ctx: IngestContext, payload: dict[str, object]) -> None:
    snapshot_id = uuid.UUID(str(payload["snapshot_id"]))
    async with ctx.sessions() as session:
        row = (
            await session.execute(
                select(Snapshot, Repo)
                .join(Repo, Repo.id == Snapshot.repo_id)
                .where(Snapshot.id == snapshot_id)
            )
        ).first()
    if row is None:
        raise PermanentIngestError(f"snapshot {snapshot_id} does not exist")
    snapshot, repo = row
    if snapshot.status == SnapshotStatus.READY:
        return

    await _set(
        ctx,
        snapshot_id,
        status=SnapshotStatus.CLONING,
        started_at=datetime.now(UTC),
        error=None,
        progress_done=0,
        progress_total=0,
    )
    dest = checkout_dir(ctx.settings.data_dir, repo.id, snapshot.commit_sha)
    await fetch_commit(
        ctx.remote_url(repo),
        snapshot.commit_sha,
        dest,
        timeout_s=ctx.settings.clone_timeout_s,
        allow_protocols=ctx.allow_protocols,
    )

    await _set(ctx, snapshot_id, status=SnapshotStatus.SCANNING)
    try:
        files = await asyncio.to_thread(list_files, dest, ctx.settings.max_files)
    except SnapshotTooLargeError as exc:
        raise PermanentIngestError(f"Repository too large: {exc}") from exc
    await _set(ctx, snapshot_id, progress_total=len(files))

    scanned: list[ScannedFile] = []
    max_total = ctx.settings.max_total_mb * 1024 * 1024
    batches = scan_batches(dest, files, ctx.settings.max_indexed_file_kb * 1024)
    while batch := await asyncio.to_thread(next, batches, None):
        scanned.extend(batch)
        if sum(f.size_bytes for f in scanned) > max_total:
            raise PermanentIngestError(
                f"Repository too large: more than {ctx.settings.max_total_mb} MB of files"
            )
        await _set(ctx, snapshot_id, progress_done=len(scanned))

    await _record_files(ctx, snapshot_id, scanned)
    await _set(
        ctx,
        snapshot_id,
        status=SnapshotStatus.READY,
        finished_at=datetime.now(UTC),
        file_count=len(scanned),
        indexed_file_count=sum(1 for f in scanned if f.indexed),
        total_bytes=sum(f.size_bytes for f in scanned),
        progress_done=len(scanned),
    )


async def _record_files(
    ctx: IngestContext, snapshot_id: uuid.UUID, scanned: list[ScannedFile]
) -> None:
    blobs: dict[str, dict[str, object]] = {
        f.sha256: {
            "sha256": f.sha256,
            "language": f.language,
            "size_bytes": f.size_bytes,
            "line_count": f.line_count,
        }
        for f in scanned
    }
    rows: list[dict[str, object]] = [
        {
            "snapshot_id": snapshot_id,
            "path": f.path,
            "blob_sha256": f.sha256,
            "indexed": f.indexed,
            "skip_reason": f.skip_reason,
        }
        for f in scanned
    ]
    async with ctx.sessions.begin() as session:
        for chunk in _chunks(list(blobs.values()), 1000):
            await session.execute(insert(Blob).values(chunk).on_conflict_do_nothing())
        await session.execute(delete(SnapshotFile).where(SnapshotFile.snapshot_id == snapshot_id))
        for chunk in _chunks(rows, 1000):
            await session.execute(insert(SnapshotFile).values(chunk))


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def mark_failed(
    ctx: IngestContext, payload: dict[str, object], error: str, *, final: bool
) -> None:
    """Called by the worker on failure: the user sees the error, and whether it will retry."""
    snapshot_id = uuid.UUID(str(payload["snapshot_id"]))
    if final:
        await _set(
            ctx,
            snapshot_id,
            status=SnapshotStatus.FAILED,
            error=error,
            finished_at=datetime.now(UTC),
        )
    else:
        await _set(ctx, snapshot_id, status=SnapshotStatus.QUEUED, error=f"{error} (retrying)")
