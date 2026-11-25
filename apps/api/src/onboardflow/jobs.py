"""A small, durable job queue in Postgres.

- `claim` atomically takes one runnable job with `FOR UPDATE SKIP LOCKED`, so concurrent
  workers never take the same job and never block each other.
- A claimed job holds a lease. The worker renews it while working; if the worker dies, the
  lease expires and the job becomes claimable again (counted as another attempt).
- Failures retry with exponential backoff until `max_attempts`, then fail for good.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow.db.models import Job, JobStatus


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


def new_job(kind: str, payload: dict[str, Any], max_attempts: int) -> Job:
    return Job(
        kind=kind, payload=payload, status=JobStatus.QUEUED, attempts=0, max_attempts=max_attempts
    )


_CLAIM_SQL = text(
    """
    UPDATE jobs
       SET status = 'running',
           attempts = attempts + 1,
           lease_until = now() + make_interval(secs => :lease_s),
           updated_at = now()
     WHERE id = (
           SELECT id FROM jobs
            WHERE (status = 'queued' AND run_after <= now())
               OR (status = 'running' AND lease_until < now())
            ORDER BY run_after, created_at
            LIMIT 1
              FOR UPDATE SKIP LOCKED
           )
    RETURNING id, kind, payload, attempts, max_attempts
    """
)


async def claim(sessions: async_sessionmaker[AsyncSession], lease_s: float) -> ClaimedJob | None:
    async with sessions.begin() as session:
        row = (await session.execute(_CLAIM_SQL, {"lease_s": lease_s})).first()
    if row is None:
        return None
    return ClaimedJob(
        id=row.id,
        kind=row.kind,
        payload=row.payload,
        attempts=row.attempts,
        max_attempts=row.max_attempts,
    )


async def renew_lease(
    sessions: async_sessionmaker[AsyncSession], job_id: uuid.UUID, lease_s: float
) -> None:
    async with sessions.begin() as session:
        await session.execute(
            text(
                "UPDATE jobs SET lease_until = now() + make_interval(secs => :lease_s), "
                "updated_at = now() WHERE id = :id AND status = 'running'"
            ),
            {"id": job_id, "lease_s": lease_s},
        )


async def complete(sessions: async_sessionmaker[AsyncSession], job_id: uuid.UUID) -> None:
    async with sessions.begin() as session:
        await session.execute(
            text(
                "UPDATE jobs SET status = 'done', lease_until = NULL, last_error = NULL, "
                "updated_at = now() WHERE id = :id"
            ),
            {"id": job_id},
        )


async def fail(
    sessions: async_sessionmaker[AsyncSession],
    job: ClaimedJob,
    error: str,
    *,
    retry_after: timedelta | None,
) -> bool:
    """Record a failure. Returns True if the job will be retried."""
    retry = retry_after is not None and job.attempts < job.max_attempts
    async with sessions.begin() as session:
        await session.execute(
            text(
                "UPDATE jobs SET status = :status, lease_until = NULL, last_error = :error, "
                "run_after = now() + make_interval(secs => :delay), updated_at = now() "
                "WHERE id = :id"
            ),
            {
                "id": job.id,
                "status": JobStatus.QUEUED.value if retry else JobStatus.FAILED.value,
                "error": error[:2000],
                "delay": retry_after.total_seconds() if retry and retry_after else 0,
            },
        )
    return retry
