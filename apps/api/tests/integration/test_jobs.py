from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow import jobs
from onboardflow.db.models import Job, JobStatus

pytestmark = pytest.mark.integration


async def _enqueue(sessions: async_sessionmaker[AsyncSession], max_attempts: int = 3) -> None:
    async with sessions.begin() as session:
        session.add(jobs.new_job("test", {"n": 1}, max_attempts))


async def test_concurrent_workers_never_claim_the_same_job(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await _enqueue(sessions)
    claims = await asyncio.gather(*(jobs.claim(sessions, lease_s=30) for _ in range(8)))
    won = [c for c in claims if c is not None]
    assert len(won) == 1
    assert won[0].attempts == 1


async def test_expired_lease_is_reclaimed_by_another_worker(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await _enqueue(sessions)
    first = await jobs.claim(sessions, lease_s=0.2)  # this "worker" then dies silently
    assert first is not None
    assert await jobs.claim(sessions, lease_s=30) is None  # lease still held
    await asyncio.sleep(0.4)
    second = await jobs.claim(sessions, lease_s=30)
    assert second is not None
    assert second.id == first.id
    assert second.attempts == 2  # the lost run counts as an attempt


async def test_renewed_lease_is_not_stolen(sessions: async_sessionmaker[AsyncSession]) -> None:
    await _enqueue(sessions)
    job = await jobs.claim(sessions, lease_s=0.3)
    assert job is not None
    for _ in range(3):
        await asyncio.sleep(0.15)
        await jobs.renew_lease(sessions, job.id, lease_s=0.3)
    assert await jobs.claim(sessions, lease_s=30) is None


async def test_retry_with_backoff_then_final_failure(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await _enqueue(sessions, max_attempts=2)
    job = await jobs.claim(sessions, lease_s=30)
    assert job is not None
    assert await jobs.fail(sessions, job, "boom", retry_after=timedelta(seconds=60)) is True
    assert await jobs.claim(sessions, lease_s=30) is None  # backoff not elapsed

    async with sessions.begin() as session:  # fast-forward the backoff
        stored = (await session.execute(select(Job))).scalar_one()
        stored.run_after = stored.created_at
    job = await jobs.claim(sessions, lease_s=30)
    assert job is not None
    assert job.attempts == 2
    assert await jobs.fail(sessions, job, "boom again", retry_after=timedelta(0)) is False

    async with sessions() as session:
        stored = (await session.execute(select(Job))).scalar_one()
    assert stored.status == JobStatus.FAILED
    assert stored.last_error == "boom again"
