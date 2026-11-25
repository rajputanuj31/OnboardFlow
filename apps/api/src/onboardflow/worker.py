"""Background worker: claims jobs from Postgres and runs them.

Run with `python -m onboardflow.worker` (or `make worker`). Several workers can run at
once; the queue guarantees each job runs on one worker at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow import jobs
from onboardflow.db.session import make_engine, make_sessionmaker
from onboardflow.ingest import PermanentIngestError
from onboardflow.ingest.clone import CloneError
from onboardflow.ingest.pipeline import INGEST_JOB, IngestContext, mark_failed, run_ingest
from onboardflow.settings import Settings, WorkerSettings, get_settings

log = logging.getLogger("onboardflow.worker")

Handler = Callable[[dict[str, object]], Awaitable[None]]
FailureHook = Callable[[dict[str, object], str, bool], Awaitable[None]]


@dataclass
class Registered:
    run: Handler
    on_failure: FailureHook


def public_error(exc: BaseException) -> str:
    """A message safe to show users: our own error types say what happened; anything else
    is reported by type only (driver/library messages can contain paths or URLs)."""
    if isinstance(exc, PermanentIngestError | CloneError):
        return str(exc)
    return f"Internal error ({type(exc).__name__})"


class Worker:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        settings: WorkerSettings,
        handlers: dict[str, Registered],
    ) -> None:
        self.sessions = sessions
        self.settings = settings
        self.handlers = handlers

    async def run_once(self) -> bool:
        """Claim and run at most one job. Returns False when there was nothing to do."""
        job = await jobs.claim(self.sessions, self.settings.lease_s)
        if job is None:
            return False
        handler = self.handlers.get(job.kind)
        if handler is None:
            await jobs.fail(
                self.sessions, job, f"no handler for job kind {job.kind!r}", retry_after=None
            )
            return True

        heartbeat = asyncio.create_task(self._heartbeat(job))
        try:
            await handler.run(job.payload)
        except PermanentIngestError as exc:
            await jobs.fail(self.sessions, job, repr(exc), retry_after=None)
            await handler.on_failure(job.payload, public_error(exc), True)
            log.warning("job %s failed permanently: %s", job.id, exc)
        except Exception as exc:
            delay = timedelta(seconds=self.settings.retry_backoff_s * 2 ** (job.attempts - 1))
            will_retry = await jobs.fail(self.sessions, job, repr(exc), retry_after=delay)
            await handler.on_failure(job.payload, public_error(exc), not will_retry)
            log.warning(
                "job %s attempt %d failed (%s)",
                job.id,
                job.attempts,
                "retrying" if will_retry else "giving up",
                exc_info=exc,
            )
        else:
            await jobs.complete(self.sessions, job.id)
            log.info("job %s (%s) done", job.id, job.kind)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        return True

    async def _heartbeat(self, job: jobs.ClaimedJob) -> None:
        while True:
            await asyncio.sleep(self.settings.lease_s / 3)
            await jobs.renew_lease(self.sessions, job.id, self.settings.lease_s)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            worked = await self.run_once()
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.settings.poll_interval_s)


def build_worker(settings: Settings, sessions: async_sessionmaker[AsyncSession]) -> Worker:
    ingest = IngestContext(sessions=sessions, settings=settings.ingest)

    async def run(payload: dict[str, object]) -> None:
        await run_ingest(ingest, payload)

    async def on_failure(payload: dict[str, object], error: str, final: bool) -> None:
        await mark_failed(ingest, payload, error, final=final)

    return Worker(sessions, settings.worker, {INGEST_JOB: Registered(run, on_failure)})


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    engine = make_engine(settings)
    worker = build_worker(settings, make_sessionmaker(engine))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    log.info("worker started")
    try:
        await worker.run_forever(stop)
    finally:
        await engine.dispose()
        log.info("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
