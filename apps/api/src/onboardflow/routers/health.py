"""Health endpoints.

- ``GET /health``: liveness. 200 whenever the process can serve requests.
- ``GET /health/ready``: readiness. Checks Postgres and Qdrant concurrently;
  200 if both are reachable, 503 otherwise, with per-dependency detail.

Errors are reported by exception type only: driver messages can contain hostnames
or connection strings, and this endpoint is public.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

import httpx
from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from onboardflow.settings import Settings

router = APIRouter(tags=["health"])


class CheckResult(BaseModel):
    ok: bool
    latency_ms: float
    error: str | None = None


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, CheckResult]


async def check_postgres(engine: AsyncEngine) -> CheckResult:
    started = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        return CheckResult(ok=False, latency_ms=_elapsed_ms(started), error=type(exc).__name__)
    return CheckResult(ok=True, latency_ms=_elapsed_ms(started))


async def check_qdrant(http: httpx.AsyncClient, settings: Settings) -> CheckResult:
    started = time.perf_counter()
    headers = {}
    if settings.qdrant.api_key is not None:
        headers["api-key"] = settings.qdrant.api_key.get_secret_value()
    url = f"{str(settings.qdrant.url).rstrip('/')}/readyz"
    try:
        response = await http.get(url, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        return CheckResult(ok=False, latency_ms=_elapsed_ms(started), error=type(exc).__name__)
    return CheckResult(ok=True, latency_ms=_elapsed_ms(started))


@router.get("/health")
async def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/health/ready",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    state = request.app.state
    postgres, qdrant = await asyncio.gather(
        check_postgres(state.db_engine),
        check_qdrant(state.http, state.settings),
    )
    checks = {"postgres": postgres, "qdrant": qdrant}
    healthy = all(check.ok for check in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ok" if healthy else "degraded", checks=checks)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
