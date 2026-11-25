from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import create_async_engine

from onboardflow.main import create_app
from onboardflow.routers import health
from onboardflow.routers.health import CheckResult
from onboardflow.settings import QdrantSettings, Settings
from tests.unit.constants import TEST_DB_URL


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _fixed_check(result: CheckResult) -> Callable[..., Awaitable[CheckResult]]:
    async def check(*_args: object) -> CheckResult:
        return result

    return check


def test_liveness(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_all_dependencies_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "check_postgres", _fixed_check(CheckResult(ok=True, latency_ms=1)))
    monkeypatch.setattr(health, "check_qdrant", _fixed_check(CheckResult(ok=True, latency_ms=1)))

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["checks"]) == {"postgres", "qdrant"}


def test_not_ready_when_a_dependency_is_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "check_postgres", _fixed_check(CheckResult(ok=True, latency_ms=1)))
    monkeypatch.setattr(
        health,
        "check_qdrant",
        _fixed_check(CheckResult(ok=False, latency_ms=1, error="ConnectError")),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"] == {"ok": False, "latency_ms": 1.0, "error": "ConnectError"}


async def test_postgres_check_reports_error_type_without_secrets() -> None:
    # Port 1 on loopback refuses connections immediately: no real database needed.
    engine = create_async_engine(TEST_DB_URL, connect_args={"timeout": 1})
    try:
        result = await health.check_postgres(engine)
    finally:
        await engine.dispose()

    assert result.ok is False
    assert result.error is not None
    assert "s3cret-pw" not in result.error
    assert "127.0.0.1" not in result.error


async def test_qdrant_check_sends_api_key_and_hits_readyz(settings: Settings) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="all shards are ready")

    keyed = settings.model_copy(update={"qdrant": QdrantSettings(api_key=SecretStr("qk-123"))})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await health.check_qdrant(http, keyed)

    assert result.ok is True
    assert seen[0].url.path == "/readyz"
    assert seen[0].headers["api-key"] == "qk-123"


async def test_qdrant_check_fails_on_http_error(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await health.check_qdrant(http, settings)

    assert result.ok is False
    assert result.error == "HTTPStatusError"
