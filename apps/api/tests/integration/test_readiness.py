"""Readiness against real services. Run with `make up && make test-integration`."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from onboardflow.main import create_app
from onboardflow.settings import get_settings

pytestmark = pytest.mark.integration


def test_ready_against_compose_services() -> None:
    settings = get_settings()
    with TestClient(create_app(settings)) as client:
        # Services may still be starting (CI, fresh `make up`): poll briefly.
        deadline = time.monotonic() + 30
        while True:
            response = client.get("/health/ready")
            if response.status_code == 200 or time.monotonic() > deadline:
                break
            time.sleep(1)

    assert response.status_code == 200, response.json()
    checks = response.json()["checks"]
    assert checks["postgres"]["ok"] is True
    assert checks["qdrant"]["ok"] is True
