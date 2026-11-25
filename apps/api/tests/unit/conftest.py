"""Unit-test fixtures. Unit tests never touch real services or the developer's .env."""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from onboardflow.settings import DatabaseSettings, Settings
from tests.unit.constants import TEST_DB_URL

_SETTINGS_ENV_PREFIXES = ("DATABASE__", "QDRANT__", "ENVIRONMENT", "LOG_LEVEL", "CORS_ORIGINS")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any settings variables inherited from the shell so tests are hermetic."""
    for key in list(os.environ):
        if key.upper().startswith(_SETTINGS_ENV_PREFIXES):
            monkeypatch.delenv(key)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database=DatabaseSettings(url=SecretStr(TEST_DB_URL)),
    )
