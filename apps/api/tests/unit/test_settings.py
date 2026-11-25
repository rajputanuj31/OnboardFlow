from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from onboardflow.settings import REPO_ROOT, Environment, Settings, render_env_example
from tests.unit.constants import TEST_DB_URL


def test_missing_database_url_fails_loudly() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "database" in str(exc_info.value)


def test_reads_nested_values_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE__URL", TEST_DB_URL)
    monkeypatch.setenv("DATABASE__POOL_SIZE", "7")
    monkeypatch.setenv("QDRANT__URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("CORS_ORIGINS", '["https://a.example", "https://b.example"]')

    settings = Settings(_env_file=None)

    assert settings.database.url.get_secret_value() == TEST_DB_URL
    assert settings.database.pool_size == 7
    assert str(settings.qdrant.url) == "http://qdrant.internal:6333/"
    assert settings.environment is Environment.STAGING
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_rejects_sync_postgres_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE__URL", "postgresql://u:p@localhost:5432/db")
    with pytest.raises(ValidationError, match="asyncpg"):
        Settings(_env_file=None)


@pytest.mark.parametrize("value", [":memory:", "./.cache/qdrant_data", "/var/qdrant"])
def test_rejects_embedded_qdrant(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """v2's embedded mode caused a cross-process lock clash; v3 is server mode only."""
    monkeypatch.setenv("DATABASE__URL", TEST_DB_URL)
    monkeypatch.setenv("QDRANT__URL", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_public_config_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE__URL", TEST_DB_URL)
    monkeypatch.setenv("QDRANT__API_KEY", "qdrant-secret-key")
    settings = Settings(_env_file=None)

    dumped = json.dumps(settings.public_config())

    assert "s3cret-pw" not in dumped
    assert "qdrant-secret-key" not in dumped
    assert "s3cret-pw" not in repr(settings)


def test_env_example_matches_settings_schema() -> None:
    """The committed .env.example is generated; regenerate with `make env-example`."""
    committed = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert committed == render_env_example()


def test_env_example_lists_every_required_key() -> None:
    rendered = render_env_example()
    assert "DATABASE__URL=\n" in rendered
    assert "QDRANT__URL=http://localhost:6333/" in rendered
