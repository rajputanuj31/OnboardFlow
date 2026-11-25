"""Typed application settings: the single source of configuration for OnboardFlow.

Every tunable value (URLs, model names, budgets, chunk sizes, k values, limits) lives
here, grouped by area, and is read from environment variables. Nested groups use a
double-underscore delimiter, e.g. ``DATABASE__URL`` or ``QDRANT__TIMEOUT_S``.

Rules:
- Add new knobs here, never as module-level constants or ad-hoc ``os.getenv`` calls.
- Secrets are ``SecretStr`` so they never appear in logs, reprs or ``public_config()``.
- ``public_config()`` is what eval runs and traces record as "the config used".
- The committed ``.env.example`` is generated from this module
  (``python -m onboardflow.settings > ../../.env.example``) and a test fails if it drifts.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/src/onboardflow/settings.py -> repo root is four levels up from this package.
REPO_ROOT = Path(__file__).resolve().parents[4]
ENV_FILE = REPO_ROOT / ".env"


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseModel):
    url: SecretStr = Field(
        description=(
            "Async SQLAlchemy URL for Postgres, "
            "e.g. postgresql+asyncpg://user:password@localhost:5432/onboardflow"
        ),
    )
    pool_size: int = Field(default=5, ge=1, le=50, description="Connection pool size.")
    connect_timeout_s: float = Field(
        default=3.0, gt=0, description="Timeout for opening a connection, in seconds."
    )

    @field_validator("url")
    @classmethod
    def _require_asyncpg(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE__URL must use the async driver: postgresql+asyncpg://...")
        return value


class QdrantSettings(BaseModel):
    # AnyHttpUrl on purpose: Qdrant runs in server mode only. v2's embedded mode (a local
    # path or ":memory:") caused a file-lock clash between processes, so it is rejected.
    url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://localhost:6333"),
        description="Qdrant server URL (server mode only; local paths are rejected).",
    )
    api_key: SecretStr | None = Field(default=None, description="Qdrant API key, if enabled.")
    timeout_s: float = Field(default=3.0, gt=0, description="Request timeout in seconds.")

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        # `QDRANT__API_KEY=` copied from .env.example must mean "no key", not an empty key.
        return None if value == "" else value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Field(
        default=Environment.LOCAL, description="local | test | staging | production"
    )
    log_level: str = Field(default="INFO", description="Python logging level.")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description='Allowed browser origins, as a JSON list: ["https://app.example.com"]',
    )
    database: DatabaseSettings
    qdrant: QdrantSettings = QdrantSettings()

    def public_config(self) -> dict[str, Any]:
        """Config safe to log or store with an eval run: secrets are masked."""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process. Raises a ValidationError naming any missing value."""
    return Settings()  # required values come from the environment


def render_env_example() -> str:
    """Render ``.env.example`` from the Settings schema (keeps docs and code in sync)."""
    lines = [
        "# Generated from apps/api/src/onboardflow/settings.py. Do not edit by hand:",
        "#   make env-example",
        "",
    ]

    def emit(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            annotation = field.annotation
            key = f"{prefix}{name}".upper()
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                lines.append(f"# --- {name} ---")
                emit(annotation, f"{prefix}{name}__")
                continue
            if field.description:
                lines.append(f"# {field.description}")
            if field.is_required():
                lines.append("# (required)")
                lines.append(f"{key}=")
            else:
                default = field.get_default(call_default_factory=True)
                lines.append(f"{key}={_format_default(default)}")
            lines.append("")

    emit(Settings, "")
    return "\n".join(lines).rstrip() + "\n"


def _format_default(value: object) -> str:
    if value is None or isinstance(value, SecretStr):
        return ""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        import json

        return json.dumps(value)
    return str(value)


if __name__ == "__main__":
    print(render_env_example(), end="")
