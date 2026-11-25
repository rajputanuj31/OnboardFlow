"""Eval harness settings, read from the environment and the repo-root .env.

Every value that affects results (models, effort, budgets) lives here and is recorded
with each run, so a number in a report can always be traced to the config that made it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EVALS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EVALS_ROOT.parent


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    github_token: SecretStr | None = None

    # Systems under test
    answerer_provider: Literal["openai", "anthropic"] = Field(
        default="openai", alias="EVAL_ANSWERER_PROVIDER"
    )
    answerer_model: str = Field(default="gpt-6.1-terra", alias="EVAL_ANSWERER_MODEL")
    answerer_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="medium", alias="EVAL_ANSWERER_EFFORT"
    )
    answerer_max_tokens: int = Field(default=8000, alias="EVAL_ANSWERER_MAX_TOKENS")

    # Judge: a different, stronger model than the answerer (self-preference bias).
    judge_model: str = Field(default="gpt-6.2-sol", alias="EVAL_JUDGE_MODEL")

    # Spend guard: a run whose estimate exceeds this needs --yes.
    budget_usd: float = Field(default=20.0, gt=0, alias="EVAL_BUDGET_USD")
    concurrency: int = Field(default=8, ge=1, le=64, alias="EVAL_CONCURRENCY")

    # v2 baseline infrastructure
    v2_port: int = Field(default=8100, alias="EVAL_V2_PORT")
    v2_qdrant_url: str = Field(default="http://localhost:6343", alias="EVAL_V2_QDRANT_URL")
    usage_proxy_port: int = Field(default=8199, alias="EVAL_USAGE_PROXY_PORT")

    data_dir: Path = EVALS_ROOT / "data"
    runs_dir: Path = EVALS_ROOT / "runs"
    reports_dir: Path = EVALS_ROOT / "reports"
    cache_dir: Path = EVALS_ROOT / ".cache"

    @field_validator("anthropic_api_key", "openai_api_key", "github_token", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        return None if value == "" else value

    def system_config(self, system: str) -> dict[str, Any]:
        """Result-relevant config for one system under test (no secrets, no local paths)."""
        if system == "v2":
            return {
                "version": "v2.0.0",
                "model": "gpt-4o-mini",
                "embeddings": "text-embedding-3-small",
            }
        return {
            "provider": self.answerer_provider,
            "model": self.answerer_model,
            "effort": self.answerer_effort,
            "max_tokens": self.answerer_max_tokens,
        }


@lru_cache(maxsize=1)
def get_settings() -> EvalSettings:
    return EvalSettings()
