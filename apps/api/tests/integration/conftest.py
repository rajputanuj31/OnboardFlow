"""Integration fixtures: a throwaway Postgres database per test session, built with the
real Alembic migrations (so the migrations themselves are under test), dropped afterwards."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncpg
import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from onboardflow.db.session import make_engine, make_sessionmaker
from onboardflow.settings import DatabaseSettings, Settings, get_settings

API_ROOT = Path(__file__).resolve().parents[2]


def _admin_dsn(url: str) -> str:
    # postgresql+asyncpg://u:p@h:5432/db -> postgresql://u:p@h:5432/postgres
    base, _, _ = url.replace("postgresql+asyncpg://", "postgresql://").rpartition("/")
    return f"{base}/postgres"


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    url = get_settings().database.url.get_secret_value()
    name = f"onboardflow_test_{uuid.uuid4().hex[:8]}"
    admin = _admin_dsn(url)

    async def admin_exec(sql: str) -> None:
        conn = await asyncpg.connect(admin)
        try:
            await conn.execute(sql)
        finally:
            await conn.close()

    asyncio.run(admin_exec(f'CREATE DATABASE "{name}"'))
    test_url = f"{url.rpartition('/')[0]}/{name}"
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=API_ROOT,
        env={**os.environ, "DATABASE__URL": test_url},
        check=True,
        capture_output=True,
    )
    yield test_url
    asyncio.run(admin_exec(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture
def settings(database_url: str, tmp_path: Path) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "database": DatabaseSettings(url=SecretStr(database_url)),
            "ingest": base.ingest.model_copy(update={"data_dir": tmp_path / "repos"}),
            "worker": base.worker.model_copy(update={"retry_backoff_s": 0.0, "lease_s": 30.0}),
        }
    )


@pytest.fixture
async def sessions(settings: Settings) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = make_engine(settings)
    factory = make_sessionmaker(engine)
    async with factory.begin() as session:  # each test starts from empty tables
        await session.execute(
            text("TRUNCATE jobs, snapshot_files, blobs, snapshots, repos RESTART IDENTITY CASCADE")
        )
    yield factory
    await engine.dispose()
