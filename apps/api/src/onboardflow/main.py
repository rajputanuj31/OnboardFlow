"""FastAPI application factory.

Run locally with:  uv run uvicorn onboardflow.main:create_app --factory --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine

from onboardflow import __version__
from onboardflow.routers import health
from onboardflow.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Shared, long-lived clients. Nothing connects until first use, so the app
        # starts even if Postgres or Qdrant are down; /health/ready reports that.
        engine = create_async_engine(
            settings.database.url.get_secret_value(),
            pool_size=settings.database.pool_size,
            pool_pre_ping=True,
            connect_args={"timeout": settings.database.connect_timeout_s},
        )
        http = httpx.AsyncClient(timeout=settings.qdrant.timeout_s)
        app.state.settings = settings
        app.state.db_engine = engine
        app.state.http = http
        try:
            yield
        finally:
            await http.aclose()
            await engine.dispose()

    app = FastAPI(title="OnboardFlow", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    return app
