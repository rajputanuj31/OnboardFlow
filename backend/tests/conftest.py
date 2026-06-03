"""Shared pytest fixtures.

Puts the backend root on sys.path so tests can `from utils.vector_db import ...`
without depending on the developer's working directory.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Default to embedded Qdrant + a dummy GitHub token so the import chain doesn't
# explode on settings construction during test collection.
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("GITHUB_TOKEN", "dummy")
os.environ.setdefault("OPENAI_API_KEY", "dummy-test-key")
os.environ.setdefault("EVAL_ENABLED", "false")


@pytest.fixture
def fresh_qdrant_path(tmp_path, monkeypatch):
    """Use a temp-dir embedded Qdrant for isolation between tests."""
    qdir = tmp_path / "qdrant"
    monkeypatch.setenv("QDRANT_URL", str(qdir))

    # Reset cached client / settings so the new env var takes effect.
    import utils.vector_db as vdb
    from config import Settings
    vdb._client = None
    monkeypatch.setattr("utils.vector_db.settings", Settings(_env_file=None))

    yield qdir

    vdb._client = None
    if qdir.exists():
        shutil.rmtree(qdir, ignore_errors=True)
