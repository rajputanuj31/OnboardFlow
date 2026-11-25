"""Build small local git repositories for tests (no network)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv in tests
        ["git", *args],  # noqa: S607
        cwd=repo,
        env=_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_repo(path: Path, files: dict[str, str]) -> str:
    """Create a repo with one commit containing `files`; return the commit sha."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    return commit(path, files)


def commit(path: Path, files: dict[str, str], symlinks: dict[str, str] | None = None) -> str:
    for rel, content in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for rel, points_to in (symlinks or {}).items():
        (path / rel).symlink_to(points_to)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "test commit")
    return git(path, "rev-parse", "HEAD")
