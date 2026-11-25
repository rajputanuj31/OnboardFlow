from __future__ import annotations

from pathlib import Path

import pytest

from onboardflow.ingest.clone import CloneError, fetch_commit
from tests.gitrepo import commit, make_repo


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "origin"
    first = make_repo(repo, {"app.py": "v1\n", "pkg/mod.py": "x = 1\n"})
    second = commit(
        repo,
        {"app.py": "v2\n"},
        symlinks={"escape": "../../../../etc/passwd"},  # a hostile symlink
    )
    return repo, first, second


async def test_checks_out_exactly_the_requested_commit(
    origin: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, first, _ = origin
    dest = tmp_path / "out" / first

    await fetch_commit(f"file://{repo}", first, dest, timeout_s=30, allow_protocols="file")

    assert (dest / "app.py").read_text() == "v1\n"  # not the later commit
    assert (dest / "pkg" / "mod.py").exists()
    assert not (dest / ".git").exists()
    assert not list(dest.parent.glob(".*.partial-*"))  # temp dir cleaned up


async def test_symlinks_become_plain_files(origin: tuple[Path, str, str], tmp_path: Path) -> None:
    repo, _, second = origin
    dest = tmp_path / "out" / second

    await fetch_commit(f"file://{repo}", second, dest, timeout_s=30, allow_protocols="file")

    escape = dest / "escape"
    assert not escape.is_symlink()
    assert escape.read_text() == "../../../../etc/passwd"  # the target path, as text


async def test_fetch_is_idempotent(origin: tuple[Path, str, str], tmp_path: Path) -> None:
    repo, first, _ = origin
    dest = tmp_path / "out" / first
    await fetch_commit(f"file://{repo}", first, dest, timeout_s=30, allow_protocols="file")
    (dest / "marker").write_text("kept")
    await fetch_commit(f"file://{repo}", first, dest, timeout_s=30, allow_protocols="file")
    assert (dest / "marker").exists()  # existing checkout reused, not rebuilt


async def test_only_allowed_protocols(origin: tuple[Path, str, str], tmp_path: Path) -> None:
    repo, first, _ = origin
    with pytest.raises(CloneError):
        await fetch_commit(f"file://{repo}", first, tmp_path / "out" / first, timeout_s=30)
    assert not (tmp_path / "out" / first).exists()


async def test_unknown_commit_fails_cleanly(origin: tuple[Path, str, str], tmp_path: Path) -> None:
    repo, _, _ = origin
    missing = "0" * 40
    with pytest.raises(CloneError):
        await fetch_commit(
            f"file://{repo}",
            missing,
            tmp_path / "out" / missing,
            timeout_s=30,
            allow_protocols="file",
        )
    assert not any((tmp_path / "out").iterdir())
