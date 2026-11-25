"""Ingest a real public repository from GitHub (needs network; no database).

Checks the plan's acceptance criterion: every tracked file outside ignored directories is
recorded. The reference list comes from an independent plain `git` fetch + `git ls-files`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from onboardflow.ingest.clone import fetch_commit
from onboardflow.ingest.scan import IGNORED_DIRS, list_files, scan_batches
from tests.gitrepo import git

pytestmark = pytest.mark.network

REQUESTS_URL = "https://github.com/psf/requests.git"
REQUESTS_V2_32_3 = "0e322af87745eff34caffe4df68456ebc20d9068"


async def test_requests_release_is_fully_recorded(tmp_path: Path) -> None:
    started = time.perf_counter()
    dest = tmp_path / "snapshot"
    await fetch_commit(REQUESTS_URL, REQUESTS_V2_32_3, dest, timeout_s=120)
    files = list_files(dest, max_files=20_000)
    scanned = [f for batch in scan_batches(dest, files, 1024 * 1024) for f in batch]
    elapsed = time.perf_counter() - started

    reference = tmp_path / "reference"
    reference.mkdir()
    git(reference, "init", "-q")
    git(reference, "fetch", "-q", "--depth", "1", REQUESTS_URL, REQUESTS_V2_32_3)
    tracked = git(reference, "ls-tree", "-r", "--name-only", "FETCH_HEAD").splitlines()
    expected = {p for p in tracked if not IGNORED_DIRS.intersection(p.split("/")[:-1])}

    assert {f.path for f in scanned} == expected
    assert any(f.path == "src/requests/sessions.py" and f.indexed for f in scanned)
    print(f"\nrequests v2.32.3: {len(scanned)} files in {elapsed:.1f}s")
