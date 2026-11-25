"""Fetch exactly one commit of a repository, as untrusted data.

Hardening (design plan 6.2):
- Fetch by commit sha (`git fetch --depth 1 <url> <sha>`), so the checkout is exactly the
  snapshot's commit even if the branch moves during ingestion.
- Isolated git config: no system/global config (no `url.insteadOf` rewrites, no
  credential helpers), no terminal prompts, hooks path set to /dev/null, only the allowed
  transport protocols (https in production), no submodules, LFS content not downloaded.
- `transfer.fsckObjects` rejects malformed objects (e.g. crafted path names).
- `core.symlinks=false`: symlinks are checked out as small text files containing the
  target path, so nothing in the checkout can point outside it.
- Nothing from the repository is ever executed.
- The checkout is built in a temporary directory and renamed into place only when complete,
  so a crash never leaves a half-written snapshot that looks finished. `.git` is removed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path

from onboardflow.ingest import PermanentIngestError


class CloneError(RuntimeError):
    pass


def _git_env(allow_protocols: str) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_ALLOW_PROTOCOL": allow_protocols,
    }


_SAFE_CONFIG = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.symlinks=false",
    "-c", "transfer.fsckObjects=true",
    "-c", "fetch.recurseSubmodules=false",
)  # fmt: skip


async def _git(*args: str, cwd: Path, env: dict[str, str], timeout_s: float) -> str:
    process = await asyncio.create_subprocess_exec(
        "git",
        *_SAFE_CONFIG,
        *args,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise CloneError(f"git {args[0]} timed out after {timeout_s:.0f}s") from None
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip().splitlines()
        raise CloneError(f"git {args[0]} failed: {message[-1] if message else 'unknown error'}")
    return stdout.decode(errors="replace").strip()


async def fetch_commit(
    remote_url: str,
    commit_sha: str,
    dest: Path,
    *,
    timeout_s: float,
    allow_protocols: str = "https",
) -> None:
    """Materialise `commit_sha` of `remote_url` at `dest` (no .git). Idempotent."""
    if await asyncio.to_thread(dest.exists):
        return
    await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
    work = dest.with_name(f".{dest.name}.partial-{uuid.uuid4().hex[:8]}")
    env = _git_env(allow_protocols)
    try:
        await _git("init", "-q", str(work), cwd=dest.parent, env=env, timeout_s=30)
        await _git(
            "fetch", "--depth", "1", "--no-tags", "--no-recurse-submodules", "-q",
            remote_url, commit_sha,
            cwd=work, env=env, timeout_s=timeout_s,
        )  # fmt: skip
        await _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=work, env=env, timeout_s=60)
        head = await _git("rev-parse", "HEAD", cwd=work, env=env, timeout_s=10)
        if head != commit_sha:
            raise PermanentIngestError(f"checked out {head[:12]}, expected {commit_sha[:12]}")
        await asyncio.to_thread(shutil.rmtree, work / ".git")
        await asyncio.to_thread(work.rename, dest)
    finally:
        if await asyncio.to_thread(work.exists):
            await asyncio.to_thread(shutil.rmtree, work, True)
