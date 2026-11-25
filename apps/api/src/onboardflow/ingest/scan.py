"""Walk a checkout and describe every file: content hash, size, language, indexability.

Ignored directories (dependencies, build output, VCS metadata) are not recorded at all.
Everything else is recorded so `glob`/`grep` see the real tree; files that should not be
embedded or parsed (binary, lockfiles, minified, generated maps, Git LFS pointers, very
large files) are recorded with `indexed=False` and a reason.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "bower_components", "vendor", "dist", "build",
        ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".tox", ".nox", ".next", ".nuxt", "target", "coverage", ".idea", ".vscode",
    }
)  # fmt: skip
LOCKFILES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
        "Pipfile.lock", "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum", "bun.lockb",
    }
)  # fmt: skip
LANGUAGES = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".rb": "ruby", ".php": "php",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".swift": "swift", ".scala": "scala", ".sh": "shell", ".md": "markdown", ".rst": "rst",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".html": "html",
    ".css": "css", ".sql": "sql",
}  # fmt: skip
_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
_SNIFF_BYTES = 8192


class SnapshotTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class ScannedFile:
    path: str  # POSIX, relative to the checkout root
    sha256: str
    size_bytes: int
    line_count: int | None
    language: str | None
    indexed: bool
    skip_reason: str | None


def list_files(root: Path, max_files: int) -> list[Path]:
    """Relative paths of recordable files, sorted. Never follows symlinks."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            found.append((Path(dirpath) / name).relative_to(root))
            if len(found) > max_files:
                raise SnapshotTooLargeError(f"more than {max_files} files")
    return found


def describe(root: Path, rel: Path, max_indexed_bytes: int) -> ScannedFile:
    full = root / rel
    if full.is_symlink():  # defensive: checkouts use core.symlinks=false
        return ScannedFile(rel.as_posix(), hashlib.sha256(b"").hexdigest(), 0, None, None,
                           False, "symlink")  # fmt: skip
    data = full.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    language = LANGUAGES.get(full.suffix.lower())
    reason = _skip_reason(rel, data, max_indexed_bytes)
    lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    return ScannedFile(
        path=rel.as_posix(),
        sha256=digest,
        size_bytes=len(data),
        line_count=None if reason == "binary" else lines,
        language=language,
        indexed=reason is None,
        skip_reason=reason,
    )


def _skip_reason(rel: Path, data: bytes, max_indexed_bytes: int) -> str | None:
    name = rel.name
    if b"\x00" in data[:_SNIFF_BYTES]:
        return "binary"
    if data.startswith(_LFS_POINTER):
        return "lfs-pointer"
    if name in LOCKFILES:
        return "lockfile"
    if name.endswith((".min.js", ".min.css", ".map")):
        return "minified"
    if len(data) > max_indexed_bytes:
        return "too-large"
    return None


def scan_batches(
    root: Path, files: list[Path], max_indexed_bytes: int, batch_size: int = 250
) -> Iterator[list[ScannedFile]]:
    for start in range(0, len(files), batch_size):
        yield [describe(root, rel, max_indexed_bytes) for rel in files[start : start + batch_size]]
