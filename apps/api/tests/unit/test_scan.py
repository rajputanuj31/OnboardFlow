from __future__ import annotations

from pathlib import Path

import pytest

from onboardflow.ingest.scan import SnapshotTooLargeError, describe, list_files, scan_batches


def _write(root: Path, rel: str, data: bytes | str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    _write(tmp_path, "src/app.py", "def main():\n    return 1\n")
    _write(tmp_path, "src/copy_of_app.py", "def main():\n    return 1\n")  # same content
    _write(tmp_path, "README.md", "# Demo")
    _write(tmp_path, "node_modules/left-pad/index.js", "module.exports = 1")
    _write(tmp_path, "dist/bundle.js", "built")
    _write(tmp_path, ".git/config", "[core]")
    _write(tmp_path, "pkg/__pycache__/x.pyc", b"\x00\x01")
    _write(tmp_path, "package-lock.json", "{}")
    _write(tmp_path, "static/app.min.js", "a=1")
    _write(tmp_path, "assets/logo.png", b"\x89PNG\x00\x00binary")
    _write(tmp_path, "data/big.txt", "x" * 3000)
    _write(
        tmp_path,
        "models/weights.bin",
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\n",
    )
    return tmp_path


def test_ignored_directories_are_not_recorded(tree: Path) -> None:
    paths = {p.as_posix() for p in list_files(tree, max_files=100)}
    assert "src/app.py" in paths
    assert not any(
        p.startswith(("node_modules/", "dist/", ".git/", "pkg/__pycache__/")) for p in paths
    )


def test_non_indexable_files_are_recorded_with_a_reason(tree: Path) -> None:
    files = list_files(tree, max_files=100)
    scanned = {f.path: f for batch in scan_batches(tree, files, 2048) for f in batch}

    assert scanned["src/app.py"].indexed
    assert scanned["src/app.py"].language == "python"
    assert scanned["src/app.py"].line_count == 2
    assert scanned["README.md"].line_count == 1  # no trailing newline still counts
    reasons = {path: f.skip_reason for path, f in scanned.items() if not f.indexed}
    assert reasons == {
        "package-lock.json": "lockfile",
        "static/app.min.js": "minified",
        "assets/logo.png": "binary",
        "data/big.txt": "too-large",
        "models/weights.bin": "lfs-pointer",
    }
    assert scanned["assets/logo.png"].line_count is None


def test_identical_content_shares_one_hash(tree: Path) -> None:
    a = describe(tree, Path("src/app.py"), 2048)
    b = describe(tree, Path("src/copy_of_app.py"), 2048)
    assert a.sha256 == b.sha256


def test_too_many_files_is_rejected(tree: Path) -> None:
    with pytest.raises(SnapshotTooLargeError):
        list_files(tree, max_files=3)


def test_symlinks_are_never_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "link").symlink_to(outside / "secret.txt")
    (repo / "linkdir").symlink_to(outside, target_is_directory=True)

    files = list_files(repo, max_files=10)
    assert [p.as_posix() for p in files] == ["link"]  # the dir link is not descended
    scanned = describe(repo, files[0], 2048)
    assert scanned.skip_reason == "symlink"
    assert scanned.size_bytes == 0  # content behind the link was not read
