"""Find file paths in free-text answers and match them against gold paths.

Matching is suffix-based on path segments, because systems see different layouts of the
same repo: v2 reads the latest commit (`src/requests/models.py`) while an old SWE-bench
issue's gold patch says `requests/models.py`. Suffix matching on whole segments treats
those as the same file without matching unrelated files that merely share a basename
deeper in another package (`a/utils.py` vs `b/utils.py` need the shared parent to match).
"""

from __future__ import annotations

import re

# A path-like token: optional dirs, a filename with an extension of 1-6 word chars.
_PATH_RE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.[A-Za-z][\w]{0,5})(?![\w/])")
_NOT_FILES = {"e.g", "i.e", "etc", "vs"}
_SOURCE_EXTS = {
    "py",
    "pyi",
    "js",
    "jsx",
    "ts",
    "tsx",
    "mjs",
    "cjs",
    "go",
    "rs",
    "java",
    "kt",
    "rb",
    "php",
    "c",
    "h",
    "cc",
    "cpp",
    "hpp",
    "cs",
    "swift",
    "scala",
    "sh",
    "toml",
    "cfg",
    "ini",
    "yaml",
    "yml",
    "json",
    "md",
    "rst",
    "txt",
    "html",
    "css",
    "sql",
    "lock",
}


def normalize(path: str) -> str:
    path = path.strip().strip("`'\"()[]<>,;:").replace("\\", "/")
    for prefix in ("a/", "b/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.strip("/")


def extract_paths(text: str) -> list[str]:
    """File paths in order of first mention (the ranking used for recall@k), deduplicated."""
    seen: list[str] = []
    for match in _PATH_RE.finditer(text):
        candidate = normalize(match.group(1))
        ext = candidate.rsplit(".", 1)[-1].lower()
        if candidate.lower() in _NOT_FILES or ext not in _SOURCE_EXTS:
            continue
        if re.fullmatch(r"\d+(\.\d+)+", candidate):  # version numbers like 2.31.0
            continue
        if candidate not in seen:
            seen.append(candidate)
    return seen


def same_file(predicted: str, gold: str) -> bool:
    """True if one path is a segment-wise suffix of the other (at least the filename)."""
    p = normalize(predicted).split("/")
    g = normalize(gold).split("/")
    shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
    if longer[-len(shorter) :] != shorter:
        return False
    # A bare filename only counts when the gold path is also just a filename or the
    # filename is distinctive enough (not an __init__.py / utils.py style name).
    if len(shorter) == 1 and len(longer) > 1:
        return shorter[0] not in {"__init__.py", "utils.py", "util.py", "helpers.py", "models.py"}
    return True


def first_hit_rank(predicted: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first predicted path matching any gold path."""
    for rank, path in enumerate(predicted, start=1):
        if any(same_file(path, g) for g in gold):
            return rank
    return None
