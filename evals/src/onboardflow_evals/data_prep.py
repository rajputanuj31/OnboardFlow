"""Build the committed eval datasets from their pinned public sources.

Downloads are pinned to exact revisions and cached under evals/.cache/raw/. Downloaded
content is treated as data only: nothing from it is executed.

- SWE-QA-Bench (Apache-2.0): repo-level questions + reference answers.
- SWE-bench (princeton-nlp/SWE-bench, test split): real issues + the patch that fixed them;
  gold files/functions for localization come from the patch, mapped onto the file at the
  issue's base commit with tree-sitter.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import tree_sitter_python
from tree_sitter import Language, Node, Parser
from unidiff.patch import PatchedFile, PatchSet

from onboardflow_evals.datasets import Item, save_items

SWE_QA_REPO = "peng-weihan/SWE-QA-Bench"
SWE_QA_REV = "c13deac7a0d99b0ca2e593e004c4739475785b08"
# Commits the SWE-QA answers were written against (from the benchmark's repo_commit.txt).
SWE_QA_TARGETS = {"requests": ("psf/requests", "46e939b"), "flask": ("pallets/flask", "85c5d93")}

SWE_BENCH_DATASET = "princeton-nlp/SWE-bench"
SWE_BENCH_REV = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"
SWE_BENCH_REPOS = ("psf/requests", "pallets/flask")

_PY = Language(tree_sitter_python.language())


class Fetcher:
    """GET with a disk cache keyed by URL; every URL here is pinned to an immutable revision."""

    def __init__(self, cache_dir: Path, client: httpx.Client | None = None) -> None:
        self.cache_dir = cache_dir
        self.client = client or httpx.Client(timeout=60.0, follow_redirects=True)

    def get(self, url: str) -> bytes:
        name = url.split("://", 1)[1].replace("/", "__")
        path = self.cache_dir / name
        if path.exists():
            return path.read_bytes()
        response = self.client.get(url)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return response.content


# ── SWE-QA ────────────────────────────────────────────────────────────────────


def prepare_swe_qa(fetcher: Fetcher, out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for short, (repo, commit) in SWE_QA_TARGETS.items():
        url = (
            f"https://raw.githubusercontent.com/{SWE_QA_REPO}/{SWE_QA_REV}/Benchmark/{short}.jsonl"
        )
        rows = [json.loads(line) for line in fetcher.get(url).decode().splitlines() if line.strip()]
        items = [
            Item(
                id=f"swe-qa/{short}/{index:03d}",
                suite="answer",
                slice="popular",
                repo=repo,
                commit=commit,
                question=row["question"],
                reference_answer=row["answer"],
                source=f"swe-qa-bench@{SWE_QA_REV[:7]}",
            )
            for index, row in enumerate(rows)
        ]
        save_items(out_dir / "answer" / f"swe_qa_{short}.jsonl", items)
        counts[f"swe_qa_{short}"] = len(items)
    return counts


# ── SWE-bench ─────────────────────────────────────────────────────────────────


def touched_source_lines(patched_file: PatchedFile) -> set[int]:
    """1-based line numbers in the *original* file that the patch removes or inserts at."""
    lines: set[int] = set()
    for hunk in patched_file:
        last_source = hunk.source_start
        for line in hunk:
            if line.source_line_no is not None:
                last_source = line.source_line_no
            if line.is_removed and line.source_line_no is not None:
                lines.add(line.source_line_no)
            elif line.is_added:
                lines.add(max(last_source, 1))  # insertion point, anchored to the line above
    return lines


def _qualified_name(node: Node) -> str | None:
    parts: list[str] = []
    current: Node | None = node
    while current is not None:
        if current.type in ("function_definition", "class_definition"):
            name = current.child_by_field_name("name")
            if name is not None and name.text is not None:
                parts.append(name.text.decode())
        current = current.parent
    return ".".join(reversed(parts)) or None


def enclosing_functions(source: str, lines: Iterable[int]) -> list[str]:
    """Innermost function/class (qualified, e.g. `Session.request`) containing each line."""
    tree = Parser(_PY).parse(source.encode())
    found: list[str] = []
    for line_no in sorted(set(lines)):
        row = line_no - 1
        node = tree.root_node.descendant_for_point_range((row, 0), (row, 0))
        while node is not None and node.type not in ("function_definition", "class_definition"):
            node = node.parent
        if node is None:
            continue
        name = _qualified_name(node)
        if name and name not in found:
            found.append(name)
    return found


def gold_from_patch(patch: str, file_at_base: dict[str, str]) -> tuple[list[str], list[str]]:
    files: list[str] = []
    functions: list[str] = []
    for patched_file in PatchSet(io.StringIO(patch)):
        path = patched_file.source_file.removeprefix("a/")
        if patched_file.is_added_file:
            path = patched_file.target_file.removeprefix("b/")
        files.append(path)
        source = file_at_base.get(path)
        if source is not None and path.endswith(".py"):
            for name in enclosing_functions(source, touched_source_lines(patched_file)):
                functions.append(f"{path}::{name}")
    return files, functions


def prepare_swe_bench(fetcher: Fetcher, out_dir: Path) -> dict[str, int]:
    url = (
        f"https://huggingface.co/datasets/{SWE_BENCH_DATASET}/resolve/{SWE_BENCH_REV}"
        "/data/test-00000-of-00001.parquet"
    )
    table = pq.read_table(io.BytesIO(fetcher.get(url)))
    rows = [row for row in table.to_pylist() if row["repo"] in SWE_BENCH_REPOS]
    items: list[Item] = []
    for row in sorted(rows, key=lambda r: r["instance_id"]):
        repo, commit = row["repo"], row["base_commit"]
        sources: dict[str, str] = {}
        for patched_file in PatchSet(io.StringIO(row["patch"])):
            if patched_file.is_added_file:
                continue
            path = patched_file.source_file.removeprefix("a/")
            raw = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
            sources[path] = fetcher.get(raw).decode("utf-8", errors="replace")
        gold_files, gold_functions = gold_from_patch(row["patch"], sources)
        items.append(
            Item(
                id=f"swe-bench/{row['instance_id']}",
                suite="localization",
                slice="popular",
                repo=repo,
                commit=commit,
                question=row["problem_statement"],
                gold_files=gold_files,
                gold_functions=gold_functions,
                source=f"swe-bench@{SWE_BENCH_REV[:7]}",
            )
        )
    save_items(out_dir / "localization" / "swe_bench_requests_flask.jsonl", items)
    return {"swe_bench_requests_flask": len(items)}
