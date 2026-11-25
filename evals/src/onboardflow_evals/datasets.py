"""Eval items and their on-disk format (one JSON object per line).

Suites:
- answer:       repo questions with a reference answer, scored by the judge.
- localization: a real issue; did the system point at the files the fix changed?
- abstention:   questions about things the repo does not have; correct = "not found".

Slices: "popular" repos are heavily represented in model training data (requests, flask),
"fresh" repos are not; reporting both exposes memorisation (see the design plan, 10.9).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Suite = Literal["answer", "localization", "abstention"]
Slice = Literal["popular", "fresh"]
SUITES: tuple[Suite, ...] = ("answer", "localization", "abstention")


class Item(BaseModel):
    id: str
    suite: Suite
    slice: Slice
    repo: str = Field(description="owner/name on GitHub")
    commit: str = Field(description="commit the question and gold labels refer to")
    question: str
    reference_answer: str | None = None
    gold_files: list[str] = Field(default_factory=list)
    gold_functions: list[str] = Field(default_factory=list)
    source: str = Field(description="dataset of origin, e.g. swe-qa-bench, swe-bench, handwritten")

    @model_validator(mode="after")
    def _suite_requirements(self) -> Item:
        if self.suite == "answer" and not self.reference_answer:
            raise ValueError(f"{self.id}: answer items need a reference_answer")
        if self.suite == "localization" and not self.gold_files:
            raise ValueError(f"{self.id}: localization items need gold_files")
        if self.suite == "abstention" and (self.gold_files or self.reference_answer):
            raise ValueError(f"{self.id}: abstention items have no gold answer or files")
        return self


def load_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as handle:
        return [Item.model_validate_json(line) for line in handle if line.strip()]


def save_items(path: Path, items: Iterable[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.model_dump(exclude_defaults=True), ensure_ascii=False))
            handle.write("\n")


def load_suite(data_dir: Path, suite: Suite) -> list[Item]:
    items: list[Item] = []
    for path in sorted((data_dir / suite).glob("*.jsonl")):
        items.extend(load_items(path))
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate item ids in {suite}: {sorted(duplicates)}")
    return items


def render_question(item: Item) -> str:
    """The exact text every system receives, so systems are compared on identical input."""
    if item.suite == "localization":
        issue = item.question
        if len(issue) > 6000:
            issue = issue[:6000] + "\n[issue text truncated]"
        return (
            f"Here is a GitHub issue reported against the {item.repo} repository:\n\n"
            f"{issue}\n\n"
            "Which source files would need to change to fix it? List the file paths, "
            "most likely first, then briefly explain why."
        )
    return item.question
