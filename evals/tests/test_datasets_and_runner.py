from __future__ import annotations

from pathlib import Path

import pytest

from onboardflow_evals.datasets import SUITES, Item, load_suite, render_question
from onboardflow_evals.report import write_report
from onboardflow_evals.runner import BudgetExceededError, Run, check_budget, run_suite
from onboardflow_evals.settings import EVALS_ROOT
from onboardflow_evals.systems.base import SystemOutput

DATA_DIR = EVALS_ROOT / "data"


@pytest.mark.parametrize("suite", SUITES)
def test_committed_datasets_are_valid(suite: str) -> None:
    items = load_suite(DATA_DIR, suite)  # type: ignore[arg-type]
    assert items, f"no items for {suite}"
    assert all(item.suite == suite for item in items)


def test_dataset_sizes_match_the_plan() -> None:
    sizes = {suite: len(load_suite(DATA_DIR, suite)) for suite in SUITES}
    assert sizes == {"answer": 116, "localization": 55, "abstention": 30}


def test_abstention_items_cannot_carry_answers() -> None:
    with pytest.raises(ValueError, match="no gold"):
        Item(
            id="x",
            suite="abstention",
            slice="fresh",
            repo="o/r",
            commit="c",
            question="?",
            reference_answer="leak",
            source="t",
        )


def test_localization_question_wraps_the_issue() -> None:
    item = Item(
        id="x",
        suite="localization",
        slice="popular",
        repo="psf/requests",
        commit="c",
        question="GET sends Content-Length",
        gold_files=["a.py"],
        source="t",
    )
    text = render_question(item)
    assert "psf/requests" in text
    assert "GET sends Content-Length" in text
    assert "file paths" in text


def test_budget_guard() -> None:
    check_budget(estimate=5, spent=10, budget=20, yes=False)
    check_budget(estimate=50, spent=0, budget=20, yes=True)
    with pytest.raises(BudgetExceededError, match="--yes"):
        check_budget(estimate=15, spent=10, budget=20, yes=False)


class FlakySystem:
    name = "fake"
    concurrency = 2

    def __init__(self, fail_ids: set[str]) -> None:
        self.fail_ids = fail_ids
        self.calls: list[str] = []

    async def answer(self, item: Item) -> SystemOutput:
        self.calls.append(item.id)
        if item.id in self.fail_ids:
            raise RuntimeError("transient")
        return SystemOutput(
            item_id=item.id,
            system=self.name,
            answer=f"see {item.id}.py",
            files=[f"{item.id}.py"],
            cost_usd=0.5,
            latency_ms=10,
        )

    def estimate_cost_usd(self, items: list[Item]) -> float:
        return 0.5 * len(items)


def _items(n: int) -> list[Item]:
    return [
        Item(
            id=f"i{k}",
            suite="localization",
            slice="popular",
            repo="o/r",
            commit="c",
            question="q",
            gold_files=[f"i{k}.py"],
            source="t",
        )
        for k in range(n)
    ]


async def test_run_resumes_and_retries_only_failures(tmp_path: Path) -> None:
    run = Run(tmp_path, "test")
    items = _items(4)

    first = FlakySystem(fail_ids={"i2"})
    assert await run_suite(run, first, "localization", items) == 4
    outputs = run.outputs("fake", "localization")
    assert sum(1 for o in outputs if o.error) == 1
    assert run.spent() == pytest.approx(1.5)  # failures cost nothing

    second = FlakySystem(fail_ids=set())
    assert await run_suite(run, second, "localization", items) == 1
    assert second.calls == ["i2"]
    outputs = run.outputs("fake", "localization")
    assert sorted(o.item_id for o in outputs) == ["i0", "i1", "i2", "i3"]
    assert all(o.error is None for o in outputs)


async def test_report_renders_localization_metrics(tmp_path: Path) -> None:
    run = Run(tmp_path / "runs", "test")
    items = _items(3)
    await run_suite(run, FlakySystem(fail_ids=set()), "localization", items)
    path = write_report(run, {"localization": items}, tmp_path / "reports")
    text = path.read_text()
    assert "hit@5" in text
    assert "100.0 [100.0, 100.0]" in text  # every item names its gold file first


async def test_retry_discards_judgment_of_the_failed_answer(tmp_path: Path) -> None:
    from onboardflow_evals.runner import Judgment, _append

    run = Run(tmp_path, "test")
    items = _items(2)
    await run_suite(run, FlakySystem(fail_ids={"i1"}), "localization", items)
    for item_id in ("i0", "i1"):  # pretend both were judged, including the empty failure
        _append(
            run.judgments_path("fake", "localization"),
            Judgment(item_id=item_id, system="fake", suite="localization", score_100=0),
        )

    await run_suite(run, FlakySystem(fail_ids=set()), "localization", items)

    assert [j.item_id for j in run.judgments("fake", "localization")] == ["i0"]
