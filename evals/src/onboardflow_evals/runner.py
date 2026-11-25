"""Run systems over suites and judge their outputs, resumably.

Layout of a run (committed, small JSON):
  runs/<run>/run.json                         config, git sha, dataset hashes
  runs/<run>/spend.json                       real (uncached) spend so far, USD
  runs/<run>/outputs/<system>/<suite>.jsonl   one SystemOutput per item
  runs/<run>/judgments/<system>/<suite>.jsonl one judgment per item

Re-running skips items that already have a result, so an interrupted run resumes where
it stopped and never pays twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from onboardflow_evals.datasets import SUITES, Item, Suite
from onboardflow_evals.judge import Judge
from onboardflow_evals.systems.base import System, SystemOutput


class BudgetExceededError(RuntimeError):
    pass


class Judgment(BaseModel):
    item_id: str
    system: str
    suite: Suite
    score_100: float | None = None
    scores: dict[str, int] | None = None
    verdict: str | None = None
    justification: str = ""
    cost_usd: float = 0.0


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _append(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


class Run:
    def __init__(self, runs_dir: Path, name: str) -> None:
        self.name = name
        self.dir = runs_dir / name

    def outputs_path(self, system: str, suite: Suite) -> Path:
        return self.dir / "outputs" / system / f"{suite}.jsonl"

    def judgments_path(self, system: str, suite: Suite) -> Path:
        return self.dir / "judgments" / system / f"{suite}.jsonl"

    def outputs(self, system: str, suite: Suite) -> list[SystemOutput]:
        return read_jsonl(self.outputs_path(system, suite), SystemOutput)

    def judgments(self, system: str, suite: Suite) -> list[Judgment]:
        return read_jsonl(self.judgments_path(system, suite), Judgment)

    def systems(self) -> list[str]:
        root = self.dir / "outputs"
        return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []

    # ── spend ──
    def spent(self) -> float:
        path = self.dir / "spend.json"
        if not path.exists():
            return 0.0
        data: dict[str, float] = json.loads(path.read_text())
        return sum(data.values())

    def add_spend(self, bucket: str, usd: float) -> None:
        path = self.dir / "spend.json"
        data: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}
        data[bucket] = round(data.get(bucket, 0.0) + usd, 6)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def metadata(self) -> dict[str, Any]:
        path = self.dir / "run.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def record_system(
        self, system: str, system_config: dict[str, Any], judge_model: str, data_dir: Path
    ) -> None:
        """Record what produced each system's outputs. Systems in one run can be run at
        different times, so each keeps its own config, git sha and timestamp."""
        meta = self.metadata() or {
            "name": self.name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        meta["judge_model"] = judge_model
        meta["datasets"] = {suite: _hash_dir(data_dir / suite) for suite in SUITES}
        meta.setdefault("systems", {})[system] = {
            **system_config,
            "git_sha": _git_sha(),
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def check_budget(estimate: float, spent: float, budget: float, yes: bool) -> None:
    if yes or spent + estimate <= budget:
        return
    raise BudgetExceededError(
        f"Estimated ${estimate:.2f} on top of ${spent:.2f} already spent exceeds the "
        f"${budget:.2f} budget (EVAL_BUDGET_USD). Re-run with --yes to proceed anyway."
    )


async def _bounded(
    items: list[Item], concurrency: int, worker: Callable[[Item], Awaitable[None]]
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(item: Item) -> None:
        async with semaphore:
            await worker(item)

    await asyncio.gather(*(guarded(item) for item in items))


def pending(run: Run, system: str, suite: Suite, items: list[Item]) -> list[Item]:
    done = {o.item_id for o in run.outputs(system, suite) if o.error is None}
    return [item for item in items if item.id not in done]


async def run_suite(run: Run, system: System, suite: Suite, items: list[Item]) -> int:
    todo = pending(run, system.name, suite, items)
    path = run.outputs_path(system.name, suite)
    # Drop earlier failed attempts for items we are about to retry.
    kept = [o for o in run.outputs(system.name, suite) if o.item_id not in {i.id for i in todo}]
    if path.exists():
        path.write_text("".join(o.model_dump_json() + "\n" for o in kept), encoding="utf-8")
    # A retried item gets a new answer, so any judgment of the old (failed) answer is stale.
    retried = {item.id for item in todo}
    judged_path = run.judgments_path(system.name, suite)
    if retried and judged_path.exists():
        fresh = [j for j in run.judgments(system.name, suite) if j.item_id not in retried]
        judged_path.write_text("".join(j.model_dump_json() + "\n" for j in fresh), encoding="utf-8")

    async def work(item: Item) -> None:
        try:
            output = await system.answer(item)
        except Exception as exc:  # recorded per item; the run continues
            output = SystemOutput(item_id=item.id, system=system.name, error=repr(exc)[:500])
        _append(path, output)
        if not output.cached:
            run.add_spend(system.name, output.cost_usd)

    await _bounded(todo, system.concurrency, work)
    return len(todo)


async def judge_suite(
    run: Run, judge: Judge, system: str, suite: Suite, items: list[Item], concurrency: int
) -> int:
    by_id = {item.id: item for item in items}
    done = {j.item_id for j in run.judgments(system, suite)}
    outputs = [
        o for o in run.outputs(system, suite) if o.item_id in by_id and o.item_id not in done
    ]
    path = run.judgments_path(system, suite)

    async def work(item: Item) -> None:
        output = next(o for o in outputs if o.item_id == item.id)
        if suite == "answer":
            score, cost = await judge.score_answer(item, output)
            judgment = Judgment(
                item_id=item.id,
                system=system,
                suite=suite,
                score_100=score.score_100,
                scores={
                    d: getattr(score, d)
                    for d in ("correctness", "completeness", "relevance", "clarity", "reasoning")
                },
                justification=score.justification,
                cost_usd=cost,
            )
        else:
            verdict, cost = await judge.classify_abstention(item, output)
            judgment = Judgment(
                item_id=item.id,
                system=system,
                suite=suite,
                verdict=verdict.verdict,
                justification=verdict.justification,
                cost_usd=cost,
            )
        _append(path, judgment)
        run.add_spend("judge", cost)

    await _bounded([by_id[o.item_id] for o in outputs], concurrency, work)
    return len(outputs)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hash_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.jsonl")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]
