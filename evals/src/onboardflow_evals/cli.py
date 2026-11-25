"""Command line: `of-eval <command>` (wrapped by the Makefile's eval-* targets).

prepare                  rebuild the public datasets (SWE-QA, SWE-bench) from pinned sources
estimate                 print the cost estimate for a run without spending anything
run      --system S      run a system over the suites (resumable)
judge                    judge every system's answer/abstention outputs in a run
report                   write reports/<run>.md
baseline                 run + judge + report for no-context and v2 (the step 2 baseline)
quick                    small cached subset for the PR gate (no-context until v3 exists)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from onboardflow_evals.cache import DiskCache
from onboardflow_evals.data_prep import Fetcher, prepare_swe_bench, prepare_swe_qa
from onboardflow_evals.datasets import SUITES, Item, Suite, load_suite
from onboardflow_evals.judge import Judge
from onboardflow_evals.llm import LLMClient, build_client
from onboardflow_evals.report import write_report
from onboardflow_evals.runner import BudgetExceededError, Run, check_budget, judge_suite, run_suite
from onboardflow_evals.settings import REPO_ROOT, EvalSettings, get_settings
from onboardflow_evals.systems.base import System
from onboardflow_evals.systems.no_context import NoContextSystem
from onboardflow_evals.systems.v2 import V2System, prepare_v2

SYSTEMS = ("no-context", "v2")


def _secret(value: object) -> str | None:
    getter = getattr(value, "get_secret_value", None)
    return getter() if getter else None


def _llm(settings: EvalSettings) -> LLMClient:
    return build_client(
        DiskCache(settings.cache_dir / "llm"),
        _secret(settings.anthropic_api_key),
        _secret(settings.openai_api_key),
    )


def _items(
    settings: EvalSettings, suites: list[Suite], limit: int | None
) -> dict[Suite, list[Item]]:
    loaded = {suite: load_suite(settings.data_dir, suite) for suite in suites}
    return {s: (items[:limit] if limit else items) for s, items in loaded.items()}


def _require(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is not set. Add it to the repo-root .env (never commit it).")
    return value


def _judge(settings: EvalSettings, llm: LLMClient) -> Judge:
    _require(_secret(settings.openai_api_key), "OPENAI_API_KEY")
    return Judge(llm, settings.judge_model)


def _no_context(settings: EvalSettings, llm: LLMClient) -> NoContextSystem:
    return NoContextSystem(
        llm,
        settings.answerer_provider,
        settings.answerer_model,
        settings.answerer_effort,
        settings.answerer_max_tokens,
        settings.concurrency,
    )


async def _open_system(
    name: str, settings: EvalSettings, llm: LLMClient, stack: AsyncExitStack
) -> System:
    if name == "no-context":
        key = (
            settings.anthropic_api_key
            if settings.answerer_provider == "anthropic"
            else settings.openai_api_key
        )
        _require(_secret(key), f"{settings.answerer_provider.upper()}_API_KEY")
        return _no_context(settings, llm)
    root = await prepare_v2(settings.cache_dir, REPO_ROOT)
    system = V2System(
        root=root,
        openai_key=_require(_secret(settings.openai_api_key), "OPENAI_API_KEY"),
        github_token=_require(_secret(settings.github_token), "GITHUB_TOKEN"),
        qdrant_url=settings.v2_qdrant_url,
        port=settings.v2_port,
        proxy_port=settings.usage_proxy_port,
    )
    return await stack.enter_async_context(system)


def _estimator(name: str, settings: EvalSettings, llm: LLMClient) -> System:
    """A system object used only for cost estimates (no servers started)."""
    if name == "no-context":
        return _no_context(settings, llm)
    return V2System(Path(), "", "", "", 0, 0)


def _estimate(
    settings: EvalSettings,
    llm: LLMClient,
    run: Run,
    systems: list[str],
    items: dict[Suite, list[Item]],
) -> float:
    total = 0.0
    for name in systems:
        system = _estimator(name, settings, llm)
        for suite, suite_items in items.items():
            done = {o.item_id for o in run.outputs(name, suite) if o.error is None}
            total += system.estimate_cost_usd([i for i in suite_items if i.id not in done])
    judge = Judge(llm, settings.judge_model)
    n_answer = len(items.get("answer", [])) * len(systems)
    n_abst = len(items.get("abstention", [])) * len(systems)
    total += judge.estimate_cost_usd(n_answer, n_abst)
    return total


async def cmd_run(args: argparse.Namespace, settings: EvalSettings) -> None:
    llm = _llm(settings)
    run = Run(settings.runs_dir, args.run)
    suites: list[Suite] = list(SUITES) if args.suite == "all" else [args.suite]
    items = _items(settings, suites, args.limit)
    estimate = _estimate(settings, llm, run, [args.system], {s: items[s] for s in suites})
    print(f"Estimated cost: ${estimate:.2f} (already spent in this run: ${run.spent():.2f})")
    check_budget(estimate, run.spent(), settings.budget_usd, args.yes)
    async with AsyncExitStack() as stack:
        system = await _open_system(args.system, settings, llm, stack)
        run.record_system(
            system.name,
            settings.system_config(system.name),
            settings.judge_model,
            settings.data_dir,
        )
        for suite in suites:
            n = await run_suite(run, system, suite, items[suite])
            print(f"  {system.name}/{suite}: {n} items run")
        if isinstance(system, V2System) and system.ingest_costs:
            # One-time repo ingestion (embedding + summaries), reported separately from $/question.
            for repo, usd in system.ingest_costs.items():
                run.add_spend(f"v2-ingest:{repo}", usd)
    print(f"Spent so far in run '{run.name}': ${run.spent():.2f}")


async def cmd_judge(args: argparse.Namespace, settings: EvalSettings) -> None:
    llm = _llm(settings)
    judge = _judge(settings, llm)
    _require(_secret(settings.openai_api_key), "OPENAI_API_KEY")
    run = Run(settings.runs_dir, args.run)
    items = _items(settings, ["answer", "abstention"], args.limit)
    for system in run.systems():
        for suite in ("answer", "abstention"):
            n = await judge_suite(run, judge, system, suite, items[suite], settings.concurrency)
            print(f"  judged {system}/{suite}: {n} items")
    print(f"Spent so far in run '{run.name}': ${run.spent():.2f}")


def cmd_report(args: argparse.Namespace, settings: EvalSettings) -> None:
    run = Run(settings.runs_dir, args.run)
    items = _items(settings, list(SUITES), args.limit)
    path = write_report(run, items, settings.reports_dir)
    print(f"Report written to {path.relative_to(REPO_ROOT)}")


def cmd_estimate(args: argparse.Namespace, settings: EvalSettings) -> None:
    llm = _llm(settings)
    run = Run(settings.runs_dir, args.run)
    items = _items(settings, list(SUITES), args.limit)
    estimate = _estimate(settings, llm, run, list(SYSTEMS), items)
    counts = ", ".join(f"{s}={len(v)}" for s, v in items.items())
    print(f"Items: {counts}. Estimated cost for {', '.join(SYSTEMS)} + judge: ${estimate:.2f}")


def cmd_prepare(_: argparse.Namespace, settings: EvalSettings) -> None:
    fetcher = Fetcher(settings.cache_dir / "raw" / "http")
    print(prepare_swe_qa(fetcher, settings.data_dir))
    print(prepare_swe_bench(fetcher, settings.data_dir))


async def cmd_baseline(args: argparse.Namespace, settings: EvalSettings) -> None:
    for system in SYSTEMS:
        await cmd_run(
            argparse.Namespace(**{**vars(args), "system": system, "suite": "all"}), settings
        )
    await cmd_judge(args, settings)
    cmd_report(args, settings)


async def cmd_quick(args: argparse.Namespace, settings: EvalSettings) -> None:
    quick = argparse.Namespace(
        run="quick", limit=20, yes=args.yes, system="no-context", suite="answer"
    )
    await cmd_run(quick, settings)
    await cmd_judge(quick, settings)
    cmd_report(quick, settings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="of-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "judge", "report", "estimate", "baseline", "quick", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--run", default="baseline", help="run name (directory under runs/)")
        p.add_argument("--limit", type=int, default=None, help="first N items per suite")
        p.add_argument("--yes", action="store_true", help="proceed even above EVAL_BUDGET_USD")
        if name == "run":
            p.add_argument("--system", choices=SYSTEMS, required=True)
            p.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        match args.command:
            case "run":
                asyncio.run(cmd_run(args, settings))
            case "judge":
                asyncio.run(cmd_judge(args, settings))
            case "report":
                cmd_report(args, settings)
            case "estimate":
                cmd_estimate(args, settings)
            case "prepare":
                cmd_prepare(args, settings)
            case "baseline":
                asyncio.run(cmd_baseline(args, settings))
            case "quick":
                asyncio.run(cmd_quick(args, settings))
    except BudgetExceededError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
