"""Turn a run's outputs and judgments into a markdown report with confidence intervals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from onboardflow_evals.datasets import Item, Slice, Suite
from onboardflow_evals.judge import ANSWER_DIMENSIONS
from onboardflow_evals.metrics import (
    Estimate,
    bootstrap_mean,
    hit_at_k,
    paired_difference,
    percentile,
    reciprocal_rank,
)
from onboardflow_evals.paths import first_hit_rank
from onboardflow_evals.runner import Run
from onboardflow_evals.systems.base import SystemOutput

SLICES: tuple[Slice, ...] = ("popular", "fresh")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(estimate: Estimate) -> str:
    return estimate.fmt(digits=1, scale=100)


class Report:
    def __init__(self, run: Run, items: dict[Suite, list[Item]]) -> None:
        self.run = run
        self.items = items
        self.systems = run.systems()

    # ── per-item values ──
    def answer_scores(self, system: str, slice_: Slice | None = None) -> dict[str, float]:
        wanted = {i.id for i in self.items.get("answer", []) if slice_ is None or i.slice == slice_}
        return {
            j.item_id: j.score_100
            for j in self.run.judgments(system, "answer")
            if j.item_id in wanted and j.score_100 is not None
        }

    def localization_ranks(self, system: str) -> dict[str, int | None]:
        gold = {i.id: i.gold_files for i in self.items.get("localization", [])}
        return {
            o.item_id: first_hit_rank(o.files, gold[o.item_id])
            for o in self.run.outputs(system, "localization")
            if o.item_id in gold
        }

    def abstention_verdicts(self, system: str) -> dict[str, str]:
        return {j.item_id: j.verdict for j in self.run.judgments(system, "abstention") if j.verdict}

    def all_outputs(self, system: str) -> list[SystemOutput]:
        return [o for suite in self.items for o in self.run.outputs(system, suite)]

    # ── sections ──
    def summary(self) -> str:
        rows = []
        for system in self.systems:
            popular = bootstrap_mean(list(self.answer_scores(system, "popular").values()))
            fresh = bootstrap_mean(list(self.answer_scores(system, "fresh").values()))
            ranks = list(self.localization_ranks(system).values())
            hit5 = bootstrap_mean([hit_at_k(r, 5) for r in ranks])
            verdicts = list(self.abstention_verdicts(system).values())
            abstained = bootstrap_mean([1.0 if v == "abstained" else 0.0 for v in verdicts])
            outputs = [o for o in self.all_outputs(system) if o.error is None]
            cost = bootstrap_mean([o.cost_usd for o in outputs])
            rows.append(
                [
                    f"**{system}**",
                    popular.fmt(1),
                    fresh.fmt(1),
                    _pct(hit5),
                    _pct(abstained),
                    f"${cost.mean:.4f}" if cost.n else "n/a",
                    f"{percentile([o.latency_ms for o in outputs], 50) / 1000:.1f} s",
                ]
            )
        return _table(
            [
                "System",
                "Answer score, popular (0-100)",
                "Answer score, fresh (0-100)",
                "Localization hit@5 (%)",
                "Correct abstention (%)",
                "$ / question",
                "p50 latency",
            ],
            rows,
        )

    def answer_section(self) -> str:
        rows = []
        for system in self.systems:
            for slice_ in SLICES:
                scores = self.answer_scores(system, slice_)
                if not scores:
                    continue
                judgments = [j for j in self.run.judgments(system, "answer") if j.item_id in scores]
                dims = [
                    f"{sum((j.scores or {}).get(d, 0) for j in judgments) / len(judgments):.2f}"
                    for d in ANSWER_DIMENSIONS
                ]
                est = bootstrap_mean(list(scores.values()))
                rows.append([system, slice_, str(est.n), est.fmt(1), *dims])
        table = _table(["System", "Slice", "n", "Score (0-100) [95% CI]", *ANSWER_DIMENSIONS], rows)
        return table + "\n\n" + self._paired("answer")

    def localization_section(self) -> str:
        rows = []
        for system in self.systems:
            ranks = list(self.localization_ranks(system).values())
            if not ranks:
                continue
            rows.append(
                [
                    system,
                    str(len(ranks)),
                    _pct(bootstrap_mean([hit_at_k(r, 1) for r in ranks])),
                    _pct(bootstrap_mean([hit_at_k(r, 5) for r in ranks])),
                    _pct(bootstrap_mean([hit_at_k(r, 10) for r in ranks])),
                    bootstrap_mean([reciprocal_rank(r) for r in ranks]).fmt(3),
                ]
            )
        table = _table(["System", "n", "hit@1 (%)", "hit@5 (%)", "hit@10 (%)", "MRR"], rows)
        return table + "\n\n" + self._paired("localization")

    def abstention_section(self) -> str:
        rows = []
        for system in self.systems:
            verdicts = list(self.abstention_verdicts(system).values())
            if not verdicts:
                continue
            row = [system, str(len(verdicts))]
            for label in ("abstained", "hedged", "hallucinated"):
                row.append(_pct(bootstrap_mean([1.0 if v == label else 0.0 for v in verdicts])))
            rows.append(row)
        return _table(["System", "n", "Abstained (%)", "Hedged (%)", "Hallucinated (%)"], rows)

    def ops_section(self) -> str:
        rows = []
        for system in self.systems:
            for suite in self.items:
                outputs = self.run.outputs(system, suite)
                if not outputs:
                    continue
                ok = [o for o in outputs if o.error is None]
                latencies = [o.latency_ms / 1000 for o in ok]
                rows.append(
                    [
                        system,
                        suite,
                        str(len(outputs)),
                        str(len(outputs) - len(ok)),
                        str(sum(1 for o in ok if o.stop_reason == "refusal")),
                        f"${sum(o.cost_usd for o in ok):.3f}",
                        f"${(sum(o.cost_usd for o in ok) / len(ok)) if ok else 0:.4f}",
                        f"{percentile(latencies, 50):.1f}",
                        f"{percentile(latencies, 95):.1f}",
                        f"{sum(o.input_tokens for o in ok) / max(len(ok), 1):,.0f}",
                    ]
                )
        return _table(
            [
                "System",
                "Suite",
                "Items",
                "Errors",
                "Refusals",
                "Total $",
                "$ / item",
                "p50 s",
                "p95 s",
                "Input tokens / item",
            ],
            rows,
        )

    def _paired(self, suite: str) -> str:
        if len(self.systems) < 2:
            return ""
        lines = ["Paired differences (same items; positive = second system higher):", ""]
        base = self.systems[0]
        for other in self.systems[1:]:
            if suite == "answer":
                for slice_ in SLICES:
                    a, b = self.answer_scores(base, slice_), self.answer_scores(other, slice_)
                    shared = sorted(set(a) & set(b))
                    if shared:
                        diff = paired_difference([a[i] for i in shared], [b[i] for i in shared])
                        lines.append(
                            f"- {other} - {base}, {slice_}: {diff.fmt(1)} points (n={diff.n})"
                        )
            else:
                ra, rb = self.localization_ranks(base), self.localization_ranks(other)
                shared = sorted(set(ra) & set(rb))
                if shared:
                    diff = paired_difference(
                        [hit_at_k(ra[i], 5) for i in shared], [hit_at_k(rb[i], 5) for i in shared]
                    )
                    lines.append(
                        f"- {other} - {base}, hit@5: {diff.fmt(1, scale=100)} points (n={diff.n})"
                    )
        return "\n".join(lines)

    def render(self) -> str:
        meta = self.run.metadata()
        spend_path = self.run.dir / "spend.json"
        spend = json.loads(spend_path.read_text()) if spend_path.exists() else {}
        settings_rows = [["Run created", str(meta.get("created_at", "?"))]]
        for system, cfg in sorted(meta.get("systems", {}).items()):
            details = ", ".join(
                f"{k}={v}" for k, v in cfg.items() if k not in ("git_sha", "recorded_at")
            )
            sha = str(cfg.get("git_sha", "?"))
            sha = sha[:7] if len(sha) == 40 else sha  # keep markers such as "-dirty"
            settings_rows.append([f"System `{system}`", f"{details} (code `{sha}`)"])
        settings_rows += [
            ["Judge model", f"`{meta.get('judge_model', '?')}`"],
            ["Real spend so far", f"${sum(spend.values()):.2f}"],
            [
                "v2 one-time ingest cost",
                ", ".join(
                    f"{k.split(':', 1)[1]} ${v:.3f}"
                    for k, v in sorted(spend.items())
                    if k.startswith("v2-ingest:")
                )
                or "n/a (repos were already ingested)",
            ],
        ]
        return "\n\n".join(
            [
                f"# Eval report: `{self.run.name}`",
                "Generated by `make eval-report`. Numbers in brackets are 95% bootstrap "
                "confidence intervals; differences between systems use the paired bootstrap.",
                _table(["Setting", "Value"], settings_rows),
                "## Summary",
                self.summary(),
                "## Answer quality (judge, vs reference answers)",
                self.answer_section(),
                "## Localization (did it name the files the real fix changed?)",
                self.localization_section(),
                "## Abstention (questions about features that do not exist)",
                self.abstention_section(),
                "## Cost and latency",
                self.ops_section(),
                "## Known limitations of this run",
                "- v2 can only ingest a repository's latest default branch, while SWE-QA and "
                "SWE-bench items refer to older commits. File matching is path-suffix based, so "
                "moved files still count, but code that changed since may disadvantage v2.\n"
                "- SWE-QA (requests, flask) and SWE-bench issues are public and likely in the "
                "models' training data (`popular` slice). The `fresh` slice (OnboardFlow v2 "
                "itself) is the contamination check.\n"
                "- One judge model (a stronger OpenAI model than the answerer, same provider), "
                "not yet calibrated against human labels (build step 11).",
            ]
        )


def write_report(run: Run, items: dict[Suite, list[Item]], reports_dir: Path) -> Path:
    path = reports_dir / f"{run.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(Report(run, items).render() + "\n", encoding="utf-8")
    return path
