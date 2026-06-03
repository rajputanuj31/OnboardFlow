"""Online LLM-as-judge.

After each user-facing response is streamed, we schedule an evaluator pass via
`spawn_eval(...)` — it runs in a fire-and-forget task and writes a feedback
entry per axis to LangSmith on the run that produced the answer. The evaluator
never blocks the response; failures are logged and swallowed.

LangSmith integration is best-effort: if `LANGCHAIN_API_KEY` is unset or the
client errors, scores are logged locally and dropped. The run_id is captured by
the SSE generator in main.py from the first `on_chain_start` event.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config import settings
from graph.helpers import _llm
from prompts import PR_REVIEW_JUDGE_PROMPT, QA_JUDGE_PROMPT

logger = logging.getLogger(__name__)


class _Score(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    rationale: str = ""
    failure_modes: list[str] = Field(default_factory=list)


_BG_TASKS: set[asyncio.Task] = set()


def spawn_eval(coro) -> None:
    """Schedule an eval coroutine without blocking the caller. Holds a hard
    reference to the task so Python's GC doesn't reap it mid-flight (the
    `"Task was destroyed but it is pending"` warning)."""
    if not settings.eval_enabled:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("spawn_eval called with no running event loop; skipping.")
        return
    task = loop.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


async def evaluate_qa(
    *,
    question: str,
    answer: str,
    rag_context: str,
    repo_name: str,
    run_id: str | None,
    api_key: str,
) -> _Score | None:
    """Score a Q&A answer and write LangSmith feedback."""
    try:
        prompt = QA_JUDGE_PROMPT.format(
            repo_name=repo_name,
            question=question,
            rag_context=rag_context[:6000] or "(no retrieved snippets)",
            answer=answer[:6000],
        )
        llm = _llm(api_key).with_structured_output(_Score)
        score: _Score = await llm.ainvoke([HumanMessage(content=prompt)])
        await _log_feedback("qa", score, run_id)
        return score
    except Exception as e:
        logger.warning("evaluate_qa failed: %s", e)
        return None


async def evaluate_pr_review(
    *,
    pr_title: str,
    pr_diff: str,
    review: str,
    run_id: str | None,
    api_key: str,
) -> _Score | None:
    """Score a PR review and write LangSmith feedback."""
    try:
        prompt = PR_REVIEW_JUDGE_PROMPT.format(
            pr_title=pr_title,
            pr_diff=pr_diff[:8000],
            review=review[:6000],
        )
        llm = _llm(api_key).with_structured_output(_Score)
        score: _Score = await llm.ainvoke([HumanMessage(content=prompt)])
        await _log_feedback("pr_review", score, run_id)
        return score
    except Exception as e:
        logger.warning("evaluate_pr_review failed: %s", e)
        return None


async def _log_feedback(scope: str, score: _Score, run_id: str | None) -> None:
    """Write per-axis feedback to LangSmith. Logs locally regardless."""
    logger.info(
        "[eval:%s] accuracy=%d completeness=%d groundedness=%d failure_modes=%s",
        scope, score.accuracy, score.completeness, score.groundedness, score.failure_modes,
    )
    if not run_id:
        return
    try:
        from langsmith import Client
    except ImportError:
        return
    try:
        client = Client()
    except Exception as e:
        logger.debug("LangSmith client not available: %s", e)
        return

    def _write() -> None:
        for axis in ("accuracy", "completeness", "groundedness"):
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=f"{scope}_{axis}",
                    score=getattr(score, axis),
                    comment=score.rationale,
                )
            except Exception as e:
                logger.debug("LangSmith create_feedback(%s) failed: %s", axis, e)
        if score.failure_modes:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=f"{scope}_failure_modes",
                    value=",".join(score.failure_modes),
                    comment=score.rationale,
                )
            except Exception as e:
                logger.debug("LangSmith failure_modes feedback failed: %s", e)

    # client.create_feedback is sync; run in thread to keep eval async-friendly.
    await asyncio.to_thread(_write)
