from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from onboardflow_evals.datasets import Item


class SystemOutput(BaseModel):
    item_id: str
    system: str
    answer: str = ""
    files: list[str] = Field(default_factory=list, description="ranked file paths, for recall@k")
    tool_trace: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    error: str | None = None
    cached: bool = False


class System(Protocol):
    name: str
    concurrency: int

    async def answer(self, item: Item) -> SystemOutput: ...

    def estimate_cost_usd(self, items: list[Item]) -> float: ...
