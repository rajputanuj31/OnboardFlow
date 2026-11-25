"""Per-model token prices, used for cost reporting and the pre-run spend guard.

Prices are USD per million tokens. A model without an entry cannot be run: the spend
guard would be guessing, so the harness fails loudly instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    source: str


PRICES: dict[str, Price] = {
    # Anthropic first-party API rates (claude-api reference, cached 2026-11-03).
    "claude-sonnet-5-5": Price(2.00, 10.00, "anthropic 2026-11-03"),
    "claude-haiku-5-5": Price(0.10, 0.50, "anthropic 2026-11-03"),
    "claude-opus-5-5": Price(4.00, 20.00, "anthropic 2026-11-03"),
    # OpenAI judge candidates (developers.openai.com/api/docs/pricing, standard, 2026-11-25).
    "gpt-6.2-sol": Price(2.00, 10.00, "openai 2026-11-25"),
    "gpt-6.1-terra": Price(1.00, 5.00, "openai 2026-11-25"),
    # OpenAI models used by v2 (fixed by v2's code).
    "gpt-4o-mini": Price(0.15, 0.60, "openai 2026-11-25"),
    "text-embedding-3-small": Price(0.02, 0.0, "openai list price"),
}


class UnknownPriceError(KeyError):
    pass


def price_for(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError as exc:
        raise UnknownPriceError(
            f"No price for model {model!r}. Add it to onboardflow_evals/pricing.py "
            "before running, so cost estimates and reports stay honest."
        ) from exc


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = price_for(model)
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
