"""Metrics and uncertainty.

Every headline number gets a 95% bootstrap confidence interval. Comparisons between two
systems use the *paired* bootstrap over per-item differences (same items, both systems),
which is far more sensitive than comparing two independent intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 2000
SEED = 20261125


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def fmt(self, digits: int = 2, scale: float = 1.0) -> str:
        if self.n == 0:
            return "n/a"
        m, lo, hi = self.mean * scale, self.low * scale, self.high * scale
        return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def bootstrap_mean(values: Sequence[float], seed: int = SEED) -> Estimate:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(N_RESAMPLES, arr.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return Estimate(float(arr.mean()), float(low), float(high), int(arr.size))


def paired_difference(a: Sequence[float], b: Sequence[float], seed: int = SEED) -> Estimate:
    """Mean of (b - a) over paired items, with a bootstrap CI. Positive = b is higher."""
    if len(a) != len(b):
        raise ValueError("paired difference needs the same items in both systems")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_mean(diffs.tolist(), seed=seed)


def hit_at_k(first_rank: int | None, k: int) -> float:
    return 1.0 if first_rank is not None and first_rank <= k else 0.0


def reciprocal_rank(first_rank: int | None) -> float:
    return 0.0 if first_rank is None else 1.0 / first_rank


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")
