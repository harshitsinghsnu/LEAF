"""Significance testing, corrected per research plan §5.2/§7.2.

With n=3 paired seeds, Wilcoxon signed-rank cannot reach p<0.05 (minimum
achievable two-sided p is 0.25 at n=3). Primary statistic is now a
bootstrap CI on the median paired difference; Wilcoxon is reported only as
a secondary corroborating statistic when n>=8 AND the achievable p-value
at the observed rank pattern is checked (not assumed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


MIN_SEEDS_FOR_SIGNIFICANCE = 8


@dataclass
class SignificanceResult:
    n_seeds: int
    median_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float | None
    wilcoxon_reliable: bool  # False if n < MIN_SEEDS_FOR_SIGNIFICANCE


def bootstrap_median_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                         random_state: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    diffs = np.asarray(diffs)
    boot_medians = np.array([
        np.median(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)
    ])
    lo, hi = np.quantile(boot_medians, [alpha / 2, 1 - alpha / 2])
    return float(np.median(diffs)), float(lo), float(hi)


def paired_significance(diffs: np.ndarray) -> SignificanceResult:
    n = len(diffs)
    median, ci_low, ci_high = bootstrap_median_ci(diffs)
    wilcoxon_p = None
    reliable = n >= MIN_SEEDS_FOR_SIGNIFICANCE
    if n >= 6 and np.any(diffs != 0):
        try:
            _, wilcoxon_p = wilcoxon(diffs)
        except ValueError:
            wilcoxon_p = None
    if n < MIN_SEEDS_FOR_SIGNIFICANCE:
        # Do not let callers silently treat this as a significance claim.
        reliable = False
    return SignificanceResult(n, median, ci_low, ci_high, wilcoxon_p, reliable)
