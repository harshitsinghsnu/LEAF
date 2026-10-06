"""Acquisition functions over the O_AL posterior.

This was missing entirely from Revision 1's Algorithm 1 (every proposed
candidate was queried against ground truth, making O_AL's refit a no-op).
Both LEAF and the static-GA baseline MUST use the same acquisition function
and same q -- see research plan §3.2 baseline-parity requirement.
"""
from __future__ import annotations

import numpy as np

from leaf.oracle.surrogate import ALSurrogate


def multi_objective_ucb(surrogate: ALSurrogate, X_pool: np.ndarray, heads: list[str],
                         signs: dict[str, float], beta: float = 2.0) -> np.ndarray:
    """Scalarized multi-objective UCB: sum_h sign_h * (mean_h + beta * std_h).

    signs: +1 to maximize a head (e.g. potency), -1 to minimize (e.g. toxicity).
    Returns one score per pool point; higher is more worth querying.
    """
    score = np.zeros(X_pool.shape[0])
    for h in heads:
        mean, std = surrogate.posterior(X_pool, h)
        score += signs.get(h, 1.0) * (mean + beta * std)
    return score


def expected_hypervolume_improvement(surrogate: ALSurrogate, X_pool: np.ndarray,
                                      heads: list[str], signs: dict[str, float],
                                      current_front: np.ndarray, n_mc: int = 64,
                                      random_state: int = 0) -> np.ndarray:
    """Monte-Carlo EHVI over the surrogate posterior, given the current
    non-dominated front (n_front x n_heads array, already sign-adjusted so
    all objectives are maximized). Config-flagged alternative to UCB (§3.2)."""
    from leaf.archive.pareto import hypervolume

    rng = np.random.default_rng(random_state)
    means, stds = [], []
    for h in heads:
        m, s = surrogate.posterior(X_pool, h)
        means.append(signs.get(h, 1.0) * m)
        stds.append(s)
    means = np.stack(means, axis=1)
    stds = np.stack(stds, axis=1)

    ref_point = np.min(current_front, axis=0) - 1.0 if current_front.size else np.zeros(len(heads))
    base_hv = hypervolume(current_front, ref_point) if current_front.size else 0.0

    ehvi = np.zeros(X_pool.shape[0])
    for i in range(X_pool.shape[0]):
        samples = means[i] + stds[i] * rng.standard_normal((n_mc, len(heads)))
        improvements = []
        for s in samples:
            candidate_front = np.vstack([current_front, s]) if current_front.size else s[None, :]
            improvements.append(hypervolume(candidate_front, ref_point) - base_hv)
        ehvi[i] = np.mean(improvements)
    return ehvi


def select_batch(scores: np.ndarray, q: int) -> np.ndarray:
    """Top-q indices by acquisition score."""
    q = min(q, scores.shape[0])
    return np.argsort(scores)[::-1][:q]
