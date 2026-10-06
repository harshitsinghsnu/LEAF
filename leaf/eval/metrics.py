"""Metrics: hypervolume (delegates to archive.pareto), Recall@k, hypothesis
precision, per-round conformal coverage, inter-agent agreement."""
from __future__ import annotations

import numpy as np


def recall_at_k(archive_fragments: list, known_high_performers: list, k: int,
                 similarity_fn, threshold: float = 0.8) -> float:
    """Fraction of a fixed list of literature-known high-performing lipids
    (MC3-, SM-102-, ALC-0315-adjacent) recovered within the top-k of the
    final archive, matched by fragment similarity threshold."""
    top_k = archive_fragments[:k]
    recovered = 0
    for known in known_high_performers:
        if any(similarity_fn(known, cand) >= threshold for cand in top_k):
            recovered += 1
    return recovered / max(len(known_high_performers), 1)


def hypothesis_precision(ledger) -> float:
    return ledger.precision()


def coverage_vs_round(coverage_history: list[dict]) -> list[dict]:
    """Pass-through formatter for the per-round coverage table required by
    research plan §5.3 / §8 reporting checklist (never a single number)."""
    return [{"round": c["round"], "empirical_coverage": c["empirical_coverage"],
              "target": c["target"]} for c in coverage_history]


def agentic_value_add(hv_leaf: np.ndarray, hv_static_ga: np.ndarray) -> np.ndarray:
    """Paired per-seed hypervolume differences; feed into eval/significance.py
    for the bootstrap CI (primary statistic, §5.2/§7.2) rather than a bare
    mean difference."""
    return np.asarray(hv_leaf) - np.asarray(hv_static_ga)
