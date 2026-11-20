"""Hypothesis Agent: literature RAG + attribution mining + counterfactual
testing against O_GT (never O_AL -- a hypothesis "validated" against the
surrogate would just be recovering the surrogate's own fit).

Rescoped from Revision 1's "structure-tropism" framing to "structure-
property" (§5.5), since the actual LUMI-lab finding it targets in E5 is a
potency association, not a tropism one. Every write to the Ledger consumes
ground-truth queries that MUST be charged against the budget (§5.10).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import wilcoxon

from leaf.fragments.vocabulary import Fragment


@dataclass
class Hypothesis:
    text: str
    fragment_slot: str          # e.g. "halogenation"
    fragment_value: str         # e.g. "Br"
    property_head: str          # e.g. "potency" -- NOT "tropism" by default (§5.5)
    status: str = "pending"     # pending | validated | rejected
    p_value: float | None = None
    effect_size: float | None = None
    n_pairs: int = 0


@dataclass
class HypothesisLedger:
    entries: list[Hypothesis] = field(default_factory=list)

    def validated(self) -> list[Hypothesis]:
        return [h for h in self.entries if h.status == "validated"]

    def bias_for_proposer(self) -> dict:
        """Fragment-slot -> list of validated values, consumed by
        Proposer.mutation(ledger_bias=...)."""
        bias: dict[str, list[str]] = {}
        for h in self.validated():
            bias.setdefault(h.fragment_slot, []).append(h.fragment_value)
        return bias

    def precision(self) -> float:
        tested = [h for h in self.entries if h.status in ("validated", "rejected")]
        if not tested:
            return float("nan")
        return len(self.validated()) / len(tested)


def make_counterfactual_pairs(archive: list[Fragment], slot: str, value: str,
                               control_value: str, max_pairs: int = 20) -> list[tuple[Fragment, Fragment]]:
    """Matched counterfactual pairs: candidates identical except for the
    hypothesized fragment slot. Both members are then queried against
    O_GT -- this is the query cost that §5.10 fixes the budget accounting
    for."""
    pairs = []
    for frag in archive:
        if getattr(frag, slot) == control_value:
            counterfactual = Fragment(**{**frag.__dict__, slot: value})
            pairs.append((frag, counterfactual))
        if len(pairs) >= max_pairs:
            break
    return pairs


def test_hypothesis(hyp: Hypothesis, control_scores: np.ndarray, treatment_scores: np.ndarray,
                     alpha: float = 0.05) -> Hypothesis:
    """Paired significance test (Wilcoxon signed-rank) over matched
    counterfactual pairs queried against O_GT. Written to the Ledger as
    validated only if p < alpha -- no exceptions (research plan, unchanged
    core rigor claim)."""
    hyp.n_pairs = len(control_scores)
    if hyp.n_pairs < 6:
        # Too few pairs for a meaningful signed-rank test at any alpha --
        # same statistical-power argument as §5.2's seed-count fix.
        hyp.status = "rejected"
        hyp.p_value = 1.0
        hyp.effect_size = 0.0
        return hyp
    diffs = treatment_scores - control_scores
    if np.all(diffs == 0):
        # scipy's wilcoxon raises ValueError here rather than returning a
        # (non-)result -- but "every matched pair shows zero difference"
        # is itself a definitive non-significant outcome, not an error
        # condition, so handle it directly rather than letting the
        # exception propagate and crash the whole sweep. In practice this
        # can happen with a folded Morgan fingerprint (leaf/fragments/
        # embed.py, n_bits=1024): two structurally distinct real
        # candidates can hash to the same bit vector, in which case O_GT
        # legitimately predicts an identical posterior mean for both --
        # not a bug in the significance test, but worth knowing if it
        # happens often (log/inspect hyp.effect_size == 0.0 cases).
        hyp.status = "rejected"
        hyp.p_value = 1.0
        hyp.effect_size = 0.0
        return hyp
    stat, p = wilcoxon(diffs)
    hyp.p_value = float(p)
    hyp.effect_size = float(np.mean(diffs))
    hyp.status = "validated" if p < alpha else "rejected"
    return hyp
