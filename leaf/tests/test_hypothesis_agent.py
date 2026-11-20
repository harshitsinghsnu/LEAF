"""Hypothesis Ledger: only written as validated if p < alpha, and too-few
pairs (n<6) must not silently pass (same power argument as §5.2)."""
import numpy as np

from leaf.agents.hypothesis import Hypothesis, HypothesisLedger
# Aliased on import: pytest collects any module-level `test_*` callable,
# including an imported one, so importing the function under test as
# `test_hypothesis` made pytest try to run it directly as a test.
from leaf.agents.hypothesis import test_hypothesis as run_hypothesis_test


def test_insufficient_pairs_rejected():
    hyp = Hypothesis(text="t", fragment_slot="halogenation", fragment_value="Br", property_head="potency")
    result = run_hypothesis_test(hyp, np.array([1.0, 2.0]), np.array([1.1, 2.2]))
    assert result.status == "rejected"


def test_significant_shift_validated():
    hyp = Hypothesis(text="t", fragment_slot="halogenation", fragment_value="Br", property_head="potency")
    rng = np.random.default_rng(0)
    control = rng.normal(0, 0.1, size=20)
    treatment = control + 2.0  # large, consistent shift
    result = run_hypothesis_test(hyp, control, treatment)
    assert result.status == "validated"
    assert result.p_value < 0.05


def test_ledger_precision():
    ledger = HypothesisLedger()
    ledger.entries.append(Hypothesis("a", "head", "x", "potency", status="validated", p_value=0.01))
    ledger.entries.append(Hypothesis("b", "head", "y", "potency", status="rejected", p_value=0.5))
    assert ledger.precision() == 0.5
