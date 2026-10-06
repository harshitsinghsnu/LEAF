"""Research plan §5.2: n=3 Wilcoxon cannot reach p<0.05; significance.py
must not claim reliability below MIN_SEEDS_FOR_SIGNIFICANCE."""
import numpy as np

from leaf.eval.significance import paired_significance, MIN_SEEDS_FOR_SIGNIFICANCE


def test_three_seeds_flagged_unreliable():
    diffs = np.array([0.5, 0.6, 0.55])
    result = paired_significance(diffs)
    assert result.n_seeds == 3
    assert result.wilcoxon_reliable is False


def test_eight_seeds_flagged_reliable_when_consistent():
    diffs = np.array([0.5, 0.6, 0.55, 0.52, 0.61, 0.58, 0.49, 0.53])
    result = paired_significance(diffs)
    assert result.n_seeds == MIN_SEEDS_FOR_SIGNIFICANCE
    assert result.wilcoxon_reliable is True
    assert result.wilcoxon_p is not None
