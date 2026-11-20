"""O_GT must never be refit once frozen -- the central Revision-1 bug
(research plan §3.1). Also checks the go/no-go threshold logic (§5.11)."""
import numpy as np
import pytest

from leaf.oracle.ground_truth import GroundTruthOracle
from leaf.oracle.eval_splits import evaluate_split, build_splits
from leaf.fragments.vocabulary import enumerate_vocabulary


def _toy_data(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=n)
    return X, {"potency": y}


def test_frozen_oracle_rejects_refit():
    X, Y = _toy_data()
    oracle = GroundTruthOracle(heads=["potency"]).fit(X, Y).freeze()
    with pytest.raises(RuntimeError):
        oracle.fit(X, Y)


def test_query_returns_point_predictions_only():
    X, Y = _toy_data()
    oracle = GroundTruthOracle(heads=["potency"]).fit(X, Y).freeze()
    out = oracle.query(X[:5])
    assert "potency" in out
    assert out["potency"].shape == (5,)


def test_go_no_go_threshold_flags_low_correlation():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred_good = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    y_pred_bad = np.array([5, 1, 4, 2, 3])
    good = evaluate_split(y_true, y_pred_good, "chemotype_cold", "potency", min_spearman=0.3)
    bad = evaluate_split(y_true, y_pred_bad, "chemotype_cold", "potency", min_spearman=0.3)
    assert good.interpretable is True
    assert bad.interpretable is False


def test_build_splits_never_returns_empty_train_or_test():
    """Regression test: group_split previously could return an empty train
    or test set (int-cast of an empty list defaults to float64 and breaks
    fancy indexing; a single dominant group could also leave train empty
    outright) -- hit in practice on a small (60-fragment) smoke-test
    vocabulary. Both index arrays must be non-empty, integer-typed."""
    vocab = enumerate_vocabulary()[:60]
    splits = build_splits(vocab, seed=0)
    for name, (train_idx, test_idx) in splits.items():
        assert train_idx.dtype.kind == "i", name
        assert test_idx.dtype.kind == "i", name
        assert len(train_idx) > 0, name
        assert len(test_idx) > 0, name
