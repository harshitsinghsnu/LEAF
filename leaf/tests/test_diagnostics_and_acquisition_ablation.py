"""Regression tests for the mechanistic-diagnostics logging and the
acquisition-disabled ablation added to isolate proposal quality from
selection quality."""
import numpy as np

from leaf.fragments.embed import FragmentEmbedder
from leaf.fragments.vocabulary import Fragment
from leaf.oracle.ground_truth import GroundTruthOracle
from leaf.orchestrator.loop import LoopConfig, run_loop, _fragment_edit_distance, _pool_diagnostics
from leaf.agents.proposer import ProposedCandidate


def test_fragment_edit_distance():
    a = Fragment("A1", "B1", "C1")
    b = Fragment("A1", "B2", "C2")
    assert _fragment_edit_distance(a, a) == 0
    assert _fragment_edit_distance(a, b) == 2


def test_pool_diagnostics_detects_duplicates_and_operator_mix():
    archive = [Fragment("A1", "B1", "C1")]
    pool = [
        ProposedCandidate(Fragment("A1", "B1", "C1"), operator="mutation"),  # exact duplicate
        ProposedCandidate(Fragment("A2", "B1", "C1"), operator="crossover"),  # 1 slot different
    ]
    frac_crossover, frac_dup, mean_dist = _pool_diagnostics(pool, archive)
    assert frac_crossover == 0.5
    assert frac_dup == 0.5
    assert mean_dist == 0.5  # (0 + 1) / 2


def _fit_toy_oracle(embedder, seed=0):
    rng = np.random.default_rng(seed)
    from leaf.fragments.vocabulary import enumerate_vocabulary
    vocab = enumerate_vocabulary()[:150]
    X = embedder.embed_batch(vocab)
    Y = {"potency": rng.normal(size=X.shape[0]), "potency_raw": rng.normal(size=X.shape[0])}
    return GroundTruthOracle(heads=["potency", "potency_raw"]).fit(X, Y).freeze()


def test_acquisition_disabled_still_respects_budget_and_q():
    embedder = FragmentEmbedder()
    oracle = _fit_toy_oracle(embedder)
    cfg = LoopConfig(budget=20, pool_size=10, q=5, use_acquisition=False,
                      use_hypothesis=False, log_diagnostics=True, seed=1)
    rng = np.random.default_rng(2)
    cal = (rng.uniform(0, 1, size=20), (rng.uniform(0, 1, size=20) > 0.5).astype(float))
    result = run_loop(cfg, oracle, embedder, ["potency", "potency_raw"],
                       reference_point=np.array([-5.0, -5.0]), calibration_pool=cal)
    assert result["total_queries_issued"] <= 20
    # Diagnostics were requested -- confirm they were actually populated.
    assert any(log.frac_crossover is not None for log in result["logs"])
