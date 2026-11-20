"""Research plan §5.10: total queries issued (acquisition rounds +
Hypothesis Agent counterfactual pairs) must never exceed the budget B."""
import numpy as np

from leaf.fragments.embed import FragmentEmbedder
from leaf.oracle.ground_truth import GroundTruthOracle
from leaf.orchestrator.loop import LoopConfig, run_loop

# This test only exercises budget-accounting mechanics, not real chemistry
# or real labels, so a synthetic toy oracle is fine here -- but its heads
# must be internally consistent with LoopConfig.heads/objective_names
# (both explicitly set to match, rather than relying on defaults, since
# the real pipeline's default heads are ["potency", "potency_raw"]).
TOY_HEADS = ["potency", "toxicity"]


def _fit_toy_oracle(embedder, seed=0):
    rng = np.random.default_rng(seed)
    from leaf.fragments.vocabulary import enumerate_vocabulary
    vocab = enumerate_vocabulary()[:150]
    X = embedder.embed_batch(vocab)
    Y = {"potency": rng.normal(size=X.shape[0]), "toxicity": rng.normal(size=X.shape[0])}
    return GroundTruthOracle(heads=TOY_HEADS).fit(X, Y).freeze()


def test_budget_never_exceeded_with_hypothesis_agent():
    embedder = FragmentEmbedder()
    oracle = _fit_toy_oracle(embedder)
    budget = 30
    cfg = LoopConfig(budget=budget, pool_size=20, q=5, hypothesis_every_k=2,
                      max_counterfactual_pairs=4, use_hypothesis=True, seed=1,
                      heads=TOY_HEADS, signs={"potency": 1.0, "toxicity": -1.0})
    rng = np.random.default_rng(2)
    cal = (rng.uniform(0, 1, size=30), (rng.uniform(0, 1, size=30) > 0.5).astype(float))
    result = run_loop(cfg, oracle, embedder, TOY_HEADS,
                       reference_point=np.array([-5.0, -5.0]), calibration_pool=cal)
    assert result["total_queries_issued"] <= budget


def test_archive_hypervolume_monotonic():
    embedder = FragmentEmbedder()
    oracle = _fit_toy_oracle(embedder)
    cfg = LoopConfig(budget=40, pool_size=15, q=5, use_hypothesis=False, seed=3,
                      heads=TOY_HEADS, signs={"potency": 1.0, "toxicity": -1.0})
    rng = np.random.default_rng(4)
    cal = (rng.uniform(0, 1, size=30), (rng.uniform(0, 1, size=30) > 0.5).astype(float))
    result = run_loop(cfg, oracle, embedder, TOY_HEADS,
                       reference_point=np.array([-5.0, -5.0]), calibration_pool=cal)
    hv_history = result["archive"].hypervolume_history
    assert all(b >= a - 1e-9 for a, b in zip(hv_history, hv_history[1:]))
