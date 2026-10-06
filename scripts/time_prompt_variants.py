"""Measure real per-candidate latency for each Llama-8B prompt variant
(v1 default, v3_scored, v4_rank, v5_cot, v6_shuffled) using the live
proposer against a small synthetic archive, for the Table-6-style cost
accounting. 8 calls per variant, local Ollama, llama3.1:8b.
"""
import json
import time
from pathlib import Path

import numpy as np

from leaf.agents.proposer import LLMProposer
from leaf.data.agile_loader import load_agile_labeled_set
from leaf.oracle.eval_splits import build_splits

frags, labels = load_agile_labeled_set()
splits = build_splits(frags, seed=0)
train_idx, _ = splits["random"]
rng = np.random.default_rng(0)
sample_idx = rng.choice(train_idx, size=8, replace=False)
archive = [frags[i] for i in sample_idx]
archive_scores = [np.array([labels["potency"][i], labels["potency_raw"][i]]) for i in sample_idx]

N_CALLS = 8
out = {}
for version in ["v1", "v3_scored", "v4_rank", "v5_cot", "v6_shuffled"]:
    proposer = LLMProposer(seed=0, prompt_version=version)
    times = []
    for i in range(N_CALLS):
        t0 = time.time()
        pool = proposer.propose_pool(archive, 1,
                                      archive_scores=archive_scores if version != "v1" else None)
        times.append(time.time() - t0)
    out[version] = {"mean_seconds": float(np.mean(times)), "times": times}
    print(version, out[version])

Path("results/e_prompt_variant_latency.json").write_text(json.dumps(out, indent=2))
