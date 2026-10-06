"""Find the real best-found candidate per method (LLM scored-prompt, GA,
random search) on the random split, seed 1 (highest-HV LLM seed in the
objective-scored-prompt check), by refitting the exact same frozen O_GT
used in that experiment and scoring every queried triple from queries.csv.
No new search is run -- these are the real queried candidates already on
disk; only their potency is (re)computed here since it isn't stored
per-candidate in queries.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np

from leaf.data.agile_loader import load_agile_labeled_set
from leaf.eval.run_experiment import _fit_real_oracle_for_split
from leaf.fragments.embed import FragmentEmbedder
from leaf.fragments.vocabulary import Fragment, assemble_lipid
from leaf.oracle.eval_splits import build_splits

EXPERIMENT_DIR = Path("results/e2_llm_v3scored_check")
SPLIT = "random"
SEED = 1
HEADS = ["potency", "potency_raw"]

embedder = FragmentEmbedder()
oracle = _fit_real_oracle_for_split(embedder, SPLIT, heads=HEADS)


def score_method(method_dirname: str):
    qpath = EXPERIMENT_DIR / f"{method_dirname}_{SPLIT}_seed{SEED}" / "queries.csv"
    rows = list(csv.DictReader(open(qpath)))
    frags = [Fragment(A=r["A"], B=r["B"], C=r["C"]) for r in rows]
    X = embedder.embed_batch(frags)
    preds = oracle.query(X)
    sums = preds["potency"] + preds["potency_raw"]
    best_i = int(np.argmax(sums))
    best_frag = frags[best_i]
    smiles = assemble_lipid(best_frag)
    return {
        "method": method_dirname,
        "A": best_frag.A, "B": best_frag.B, "C": best_frag.C,
        "smiles": smiles,
        "pred_potency": float(preds["potency"][best_i]),
        "pred_potency_raw": float(preds["potency_raw"][best_i]),
        "pred_sum": float(sums[best_i]),
        "n_queried": len(frags),
    }


results = {}
for method in ["leaf_llm_v3scored", "static_ga", "random_search"]:
    results[method] = score_method(method)
    print(json.dumps(results[method], indent=2))

Path("results/e_best_lipid_structures.json").write_text(json.dumps(results, indent=2))
