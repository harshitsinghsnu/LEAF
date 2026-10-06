"""Post-hoc fragment-importance analysis, run INDEPENDENTLY of the
(inactive) Hypothesis Ledger mechanism. This does not use counterfactual
pairs, Integrated Gradients, or KernelSHAP as the Ledger's design
specified -- it is a much simpler, fully transparent diagnostic: for each
individual A/B/C fragment value appearing anywhere in the frozen O_GT's
random-split training partition, compute the mean predicted potency+
potency_raw across every full-vocabulary candidate that contains it, and
report the top and bottom fragments by this mean. This is retrospective
inspection of what the already-fitted GP associates with higher/lower
potency -- not a claim that the search loop's Proposer used or discovered
this association, and not a replacement for the Ledger's intended
matched-counterfactual-pair statistical test.
"""
import json
from pathlib import Path

import numpy as np

from leaf.data.agile_loader import load_agile_labeled_set
from leaf.eval.run_experiment import _fit_real_oracle_for_split
from leaf.fragments.embed import FragmentEmbedder
from leaf.fragments.vocabulary import enumerate_vocabulary

SPLIT = "random"
HEADS = ["potency", "potency_raw"]

embedder = FragmentEmbedder()
oracle = _fit_real_oracle_for_split(embedder, SPLIT, heads=HEADS)

all_frags = enumerate_vocabulary()
print(f"Full vocabulary size: {len(all_frags)}")

X = embedder.embed_batch(all_frags)
preds = oracle.query(X)
combined = preds["potency"] + preds["potency_raw"]

from collections import defaultdict
slot_values = defaultdict(lambda: defaultdict(list))
for frag, score in zip(all_frags, combined):
    slot_values["A"][frag.A].append(score)
    slot_values["B"][frag.B].append(score)
    slot_values["C"][frag.C].append(score)

out = {}
for slot in ("A", "B", "C"):
    means = {v: float(np.mean(scores)) for v, scores in slot_values[slot].items() if len(scores) >= 20}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    out[slot] = {
        "n_values_with_ge20_candidates": len(ranked),
        "top5": ranked[:5],
        "bottom5": ranked[-5:],
        "overall_mean": float(np.mean(combined)),
        "overall_std": float(np.std(combined)),
    }
    print(f"\nSlot {slot}: {len(ranked)} values with >=20 candidates")
    print("  top5:", ranked[:5])
    print("  bottom5:", ranked[-5:])

Path("results/e_fragment_importance_posthoc.json").write_text(json.dumps(out, indent=2))
