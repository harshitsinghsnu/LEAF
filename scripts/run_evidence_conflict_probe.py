"""Evidence-conflict probe (external review, following arXiv:2610.00898's
second diagnostic tool): make the injected evidence shown to the Proposer
DISAGREE with the real underlying quality of the candidates, and measure
which one the model's next proposal follows.

Design: build a fixed, real 8-member sample from the real AGILE labeled
set (real fragments, real potency+potency_raw scores). Compute the TRUE
rank (1=best) by summed score. Render the v4_rank-style prompt but with
the rank labels DELIBERATELY INVERTED (fake_rank = 9 - true_rank), so the
member the model is TOLD is "rank 1 (best)" is actually the real worst of
the eight, and vice versa -- everything else about the prompt (phrasing,
objective description, archive identities) is held identical to a
genuine v4_rank call.

Call the real Proposer model (llama3.1:8b, matching every other
experiment in this paper) M times against this fixed, fake-labeled
archive. For each returned candidate, find its nearest neighbor (by
fragment-slot edit distance) among the 8 shown archive members. If the
model is causally using the (now false) injected rank label, its
proposals should cluster near the FAKE-best-labeled member (the real
worst) far more often than the 1-in-8 (12.5%) base rate; if it instead
clusters near the REAL-best member despite the false label, that shows
it is not simply reading the label at face value.

Usage: python scripts/run_evidence_conflict_probe.py [--n_calls 40]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leaf.agents.proposer import LLMProposer
from leaf.data.agile_loader import load_agile_labeled_set
from leaf.fragments.vocabulary import Fragment
from leaf.oracle.eval_splits import build_splits

OUT_DIR = Path("results") / "e_evidence_conflict_probe"


def _edit_distance(a: Fragment, b: Fragment) -> int:
    return sum(1 for slot in ("A", "B", "C") if getattr(a, slot) != getattr(b, slot))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_calls", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frags, labels = load_agile_labeled_set()
    splits = build_splits(frags, seed=0)
    train_idx, _ = splits["random"]

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(train_idx, size=8, replace=False)
    archive_sample = [frags[i] for i in sample_idx]
    score_sample = [np.array([labels["potency"][i], labels["potency_raw"][i]]) for i in sample_idx]

    true_order = sorted(range(8), key=lambda k: -(score_sample[k][0] + score_sample[k][1]))
    true_rank_of = {idx: r + 1 for r, idx in enumerate(true_order)}  # 1 = real best
    fake_rank_of = {idx: 9 - true_rank_of[idx] for idx in range(8)}  # inverted: real best gets fake rank 8

    real_best_idx = true_order[0]
    real_worst_idx = true_order[-1]
    fake_best_idx = real_worst_idx  # the member the model is TOLD is rank 1

    print(f"Real best (true rank 1): {archive_sample[real_best_idx]} score_sum={score_sample[real_best_idx].sum():.3f}")
    print(f"Real worst (true rank 8), shown as FAKE rank 1: {archive_sample[real_worst_idx]} score_sum={score_sample[real_worst_idx].sum():.3f}")

    archive_lines = "\n".join(
        f"- A={f.A} B={f.B} C={f.C} | rank {fake_rank_of[k]} of 8 shown (1 = best)"
        for k, f in enumerate(archive_sample)
    )
    prompt = f"""You are the Proposer Agent in an evolutionary lipid-design loop. Candidates are Ugi
three-component-reaction lipids, each a triple (A, B, C):
A = amine, one of: {json.dumps(sorted(set(f.A for f in frags)))}
B = aldehyde, one of: {json.dumps(sorted(set(f.B for f in frags)))}
C = isocyanide, one of: {json.dumps(sorted(set(f.C for f in frags)))}

Current Pareto archive (sample):
{archive_lines}

Each archive member above is ranked only relative to the others shown here
(rank 1 = most promising of this sample, by real transfection-potency
measurements on two assays). Exact scores are not given, only relative
preference. Prefer proposals you expect to rank at least as well as the
top-ranked archive members shown above.

Propose exactly ONE new candidate via crossover (recombining A/B/C from two archive
parents) or mutation (changing exactly one of A/B/C from a single archive parent, or
a random triple if the archive is empty). The value of A, B, and C in your output
MUST be copied EXACTLY (character for character) from the lists above -- do not
paraphrase or alter the SMILES strings in any way.

Respond with ONLY a single JSON object shaped like:
{{"A": "<one of the A values>", "B": "<one of the B values>", "C": "<one of the C values>",
  "provenance": {{"operator": "crossover" or "mutation", "parents": ["..."]}}}}
"""

    proposer = LLMProposer(seed=args.seed)
    nearest_counts = {"fake_best(real_worst)": 0, "real_best": 0, "other": 0, "tie_or_none": 0}
    raw_results = []
    for i in range(args.n_calls):
        raw = proposer._call_llm(prompt)
        candidate = proposer._parse_one(raw)
        if candidate is None:
            nearest_counts["tie_or_none"] += 1
            raw_results.append(None)
            continue
        dists = [_edit_distance(candidate.fragment, m) for m in archive_sample]
        min_dist = min(dists)
        nearest = [k for k, d in enumerate(dists) if d == min_dist]
        if len(nearest) > 1:
            nearest_counts["tie_or_none"] += 1
        elif nearest[0] == fake_best_idx:
            nearest_counts["fake_best(real_worst)"] += 1
        elif nearest[0] == real_best_idx:
            nearest_counts["real_best"] += 1
        else:
            nearest_counts["other"] += 1
        raw_results.append({"A": candidate.fragment.A, "B": candidate.fragment.B, "C": candidate.fragment.C,
                             "nearest_idx": nearest[0] if len(nearest) == 1 else None, "min_dist": min_dist})
        print(f"call {i}: nearest={nearest if len(nearest)<=2 else 'many'} min_dist={min_dist}", flush=True)

    n_resolved = args.n_calls - nearest_counts["tie_or_none"]
    result = {
        "model": proposer.model,
        "n_calls": args.n_calls,
        "n_resolved": n_resolved,
        "real_best_fragment": str(archive_sample[real_best_idx]),
        "fake_best_fragment_actually_real_worst": str(archive_sample[real_worst_idx]),
        "nearest_counts": nearest_counts,
        "frac_follows_fake_label": nearest_counts["fake_best(real_worst)"] / n_resolved if n_resolved else None,
        "frac_follows_real_quality": nearest_counts["real_best"] / n_resolved if n_resolved else None,
        "base_rate_if_uniform": 1 / 8,
        "raw_results": raw_results,
    }
    out_path = OUT_DIR / f"result_sample_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "raw_results"}, indent=2))


if __name__ == "__main__":
    main()
