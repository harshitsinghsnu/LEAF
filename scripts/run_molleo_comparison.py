"""Fair, same-metric comparison: MolLEO-style baseline (leaf.baselines.
molleo_proposer.MolLEOProposer, arXiv:2406.16976) vs. leaf_llm, under
IDENTICAL budget/pool_size/q/acquisition/verifier/splits/seeds -- see
leaf/eval/run_experiment.py METHOD_SPECS["molleo"] for why this isolates
"how the LLM is used in the loop" rather than confounding it with model
choice, budget, or selection strategy.

Runs "molleo" for every (split, seed) in leaf/configs/e2_llm_no_acquisition.yaml
(same budget=40/pool_size=10/q=5 as e2_llm_full.yaml) under the SAME
results/e2_llm_full/ experiment directory as the already-run "leaf_llm"
method, then computes the same bootstrap-CI / Wilcoxon comparison already
used for the agentic-value-add and no-acquisition-ablation comparisons.

Usage:
    python scripts/run_molleo_comparison.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

RUN_MAX_RETRIES = 8
RUN_RETRY_BACKOFF_S = 60.0

import yaml

from leaf.eval.run_experiment import run_one
from leaf.eval import plotting

CONFIG_PATH = Path("leaf/configs/e2_llm_no_acquisition.yaml")  # same budget/pool/q as e2_llm_full.yaml
TARGET_EXPERIMENT = "e2_llm_full"
METHOD = "molleo"
BASELINE_METHOD = "leaf_llm"


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=cfg["splits"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    args = parser.parse_args()

    print(f"=== {TARGET_EXPERIMENT}/{METHOD}: splits={args.splits} seeds={args.seeds} ===")
    for split, seed in itertools.product(args.splits, args.seeds):
        out_dir = Path("results") / TARGET_EXPERIMENT / f"{METHOD}_{split}_seed{seed}"
        if (out_dir / "result.json").exists():
            print(f"  skip (exists) {TARGET_EXPERIMENT}/{METHOD}/{split}/seed{seed}")
            continue
        t0 = time.time()
        # Ollama has intermittently returned a transient HTTP 500 under
        # sustained load during this sweep (confirmed NOT a missing-
        # server/missing-model issue: `ollama serve` answers a bare
        # /api/generate call fine immediately after the failure) --
        # retry the whole (split, seed) run rather than letting one
        # flaky call abort hours of an otherwise-working sweep.
        for attempt in range(RUN_MAX_RETRIES):
            try:
                run_one(str(CONFIG_PATH), TARGET_EXPERIMENT, METHOD, split, seed)
                break
            except RuntimeError as e:
                if attempt == RUN_MAX_RETRIES - 1:
                    raise
                print(f"  retrying {TARGET_EXPERIMENT}/{METHOD}/{split}/seed{seed} "
                      f"after transient error ({attempt + 1}/{RUN_MAX_RETRIES}): {e}")
                time.sleep(RUN_RETRY_BACKOFF_S)
        print(f"  ({time.time() - t0:.1f}s)")

    print("\n=== comparing against baseline:", BASELINE_METHOD, "===")
    all_stats = {}
    for split in args.splits:
        _, stats = plotting.plot_agentic_value_add(TARGET_EXPERIMENT, METHOD, BASELINE_METHOD, split, args.seeds)
        all_stats[split] = stats
        print(f"  {split}: median_diff(molleo - leaf_llm)={stats['median_diff']:.3f} "
              f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}) "
              f"wilcoxon_p={stats['wilcoxon_p']}")

    for split in args.splits:
        plotting.plot_hypervolume_curves(TARGET_EXPERIMENT, [BASELINE_METHOD, METHOD], split, args.seeds)

    out_path = Path("figures") / TARGET_EXPERIMENT / "molleo_comparison_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
