"""E2 proposer-only-vs-full-loop ablation (cf. MolLEO arXiv:2406.16976's
direct-query-vs-evolutionary-loop ablation).

Runs method "leaf_llm_no_acquisition" (LoopConfig.use_acquisition=False --
the first q pool survivors are queried directly, no UCB scoring) for every
(split, seed) in leaf/configs/e2_llm_no_acquisition.yaml, under the SAME
experiment directory ("e2_llm_full") as the already-run "leaf_llm" method,
so plot_agentic_value_add can compare the two directly without re-running
leaf_llm.

Usage:
    python scripts/run_e2_no_acquisition_ablation.py
    python scripts/run_e2_no_acquisition_ablation.py --splits random --seeds 0 1
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import yaml

from leaf.eval.run_experiment import run_one
from leaf.eval import plotting

CONFIG_PATH = Path("leaf/configs/e2_llm_no_acquisition.yaml")
TARGET_EXPERIMENT = "e2_llm_full"  # shares a dir with the already-run leaf_llm method
METHOD = "leaf_llm_no_acquisition"
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
        run_one(str(CONFIG_PATH), TARGET_EXPERIMENT, METHOD, split, seed)
        print(f"  ({time.time() - t0:.1f}s)")

    print("\n=== comparing against baseline:", BASELINE_METHOD, "===")
    all_stats = {}
    for split in args.splits:
        _, stats = plotting.plot_agentic_value_add(TARGET_EXPERIMENT, BASELINE_METHOD, METHOD, split, args.seeds)
        all_stats[split] = stats
        print(f"  {split}: median_diff(leaf_llm - no_acquisition)={stats['median_diff']:.3f} "
              f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}) "
              f"wilcoxon_p={stats['wilcoxon_p']}")

    out_path = Path("figures") / TARGET_EXPERIMENT / "no_acquisition_ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
