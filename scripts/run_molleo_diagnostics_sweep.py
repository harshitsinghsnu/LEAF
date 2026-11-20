"""MolLEO follow-up sweep: diagnostics (duplicate rate / edit distance),
scored-prompt variant (molleo_v3scored), and LLM-mutation extension
(molleo_llm_mutation), all vs. a log_diagnostics-enabled re-run of
leaf_llm, under leaf/configs/e2_molleo_diagnostics.yaml.

Usage:
    python scripts/run_molleo_diagnostics_sweep.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from leaf.eval.run_experiment import run_one
from leaf.eval import plotting

CONFIG_PATH = Path("leaf/configs/e2_molleo_diagnostics.yaml")
TARGET_EXPERIMENT = "e2_molleo_diagnostics"
RUN_MAX_RETRIES = 8
RUN_RETRY_BACKOFF_S = 60.0


def _mean_diagnostics(experiment: str, method: str, split: str, seeds: list[int]) -> dict:
    """Mean duplicate rate / edit distance over all rounds of all seeds --
    the same statistic the poster's '79.1% duplicates' tile reports, but
    computed generically for any method here."""
    rows = []
    for s in seeds:
        p = Path("results") / experiment / f"{method}_{split}_seed{s}" / "rounds.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    out = {}
    for col in ("frac_duplicate_of_archive", "mean_min_edit_distance", "frac_selected_duplicate_of_archive"):
        if col in df and df[col].notna().any():
            out[col] = float(df[col].mean(skipna=True))
    return out


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=cfg["methods"])
    parser.add_argument("--splits", nargs="+", default=cfg["splits"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    args = parser.parse_args()

    print(f"=== {TARGET_EXPERIMENT}: methods={args.methods} splits={args.splits} seeds={args.seeds} ===")
    for method, split, seed in itertools.product(args.methods, args.splits, args.seeds):
        out_dir = Path("results") / TARGET_EXPERIMENT / f"{method}_{split}_seed{seed}"
        if (out_dir / "result.json").exists():
            print(f"  skip (exists) {TARGET_EXPERIMENT}/{method}/{split}/seed{seed}")
            continue
        t0 = time.time()
        for attempt in range(RUN_MAX_RETRIES):
            try:
                run_one(str(CONFIG_PATH), TARGET_EXPERIMENT, method, split, seed)
                break
            except RuntimeError as e:
                if attempt == RUN_MAX_RETRIES - 1:
                    raise
                print(f"  retrying {TARGET_EXPERIMENT}/{method}/{split}/seed{seed} "
                      f"after transient error ({attempt + 1}/{RUN_MAX_RETRIES}): {e}")
                time.sleep(RUN_RETRY_BACKOFF_S)
        print(f"  ({time.time() - t0:.1f}s)")

    print("\n=== hypervolume comparisons (vs. leaf_llm) ===")
    all_stats = {}
    for split in args.splits:
        for method in args.methods:
            if method == "leaf_llm":
                continue
            _, stats = plotting.plot_agentic_value_add(TARGET_EXPERIMENT, method, "leaf_llm", split, args.seeds)
            all_stats[f"{method}_vs_leaf_llm_{split}"] = stats
            print(f"  {method} vs leaf_llm, {split}: median_diff={stats['median_diff']:.3f} "
                  f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}) wilcoxon_p={stats['wilcoxon_p']}")
    # molleo_v3scored / molleo_llm_mutation vs. the paper-faithful molleo baseline
    for split in args.splits:
        for method in ("molleo_v3scored", "molleo_llm_mutation"):
            if method not in args.methods or "molleo" not in args.methods:
                continue
            _, stats = plotting.plot_agentic_value_add(TARGET_EXPERIMENT, method, "molleo", split, args.seeds)
            all_stats[f"{method}_vs_molleo_{split}"] = stats
            print(f"  {method} vs molleo, {split}: median_diff={stats['median_diff']:.3f} "
                  f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}) wilcoxon_p={stats['wilcoxon_p']}")

    print("\n=== duplicate-rate / edit-distance diagnostics ===")
    diag = {}
    for method, split in itertools.product(args.methods, args.splits):
        d = _mean_diagnostics(TARGET_EXPERIMENT, method, split, args.seeds)
        diag[f"{method}_{split}"] = d
        print(f"  {method}/{split}: {d}")

    out_path = Path("figures") / TARGET_EXPERIMENT / "molleo_diagnostics_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hv_comparisons": all_stats, "diagnostics": diag}, indent=2))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
