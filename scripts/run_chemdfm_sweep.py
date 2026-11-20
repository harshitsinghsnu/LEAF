"""Chemistry-specialized LLM baseline sweep: ChemDFM-v2.0-14B and
ChemDFM-R-14B vs. leaf_llm (general-purpose llama3.1:8b) and molleo
(MolLEO-style), under leaf/configs/e2_chemdfm.yaml. Reuses leaf_llm/
molleo results already logged under the shared e2_molleo_diagnostics
experiment dir -- only leaf_chemdfm_v2/leaf_chemdfm_r are actually run
here.

Usage:
    python scripts/run_chemdfm_sweep.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

os.environ.setdefault("HF_HOME", "D:/hf_cache")

from leaf.eval.run_experiment import run_one
from leaf.eval import plotting

CONFIG_PATH = Path("leaf/configs/e2_chemdfm.yaml")
TARGET_EXPERIMENT = "e2_molleo_diagnostics"
RUN_MAX_RETRIES = 4
RUN_RETRY_BACKOFF_S = 60.0


def _mean_diagnostics(experiment: str, method: str, split: str, seeds: list[int]) -> dict:
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

    print(f"=== {TARGET_EXPERIMENT} (chemdfm): methods={args.methods} splits={args.splits} seeds={args.seeds} ===")
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
                      f"after error ({attempt + 1}/{RUN_MAX_RETRIES}): {e}")
                time.sleep(RUN_RETRY_BACKOFF_S)
        print(f"  ({time.time() - t0:.1f}s)")

    print("\n=== hypervolume comparisons (vs leaf_llm, vs molleo) ===")
    all_stats = {}
    for split in args.splits:
        for method in args.methods:
            for baseline in ("leaf_llm", "molleo"):
                base_dir = Path("results") / TARGET_EXPERIMENT / f"{baseline}_{split}_seed0"
                if not (base_dir / "result.json").exists():
                    continue
                _, stats = plotting.plot_agentic_value_add(TARGET_EXPERIMENT, method, baseline, split, args.seeds)
                all_stats[f"{method}_vs_{baseline}_{split}"] = stats
                print(f"  {method} vs {baseline}, {split}: median_diff={stats['median_diff']:.3f} "
                      f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}) wilcoxon_p={stats['wilcoxon_p']}")

    print("\n=== duplicate-rate / edit-distance diagnostics ===")
    diag = {}
    for method, split in itertools.product(args.methods, args.splits):
        d = _mean_diagnostics(TARGET_EXPERIMENT, method, split, args.seeds)
        diag[f"{method}_{split}"] = d
        print(f"  {method}/{split}: {d}")

    out_path = Path("figures") / TARGET_EXPERIMENT / "chemdfm_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hv_comparisons": all_stats, "diagnostics": diag}, indent=2))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
