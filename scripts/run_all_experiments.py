"""Run the full experiment suite (E1-E6; E8 conditional) end to end:
sweep every (method, split, seed) triple each config specifies, save all
raw results under results/, then generate every figure/table required by
the research plan's §8 reporting checklist under figures/.

This is the single command a reviewer (or you, in six months) runs to
reproduce every number in the paper:

    conda activate leaf
    python scripts/run_all_experiments.py --experiments e1 e2 e3 e4 e6

Omit --experiments to run everything except E5 and E8, which have
preconditions (E5 requires the leakage probe to have been run and
disclosed; E8 requires the conditional LUMI-lab data release) and are
run individually via run_experiment.py once those preconditions are met.

IMPORTANT: the placeholder O_GT fit inside run_experiment.py
(_fit_placeholder_oracle_for_split) is a synthetic-label smoke test. Do not treat
numbers produced this way as reportable results -- see README.md and
research plan §3.1/§5.11. Pass --oracle path/to/frozen_oracle.pkl once the
real harmonized-data oracle (data/processed/, §5.8) has been fit, frozen,
and validated (E1) to run reportable experiments.
"""
from __future__ import annotations

import argparse
import itertools
import runpy
import sys
import time
from pathlib import Path

import yaml

from leaf.eval.run_experiment import run_one, NOT_YET_WIRED
from leaf.eval import plotting

CONFIG_DIR = Path("leaf/configs")

EXPERIMENT_SPECS = {
    "e2": {
        "config": CONFIG_DIR / "e2_agentic_value_add.yaml",
        "methods": None,      # read from config
        "splits": None,
    },
    "e3": {
        "config": CONFIG_DIR / "e3_hypothesis_feedback_ablation.yaml",
        "methods": None,
        "splits": ["chemotype_cold"],
    },
    "e4": {
        "config": CONFIG_DIR / "e4_verifier_calibration_ablation.yaml",
        "methods": None,
        "splits": ["random"],
    },
    "e6": {
        "config": CONFIG_DIR / "e6_prompt_temperature_sensitivity.yaml",
        "methods": ["leaf"],
        "splits": ["random"],
    },
}


def _load(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def sweep(experiment: str, oracle_path: str | None, dry_run: bool = False) -> None:
    spec = EXPERIMENT_SPECS[experiment]
    cfg = _load(spec["config"])
    methods = spec["methods"] or cfg.get("methods", ["leaf"])
    splits = spec["splits"] or cfg.get("splits", [cfg.get("split", "random")])
    seeds = cfg.get("seeds", [0])

    # E2's whole point is comparing methods ACROSS splits -- that only
    # means something if each split gets its own O_GT (fit on that
    # split's real train partition only, so a harder split is genuinely a
    # harder/less-informed ground truth). The frozen, full-1200-point
    # --oracle is fit on everything regardless of split, so passing it to
    # E2 silently makes split a no-op (found empirically: byte-identical
    # results across random/scaffold_cold/chemotype_cold). --oracle
    # remains valid for E3/E4/E6, which each use one fixed split.
    effective_oracle_path = oracle_path
    if experiment == "e2" and oracle_path is not None:
        print("  NOTE: ignoring --oracle for e2 -- each split fits its own O_GT on that "
              "split's real train partition (required for split to be meaningful here); "
              "the frozen full-data oracle is used for e3/e4/e6 instead.")
        effective_oracle_path = None

    print(f"=== {experiment}: methods={methods} splits={splits} n_seeds={len(seeds)} ===")
    skipped_methods = set()
    for method, split, seed in itertools.product(methods, splits, seeds):
        if method in NOT_YET_WIRED:
            if method not in skipped_methods:
                print(f"  SKIPPING method '{method}': {NOT_YET_WIRED[method]}")
                skipped_methods.add(method)
            continue
        if dry_run:
            print(f"  [dry-run] would run {experiment}/{method}/{split}/seed{seed}")
            continue
        t0 = time.time()
        run_one(str(spec["config"]), experiment, method, split, seed, oracle_path=effective_oracle_path)
        print(f"  ({time.time() - t0:.1f}s)")

    if dry_run:
        return

    methods = [m for m in methods if m not in skipped_methods]

    # Generate figures/tables for this experiment.
    if len(methods) >= 2:
        for split in splits:
            plotting.plot_hypervolume_curves(experiment, methods, split, seeds)
            plotting.plot_seed_sensitivity(experiment, methods, split, seeds)
            plotting.write_summary_table(experiment, methods, split, seeds)
            if experiment == "e2" and "leaf" in methods and "static_ga" in methods:
                _, stats = plotting.plot_agentic_value_add(experiment, "leaf", "static_ga", split, seeds)
                print(f"  agentic value-add ({split}): {stats}")
    for method in methods:
        for split in splits:
            for seed in seeds[:1]:
                plotting.plot_coverage_vs_round(experiment, method, split, seed)
            plotting.plot_inter_agent_agreement(experiment, method, split, seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["e1", "e2", "e3", "e4", "e6"],
                         choices=["e1"] + list(EXPERIMENT_SPECS))
    parser.add_argument("--oracle", default=None,
                         help="path to a frozen O_GT (GroundTruthOracle.save output). "
                              "Without this, a synthetic-label smoke-test oracle is used "
                              "and results are NOT reportable (see module docstring).")
    parser.add_argument("--dry-run", action="store_true", help="print the planned runs without executing")
    args = parser.parse_args()

    if args.oracle is None:
        print("WARNING: no --oracle given. Running with the synthetic-label placeholder "
              "O_GT (leaf/eval/run_experiment.py:_fit_placeholder_oracle_for_split). These results "
              "are a pipeline smoke test only and must not be reported in the paper.")

    for experiment in args.experiments:
        if experiment == "e1":
            # E1 validates O_GT itself (RQ5/RQ6) -- it does not run the
            # round loop, so it has its own script rather than being
            # forced through sweep()/run_one().
            if args.dry_run:
                print("=== e1: would run scripts/run_e1_oracle_validation.py ===")
                continue
            sys.argv = ["run_e1_oracle_validation.py", "--config", str(CONFIG_DIR / "e1_oracle_validation.yaml")]
            runpy.run_path("scripts/run_e1_oracle_validation.py", run_name="__main__")
            continue
        sweep(experiment, args.oracle, dry_run=args.dry_run)

    print("\nDone. Raw results: results/<experiment>/  Figures/tables: figures/<experiment>/")


if __name__ == "__main__":
    main()
