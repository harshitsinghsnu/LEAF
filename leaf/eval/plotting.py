"""Generate every figure required by the research plan's §8 reporting
checklist, from the results/ directory tree written by run_experiment.py.

All figures are saved as both .png (quick viewing) and .pdf (camera-ready)
under figures/<experiment>/. Nothing is only shown interactively -- a
paper figure that only exists in a notebook cell is not reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leaf.eval.significance import paired_significance

RESULTS_ROOT = Path("results")
FIGURES_ROOT = Path("figures")


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_rounds(experiment: str, method: str, split: str, seed: int) -> pd.DataFrame | None:
    p = RESULTS_ROOT / experiment / f"{method}_{split}_seed{seed}" / "rounds.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_summary(experiment: str, method: str, split: str, seed: int) -> dict | None:
    p = RESULTS_ROOT / experiment / f"{method}_{split}_seed{seed}" / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def plot_hypervolume_curves(experiment: str, methods: list[str], split: str, seeds: list[int]) -> Path:
    """Mean +/- std hypervolume vs. round, one line per method (E1/E2)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        runs = [_load_rounds(experiment, method, split, s) for s in seeds]
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        max_len = max(len(r) for r in runs)
        padded = np.full((len(runs), max_len), np.nan)
        for i, r in enumerate(runs):
            padded[i, :len(r)] = r["archive_hypervolume"].values
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Pareto hypervolume")
    ax.set_title(f"Hypervolume vs. round ({experiment}, split={split}, n={len(seeds)} seeds)")
    ax.legend()
    out = FIGURES_ROOT / experiment / f"hypervolume_curves_{split}"
    _save(fig, out)
    return out.with_suffix(".png")


def plot_agentic_value_add(experiment: str, method_a: str, method_b: str, split: str,
                            seeds: list[int]) -> tuple[Path, dict]:
    """Bootstrap-CI bar for the primary claim: HV(method_a) - HV(method_b),
    per §5.2/§7.2 -- bootstrap CI is the primary statistic, not a bare
    Wilcoxon p-value at n<8."""
    hv_a, hv_b = [], []
    for s in seeds:
        sa = _load_summary(experiment, method_a, split, s)
        sb = _load_summary(experiment, method_b, split, s)
        if sa is None or sb is None:
            continue
        hv_a.append(sa["final_hypervolume"])
        hv_b.append(sb["final_hypervolume"])
    diffs = np.array(hv_a) - np.array(hv_b)
    sig = paired_significance(diffs)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([f"{method_a} - {method_b}"], [sig.median_diff],
           yerr=[[sig.median_diff - sig.ci_low], [sig.ci_high - sig.median_diff]], capsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    title = f"n={sig.n_seeds} seeds"
    if sig.wilcoxon_p is not None:
        title += f", Wilcoxon p={sig.wilcoxon_p:.3f}" + ("" if sig.wilcoxon_reliable else " (UNRELIABLE, n<8)")
    ax.set_title(title)
    ax.set_ylabel("Median paired hypervolume difference (95% bootstrap CI)")
    out = FIGURES_ROOT / experiment / f"agentic_value_add_{split}"
    _save(fig, out)
    stats = {
        "method_a": method_a, "method_b": method_b, "split": split,
        "n_seeds": sig.n_seeds, "median_diff": sig.median_diff,
        "ci_low": sig.ci_low, "ci_high": sig.ci_high,
        "wilcoxon_p": sig.wilcoxon_p, "wilcoxon_reliable": sig.wilcoxon_reliable,
        "hv_a_per_seed": hv_a, "hv_b_per_seed": hv_b,
    }
    # Previously this was only ever printed to stdout by
    # run_all_experiments.py -- the PRIMARY paper claim (H1) living only in
    # terminal scrollback, with no artifact on disk, was a real gap given
    # the reproducibility standard the rest of the pipeline holds itself
    # to (manifests, checksums, etc.). Persist it alongside the figure.
    stats_path = FIGURES_ROOT / experiment / f"agentic_value_add_{split}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    return out.with_suffix(".png"), stats


def plot_coverage_vs_round(experiment: str, method: str, split: str, seed: int) -> Path:
    """Per-round conformal coverage vs. the nominal target, disclosing any
    degradation under the loop's covariate shift (§5.3) -- never collapsed
    to a single number."""
    df = _load_rounds(experiment, method, split, seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    if df is not None and "empirical_coverage" in df:
        ax.plot(df["round"], df["empirical_coverage"], label="empirical coverage", marker="o")
        if "target_coverage" in df:
            ax.axhline(df["target_coverage"].iloc[0], color="red", linestyle="--", label="target")
    ax.set_xlabel("Round")
    ax.set_ylabel("Conformal coverage")
    ax.set_ylim(0, 1)
    ax.set_title(f"Coverage vs. round ({method}, {split}, seed={seed})")
    ax.legend()
    out = FIGURES_ROOT / experiment / f"coverage_vs_round_{method}_{split}_seed{seed}"
    _save(fig, out)
    return out.with_suffix(".png")


def plot_inter_agent_agreement(experiment: str, method: str, split: str, seeds: list[int]) -> Path:
    """First-class inter-agent agreement metric (§5.4) -- near-1.0 warns
    the abstention set may be too small to test H3 meaningfully."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in seeds:
        df = _load_rounds(experiment, method, split, s)
        if df is not None and "inter_agent_agreement" in df:
            ax.plot(df["round"], df["inter_agent_agreement"], alpha=0.6, label=f"seed {s}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Inter-agent agreement")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Verifier inter-agent agreement ({method}, {split})")
    ax.legend(fontsize=7)
    out = FIGURES_ROOT / experiment / f"inter_agent_agreement_{method}_{split}"
    _save(fig, out)
    return out.with_suffix(".png")


def plot_seed_sensitivity(experiment: str, methods: list[str], split: str, seeds: list[int]) -> Path:
    """E6-style box/strip plot of final hypervolume across seeds, per method
    -- the reportable variance budget for RQ7."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data, labels = [], []
    for method in methods:
        vals = []
        for s in seeds:
            summary = _load_summary(experiment, method, split, s)
            if summary is not None:
                vals.append(summary["final_hypervolume"])
        if vals:
            data.append(vals)
            labels.append(method)
    if data:
        try:
            ax.boxplot(data, tick_labels=labels)   # matplotlib >= 3.9
        except TypeError:
            ax.boxplot(data, labels=labels)         # matplotlib < 3.9
    ax.set_ylabel("Final hypervolume")
    ax.set_title(f"Seed sensitivity ({experiment}, split={split}, n={len(seeds)} seeds)")
    out = FIGURES_ROOT / experiment / f"seed_sensitivity_{split}"
    _save(fig, out)
    return out.with_suffix(".png")


def write_summary_table(experiment: str, methods: list[str], split: str, seeds: list[int]) -> Path:
    """CSV table of per-method mean/std final hypervolume and hypothesis
    precision -- the raw numbers behind every figure, for the paper's
    tables and for independent re-analysis."""
    rows = []
    for method in methods:
        hvs, precisions = [], []
        for s in seeds:
            summary = _load_summary(experiment, method, split, s)
            if summary is None:
                continue
            hvs.append(summary["final_hypervolume"])
            if not np.isnan(summary.get("hypothesis_precision", np.nan)):
                precisions.append(summary["hypothesis_precision"])
        rows.append({
            "method": method, "split": split, "n_seeds": len(hvs),
            "hv_mean": np.mean(hvs) if hvs else np.nan,
            "hv_std": np.std(hvs) if hvs else np.nan,
            "hypothesis_precision_mean": np.mean(precisions) if precisions else np.nan,
        })
    out = FIGURES_ROOT / experiment / f"summary_table_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
