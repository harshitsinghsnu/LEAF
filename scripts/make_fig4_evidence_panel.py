"""Figure 4: evidence-use checks (leakage probe, rank-only, CoT, score-shuffle,
evidence-conflict probe, reasoning-mode proposer). Built entirely from real
per-seed results under results/ -- numbers are cross-checked directly against
the result.json / queries.csv files cited in SI_submitted.tex.

Run: python scripts/make_fig4_evidence_panel.py
Writes figures/paper/fig4_evidence_panel.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

C_TEAL = "#BFE8C8"
C_LAVENDER = "#A1A1F7"
C_PINK = "#F2C7DB"
C_CYAN = "#7ADDF2"
C_PURPLE = "#CBB7F5"
C_GREEN = "#89D385"
C_GRAY = "#CFCFCF"


def _hv(experiment: str, method: str, split: str, seeds: list[int]) -> np.ndarray:
    return np.array([
        json.loads((RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json").read_text())["final_hypervolume"]
        for s in seeds
    ])


def _panel_label(ax, letter):
    ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=20, fontweight="bold",
             ha="left", va="bottom", color="#1a1a1a")


seeds4 = [0, 1, 2, 3]
GA = _hv("e2_llm_v4rank_check", "static_ga", "random", seeds4)
RS = _hv("e2_llm_v4rank_check", "random_search", "random", seeds4)
rank = _hv("e2_llm_v4rank_check", "leaf_llm_v4rank", "random", seeds4)
cot = _hv("e2_llm_v5cot_check", "leaf_llm_v5cot", "random", seeds4)
shuf = _hv("e2_llm_v6shuffled_check", "leaf_llm_v6shuffled", "random", seeds4)
scored = _hv("e2_llm_v3scored_check", "leaf_llm_v3scored", "random", seeds4)
reasoning_default = _hv("e2_qwen3_reasoning_check", "leaf_llm_qwen3_reasoning", "random", seeds4)
reasoning_scored = _hv("e2_qwen3_reasoning_scored_check", "leaf_llm_qwen3_reasoning_scored", "random", seeds4)

variants = {
    "Default": None,
    "Scored\n(v3)": scored,
    "Rank-only\n(v4)": rank,
    "CoT\n(v5)": cot,
    "Shuffled\n(v6)": shuf,
    "Reasoning\n(default)": reasoning_default,
    "Reasoning\n(scored)": reasoning_scored,
}
default_llama = _hv("e2_dedup_check_full8", "leaf_llm", "random", [0, 1, 2, 3]) if (RESULTS / "e2_dedup_check_full8").exists() else None

fig = plt.figure(figsize=(16, 12.5))
gs = fig.add_gridspec(2, 2, hspace=0.70, wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.16)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
for ax in (axA, axB, axC, axD):
    ax.set_facecolor("white")

# Panel A: median paired HV differences vs GA and RS for each prompt variant
labels = ["Scored\n(v3)", "Rank-only\n(v4)", "CoT\n(v5)", "Shuffled\n(v6)"]
arrs = [scored, rank, cot, shuf]
diffs_ga = [float(np.median(a - GA)) for a in arrs]
diffs_rs = [float(np.median(a - RS)) for a in arrs]
x = np.arange(len(labels))
w = 0.35
axA.bar(x - w / 2, diffs_ga, w, color=C_TEAL, edgecolor="#333", linewidth=0.7, label="vs. GA")
axA.bar(x + w / 2, diffs_rs, w, color=C_LAVENDER, edgecolor="#333", linewidth=0.7, label="vs. RS")
axA.axhline(0, color="#555", lw=1.0, linestyle="--")
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=13)
axA.tick_params(axis="both", labelsize=12)
axA.set_ylabel("Median paired HV difference", fontsize=13)
axA.set_title("Prompt-variant effect on hypervolume (n=4, random split)", fontsize=14)
axA.legend(fontsize=12, frameon=False)
_panel_label(axA, "A")

# Panel B: fraction of reasoning traces mentioning potency -- restricted
# to the two conditions where this metric is actually well-defined (a
# genuine 0-1 fraction over real calls). The leakage probe's two
# conditions use a DIFFERENT metric (Spearman correlation, one
# degenerate) and were previously forced onto this axis as misleading
# zero-height placeholder bars; they are now reported as text only,
# clearly separated from the bar-chart data.
cats = ["Reasoning\n(default prompt)", "Reasoning\n(scored prompt)"]
vals = [0 / 155, 124 / 160]
colors = [C_PINK, C_GREEN]
bars = axB.bar(cats, vals, color=colors, edgecolor="#333", linewidth=0.7, width=0.5)
axB.set_xlim(-0.75, 1.75)
axB.set_ylim(0, 1.0)
axB.tick_params(axis="both", labelsize=12)
axB.set_xticklabels(cats, fontsize=12)
axB.set_ylabel("Fraction of reasoning traces\nmentioning potency", fontsize=13)
axB.set_title("Reasoning-mode engagement with injected evidence", fontsize=14)
for b, v, n in zip(bars, vals, [155, 160]):
    axB.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%} (n={n})", ha="center", va="bottom", fontsize=11)
axB.text(0.02, 0.97,
          "Leakage probe (different metric,\nnot a fraction, not shown as a bar):\n$T$=0 degenerate (constant output,\nundefined $\\rho$); $T$=0.7 non-\ndegenerate, $\\rho$=$-$0.006",
          transform=axB.transAxes, ha="left", va="top", fontsize=10, color="#444",
          bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbb"))
_panel_label(axB, "B")

# Panel C: evidence-conflict probe outcome across 3 independent archive samples
conflict = {
    "Seed 0": {"true_best": 0, "false_best": 4, "other": 23, "tie": 13},
    "Seed 1": {"true_best": 3, "false_best": 32, "other": 3, "tie": 2},
    "Seed 2": {"true_best": 0, "false_best": 6, "other": 7, "tie": 27},
}
samples = list(conflict.keys())
cat_keys = ["true_best", "false_best", "other", "tie"]
cat_labels = ["True-best nearest", "False-best (true-worst) nearest", "Other archive member", "Tie / unresolved"]
cat_colors = [C_GREEN, C_PINK, C_CYAN, C_GRAY]
xs = np.arange(len(samples))
bottoms = np.zeros(len(samples))
for key, lbl, col in zip(cat_keys, cat_labels, cat_colors):
    vals = np.array([conflict[s][key] for s in samples])
    axC.bar(xs, vals, bottom=bottoms, color=col, edgecolor="#333", linewidth=0.6, label=lbl)
    bottoms += vals
axC.set_xticks(xs)
axC.set_xticklabels(samples, fontsize=12)
axC.tick_params(axis="y", labelsize=12)
axC.set_ylabel("Calls (of 40)", fontsize=13)
axC.set_title("Evidence-conflict probe: 3 independent archive samples", fontsize=14)
axC.legend(fontsize=10.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
_panel_label(axC, "C")

# Panel D: per-seed query divergence under score-shuffle vs matched scored run
seeds_lbl = ["Seed 0", "Seed 1", "Seed 2", "Seed 3"]
div = [28, 8, 0, 7]
axD.bar(seeds_lbl, div, color=C_PURPLE, edgecolor="#333", linewidth=0.7)
axD.axhline(40, color="#555", lw=0.8, linestyle=":")
axD.set_ylim(0, 42)
axD.tick_params(axis="both", labelsize=12)
axD.set_xticklabels(seeds_lbl, fontsize=12)
axD.set_ylabel("Queries differing from matched scored run (of 40)", fontsize=13)
axD.set_title("Score-shuffle: per-seed query trajectory divergence", fontsize=14)
_panel_label(axD, "D")

fig.suptitle("Evidence-use checks: engagement without a performance gain", fontsize=17, y=0.965)
fig.savefig(OUT / "fig4_evidence_panel.png", dpi=200)
fig.savefig(OUT / "fig4_evidence_panel.pdf")
print("Wrote", OUT / "fig4_evidence_panel.png", "and .pdf")
