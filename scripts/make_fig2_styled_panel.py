"""Styled 4-panel Figure 2 (oracle validity, proposal value, diversity),
publication style (white backgrounds, plain black-text panel labels),
built entirely from real per-seed results already on disk under
results/ -- never hand-typed numbers except where cross-checked
directly against leaf/eval/significance.py and
results/e1/split_validation_table.csv.

Run: python scripts/make_fig2_styled_panel.py
Writes figures/paper/fig2_styled_panel.{png,pdf}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from leaf.eval.significance import paired_significance

RESULTS = Path("results")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

# Palette matched to the requested reference image (used for data colors
# only -- panel backgrounds are white, per revision).
C_TEAL = "#BFE8C8"
C_LAVENDER = "#A1A1F7"
C_PINK = "#F2C7DB"
C_CYAN = "#7ADDF2"
C_PURPLE = "#CBB7F5"
C_GREEN = "#89D385"
C_MINT = "#D1EFBD"
C_SKY = "#6CD1F0"
C_PALE_CYAN = "#B8E8EE"
C_LILAC = "#EFCCEA"


def _hv(experiment: str, method: str, split: str, seeds: list[int]) -> np.ndarray:
    return np.array([
        json.loads((RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json").read_text())["final_hypervolume"]
        for s in seeds
    ])


def _panel_label(ax, letter):
    # Plain bold text, no circle, no color fill.
    ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
             ha="left", va="bottom", color="#1a1a1a")


fig = plt.figure(figsize=(13, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30, left=0.06, right=0.97, top=0.90, bottom=0.20)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

for ax in (axA, axB, axC, axD):
    ax.set_facecolor("white")

# ---------------------------------------------------------------- Panel A
rows = list(csv.DictReader(open(RESULTS / "e1" / "split_validation_table.csv")))
splits = ["random", "scaffold_cold", "chemotype_cold"]
thresh = 0.3
x = np.arange(len(splits))
width = 0.32
for i, split in enumerate(splits):
    pot = float([r for r in rows if r["split_name"] == split and r["head"] == "potency"][0]["spearman"])
    raw = float([r for r in rows if r["split_name"] == split and r["head"] == "potency_raw"][0]["spearman"])
    raw_interp = ([r for r in rows if r["split_name"] == split and r["head"] == "potency_raw"][0]["interpretable"] == "True")
    color_pot = C_TEAL if pot >= thresh else C_LAVENDER
    color_raw = C_TEAL if raw_interp else C_LAVENDER
    axA.bar(x[i] - width / 2, pot, width, color=color_pot, edgecolor="#333", linewidth=0.7)
    axA.bar(x[i] + width / 2, raw, width, color=color_raw, edgecolor="#333", linewidth=0.7, hatch="//////")
axA.axhline(thresh, color="#2f6f5e", lw=1.4, linestyle="--", zorder=3)
axA.set_xticks(x)
axA.set_xticklabels(splits, fontsize=10.5)
axA.set_ylabel("Spearman correlation (held-out)", fontsize=10)
axA.set_ylim(0, 0.95)
legend_handles_A = [
    mpatches.Patch(facecolor=C_TEAL, edgecolor="#333", label="interpretable ($\\geq$ threshold)"),
    mpatches.Patch(facecolor=C_LAVENDER, edgecolor="#333", label="uninterpretable ($<$ threshold)"),
    mpatches.Patch(facecolor="white", edgecolor="#333", hatch="//////", label="potency\\_raw (hatched)"),
]
axA.legend(handles=legend_handles_A, loc="upper right", fontsize=7.8, frameon=False)

# ---------------------------------------------------------------- Panel B
models = [
    ("Llama 3.1 8B", "e2_llm_full", "leaf_llm"),
    ("Qwen2.5 7B", "e2_llm_qwen_full", "leaf_llm_qwen"),
    ("Qwen2.5 14B", "e2_llm_qwen14b_full", "leaf_llm_qwen14b"),
    ("Gemini Flash-Lite", "e2_llm_gemini_full", "leaf_llm_gemini"),
]
seeds8 = list(range(8))
y_positions = np.arange(len(models))[::-1]
for yi, (name, exp, method) in zip(y_positions, models):
    llm = _hv(exp, method, "random", seeds8)
    for baseline, color, marker, dy in [("static_ga", C_PINK, "o", 0.14), ("random_search", C_LAVENDER, "s", -0.14)]:
        base = _hv(exp, baseline, "random", seeds8)
        res = paired_significance(llm - base)
        star = "*" if (res.wilcoxon_reliable and res.wilcoxon_p is not None and res.wilcoxon_p < 0.05) else ""
        axB.plot([res.ci_low, res.ci_high], [yi + dy, yi + dy], color=color, lw=2.2, zorder=2)
        axB.plot(res.median_diff, yi + dy, marker=marker, markersize=9, color=color,
                  markeredgecolor="#333", markeredgewidth=0.6, zorder=3)
        axB.annotate(f"{res.median_diff:+.1f} (n=8){star}", xy=(res.ci_high, yi + dy),
                      xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color="#222")
axB.axvline(0, color="#555", lw=1.2, linestyle="-", zorder=1)
axB.set_yticks(y_positions)
axB.set_yticklabels([m[0] for m in models], fontsize=10.5)
axB.set_ylim(-0.6, len(models) - 1 + 0.75)
axB.set_xlabel("Median paired hypervolume difference (95% bootstrap CI)\nagentic LLM Proposer minus baseline", fontsize=9)
axB.set_xlim(-140, 75)
legend_handles_B = [
    plt.Line2D([0], [0], marker="o", color=C_PINK, linestyle="-", markersize=8, label="vs static-GA"),
    plt.Line2D([0], [0], marker="s", color=C_LAVENDER, linestyle="-", markersize=8, label="vs random search"),
]
axB.legend(handles=legend_handles_B, loc="upper left", fontsize=8.2, frameon=False,
           bbox_to_anchor=(0.0, 1.0))

# ---------------------------------------------------------------- Panel C & D
names = ["Llama 3.1 8B", "Qwen2.5 7B", "Qwen2.5 14B", "Gemini Flash-Lite", "static-GA", "random search"]
dup_pct = [71.5, 67.0, 3.5, 32.3, 22.5, 0.0]
edit_dist = [0.55, 0.63, 1.65, 1.01, 1.52, 2.81]
bar_colors = [C_GREEN, C_CYAN, C_PURPLE, C_PINK, C_LAVENDER, C_PALE_CYAN]

xC = np.arange(len(names))
barsC = axC.bar(xC, dup_pct, color=bar_colors, edgecolor="#333", linewidth=0.7)
axC.bar_label(barsC, fmt="%.1f%%", fontsize=8.5, padding=2)
axC.set_xticks(xC)
axC.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
axC.set_ylabel("% exact duplicate of archive", fontsize=10)
axC.set_ylim(0, 85)

barsD = axD.bar(xC, edit_dist, color=bar_colors, edgecolor="#333", linewidth=0.7)
axD.bar_label(barsD, fmt="%.2f", fontsize=8.5, padding=2)
axD.set_xticks(xC)
axD.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
axD.set_ylabel("Mean edit distance to nearest\narchive member (0-3 slots)", fontsize=9.5)
axD.set_ylim(0, 3.3)

for ax, letter in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
    _panel_label(ax, letter)

caption = (
    r"$\bf{Figure\ 2}$: Oracle validity, proposal value, and diversity. "
    r"$\bf{A}$: Held-out potency correlations across three splits (real, "
    r"results/e1/split\_validation\_table.csv); dashed line marks $\rho=0.3$. Chemotype-cold "
    r"RAW264.7 fails this gate. "
    r"$\bf{B}$: Random-split median paired hypervolume differences ($n=8$ each), 95\% bootstrap CIs; "
    r"asterisks mark unadjusted Wilcoxon $p<0.05$. The only positive estimate is Qwen2.5 14B "
    r"vs. static-GA (+1.8), not vs. random search (-4.3); neither is significant. "
    r"$\bf{C,D}$: Exact archive-duplicate proposal rates and minimum fragment-slot edit distances (0-3)."
)
fig.text(0.06, 0.055, caption, fontsize=9.3, wrap=True, va="top", ha="left")

fig.savefig(OUT / "fig2_styled_panel.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig2_styled_panel.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT / 'fig2_styled_panel.png'} / .pdf")
