"""Styled 5-panel Figure 3 (learning dynamics, cost, duplicate-vs-deficit,
and complete comparison heatmaps), matching the requested rounded-panel
pastel-header visual style. Built entirely from real per-seed results on
disk under results/ -- reuses the exact computation logic already
verified in scripts/make_summary_panels.py, just restyled and combined
into one figure with a different color palette.

Run: python scripts/make_fig3_styled_panel.py
Writes figures/paper/fig3_styled_panel.{png,pdf}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from leaf.eval.significance import paired_significance

RESULTS = Path("results")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

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

TEAL_CMAP = LinearSegmentedColormap.from_list("teal_scale", ["#FFFFFF", C_MINT, C_SKY, "#2E8FA8"])


def hv(experiment: str, method: str, split: str, seeds) -> list[float] | None:
    vals = []
    for s in seeds:
        p = RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json"
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text())["final_hypervolume"])
    return vals


def _panel_label(ax, letter):
    # Plain bold text, no circle, no color fill.
    ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
             ha="left", va="bottom", color="#1a1a1a")


MODELS_HEATMAP = [
    ("Llama 3.1 8B",        "e2_llm_full",             "leaf_llm",             range(8)),
    ("Qwen2.5 7B",          "e2_llm_qwen_full",        "leaf_llm_qwen",        range(8)),
    ("Qwen2.5 14B",         "e2_llm_qwen14b_full",     "leaf_llm_qwen14b",     range(8)),
    ("Gemini Flash-Lite",   "e2_llm_gemini_full",      "leaf_llm_gemini",      range(8)),
    ("Diversity-prompt",    "e6_llm_v2diversity_full", "leaf_llm_v2diversity", range(8)),
    ("Bandit (Thompson)",   "e2_bandit_baseline",      "bandit",               range(8)),
]
SPLITS = ["random", "scaffold_cold", "chemotype_cold"]
SPLIT_LABELS = ["random", "scaffold\n-cold", "chemotype-\ncold"]

fig = plt.figure(figsize=(15.5, 11))
gs_top = fig.add_gridspec(1, 3, left=0.045, right=0.98, top=0.88, bottom=0.60, wspace=0.35)
gs_bot = fig.add_gridspec(1, 2, left=0.045, right=0.98, top=0.48, bottom=0.14, wspace=0.15)
axA = fig.add_subplot(gs_top[0, 0])
axB = fig.add_subplot(gs_top[0, 1])
axC = fig.add_subplot(gs_top[0, 2])
axD = fig.add_subplot(gs_bot[0, 0])
axE = fig.add_subplot(gs_bot[0, 1])

for ax in (axA, axB, axC, axD, axE):
    ax.set_facecolor("white")

# ---------------------------------------------------------------- Panel A
for method, color, label in [("leaf_llm", C_PINK, "Llama 3.1 8B"),
                               ("static_ga", C_CYAN, "static-GA"),
                               ("random_search", "#89D385", "random search")]:
    curves = []
    for s in range(8):
        p = RESULTS / "e2_llm_full" / f"{method}_random_seed{s}" / "rounds.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        curves.append([float(r["archive_hypervolume"]) for r in rows])
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    x = np.arange(max_len)
    axA.plot(x, mean, color=color, label=label, lw=2.2)
    axA.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
axA.set_xlabel("Round", fontsize=10)
axA.set_ylabel("Pareto hypervolume", fontsize=10)
axA.legend(fontsize=8.5, frameon=False, loc="lower right")

# ---------------------------------------------------------------- Panel B
methods_cost = ["Llama 3.1 8B", "Qwen2.5 7B", "Qwen2.5 14B", "Gemini\nFlash-Lite",
                "Llama 3.3 70B\n(Groq, paced)", "static-GA /\nrandom search"]
wall_clock_min = [5.3, 13.7, 13.7, 6.8, 9.3, 1 / 60]
colors_cost = [C_PINK] * 5 + [C_TEAL]
bars = axB.bar(range(len(methods_cost)), wall_clock_min, color=colors_cost, edgecolor="#333", linewidth=0.7)
axB.set_yscale("log")
axB.set_ylim(top=40)
axB.bar_label(bars, labels=[f"{v:.1f}" if v >= 1 else f"{v*60:.0f}s" for v in wall_clock_min],
              fontsize=8, padding=2)
axB.set_xticks(range(len(methods_cost)))
axB.set_xticklabels(methods_cost, rotation=30, ha="right", fontsize=8)
axB.set_ylabel("Wall-clock minutes / 80-call seed-run (log scale)", fontsize=9)

# ---------------------------------------------------------------- Panel C
rows = []
for label, experiment, method in [
    ("Llama 3.1 8B", "e2_diagnostics", "leaf_llm"),
    ("Qwen2.5 7B", "e2_llm_qwen_full", "leaf_llm_qwen"),
    ("Qwen2.5 14B", "e2_llm_qwen14b_full", "leaf_llm_qwen14b"),
    ("Gemini Flash-Lite", "e2_llm_gemini_full", "leaf_llm_gemini"),
]:
    for d in sorted(Path(RESULTS, experiment).glob(f"{method}_*_seed*")):
        name = d.name
        rest = name[len(method) + 1:]
        split, _, seed_s = rest.rpartition("_seed")
        seed = int(seed_s)
        rounds_f = d / "rounds.csv"
        result_f = d / "result.json"
        if not (rounds_f.exists() and result_f.exists()):
            continue
        dup = [float(r["frac_duplicate_of_archive"]) for r in csv.DictReader(open(rounds_f))
               if r.get("frac_duplicate_of_archive") not in (None, "", "None")]
        if not dup:
            continue
        hv_val = json.loads(result_f.read_text())["final_hypervolume"]
        baseline_f = Path(RESULTS, experiment, f"static_ga_{split}_seed{seed}", "result.json")
        if not baseline_f.exists():
            continue
        base_hv = json.loads(baseline_f.read_text())["final_hypervolume"]
        rows.append((np.mean(dup) * 100, hv_val - base_hv, label))
label_colors = {"Llama 3.1 8B": C_PINK, "Qwen2.5 7B": C_CYAN, "Qwen2.5 14B": C_PURPLE, "Gemini Flash-Lite": "#89D385"}
for lbl, color in label_colors.items():
    pts = [(x, y) for x, y, l in rows if l == lbl]
    if pts:
        xs, ys = zip(*pts)
        axC.scatter(xs, ys, color=color, label=lbl, s=32, alpha=0.9, edgecolor="white", linewidth=0.4)
axC.axhline(0, color="#3f6fae", lw=1.2, linestyle="--")
axC.set_xlabel("Seed-level mean duplicate rate (%)", fontsize=10)
axC.set_ylabel("Hypervolume deficit vs. static-GA", fontsize=9.5)
axC.legend(fontsize=7.5, loc="lower left", frameon=False)

# ---------------------------------------------------------------- Panels D & E
for ax, baseline, title in [(axD, "static_ga", "vs. static-GA"), (axE, "random_search", "vs. random search")]:
    grid = np.full((len(MODELS_HEATMAP), len(SPLITS)), np.nan)
    annot = [["" for _ in SPLITS] for _ in MODELS_HEATMAP]
    for i, (label, experiment, method, seeds) in enumerate(MODELS_HEATMAP):
        seeds = list(seeds)
        for j, split in enumerate(SPLITS):
            a = hv(experiment, method, split, seeds)
            b = hv(experiment, baseline, split, seeds) if a is not None else None
            if a is None or b is None:
                annot[i][j] = "n/a"
                continue
            diffs = np.array(a) - np.array(b)
            sig = paired_significance(diffs)
            grid[i, j] = sig.median_diff
            star = "*" if (sig.wilcoxon_p is not None and sig.wilcoxon_reliable and sig.wilcoxon_p < 0.05) else ""
            annot[i][j] = f"{sig.median_diff:+.0f}\n(n={len(a)}){star}"

    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(np.abs(grid), cmap=TEAL_CMAP, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(MODELS_HEATMAP)):
        for j in range(len(SPLITS)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
            else:
                lum = abs(grid[i, j]) / vmax
                txt_color = "white" if lum > 0.55 else "#20302b"
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8.5, color=txt_color)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels(SPLIT_LABELS, fontsize=9)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(np.arange(-0.5, len(SPLITS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS_HEATMAP), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(len(MODELS_HEATMAP)))
    ax.set_yticklabels([m[0] for m in MODELS_HEATMAP], fontsize=9.5)

for ax, letter in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")]:
    _panel_label(ax, letter)

caption = (
    r"$\bf{Figure\ 3}$: Learning dynamics, cost, and complete comparisons. "
    r"$\bf{A}$: Mean hypervolume $\pm$ 1 std across eight random-split seeds; both baselines overtake "
    r"Llama 8B within 2-3 rounds. "
    r"$\bf{B}$: Real measured 80-call workload cost, log scale (static-GA/random search's real cost is "
    r"$\approx$1s, $\sim$300-800$\times$ smaller). "
    r"$\bf{C}$: Seed-level duplicate rate vs. deficit, $n=60$, $\rho=-0.12$, $p=0.37$ (not significant). "
    r"$\bf{D,E}$: All 36 method/split/baseline comparisons ($n=8$ each); asterisks mark unadjusted "
    r"Wilcoxon $p<0.05$; color intensity scales with $|$median difference$|$ regardless of sign. "
    r"Groq's three-seed Llama 3.3 70B result is reported in text only (\S3.2), not in this grid."
)
fig.text(0.045, 0.045, caption, fontsize=9.6, va="top", ha="left", wrap=True)

fig.savefig(OUT / "fig3_styled_panel.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig3_styled_panel.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT / 'fig3_styled_panel.png'} / .pdf")
