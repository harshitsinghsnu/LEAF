"""Real de-duplication check figure, slope-chart style: shows the shift
in median paired hypervolume difference from "no dedup" to "with dedup"
for each (split, baseline) pair, pulled directly from
results/e2_dedup_check_full8* (n=8, random) and
results/e2_dedup_check_other_splits* (n=4, scaffold-cold/chemotype-cold)
via leaf.eval.significance.paired_significance -- never hand-typed.

Run: python scripts/make_dedup_panel.py
Writes figures/paper/fig_dedup_check.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leaf.eval.significance import paired_significance

RESULTS = Path("results")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

SPLIT_COLORS = {
    "Random (n=8)": "#2c7fb8",
    "Scaffold-cold (n=4, exploratory)": "#41ab5d",
    "Chemotype-cold (n=4, exploratory)": "#d94801",
}


def _hv(experiment: str, method: str, split: str, seeds: list[int]) -> np.ndarray:
    return np.array([
        json.loads((RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json").read_text())["final_hypervolume"]
        for s in seeds
    ])


SPLITS = [
    ("Random (n=8)", "random", list(range(8)), "e2_dedup_check_full8_control", "e2_dedup_check_full8", True),
    ("Scaffold-cold (n=4, exploratory)", "scaffold_cold", list(range(4)), "e2_dedup_check_other_splits_control", "e2_dedup_check_other_splits", False),
    ("Chemotype-cold (n=4, exploratory)", "chemotype_cold", list(range(4)), "e2_dedup_check_other_splits_control", "e2_dedup_check_other_splits", False),
]
BASELINES = [("static_ga", "vs. static-GA", "-"), ("random_search", "vs. random search", "--")]

fig, ax = plt.subplots(figsize=(7.8, 6.4))

x_ctrl, x_dedup = 0, 1
for split_label, split, seeds, ctrl_exp, dedup_exp, reliable in SPLITS:
    color = SPLIT_COLORS[split_label]
    for baseline, baseline_label, linestyle in BASELINES:
        llm_ctrl = _hv(ctrl_exp, "leaf_llm", split, seeds)
        llm_dedup = _hv(dedup_exp, "leaf_llm", split, seeds)
        base_ctrl = _hv(ctrl_exp, baseline, split, seeds)
        base_dedup = _hv(dedup_exp, baseline, split, seeds)
        res_ctrl = paired_significance(llm_ctrl - base_ctrl)
        res_dedup = paired_significance(llm_dedup - base_dedup)

        ax.plot([x_ctrl, x_dedup], [res_ctrl.median_diff, res_dedup.median_diff],
                color=color, linestyle=linestyle, lw=2.4, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                label=f"{split_label.split(' (')[0]}, {baseline_label}")

        star_ctrl = "*" if (reliable and res_ctrl.wilcoxon_reliable and res_ctrl.wilcoxon_p is not None and res_ctrl.wilcoxon_p < 0.05) else ""
        star_dedup = "*" if (reliable and res_dedup.wilcoxon_reliable and res_dedup.wilcoxon_p is not None and res_dedup.wilcoxon_p < 0.05) else ""
        ax.annotate(f"{res_ctrl.median_diff:+.1f}{star_ctrl}", xy=(x_ctrl, res_ctrl.median_diff),
                    xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=color, fontweight="bold")
        ax.annotate(f"{res_dedup.median_diff:+.1f}{star_dedup}", xy=(x_dedup, res_dedup.median_diff),
                    xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=color, fontweight="bold")

ax.axhline(0, color="#bdbdbd", lw=1.2, linestyle=":", zorder=0)
ax.set_xlim(-0.55, 1.65)
ax.set_xticks([x_ctrl, x_dedup])
ax.set_xticklabels(["No dedup\n(control)", "With\nde-duplication"], fontsize=11)
ax.set_ylabel("Median paired hypervolume difference (LLM Proposer − baseline)", fontsize=9.5)
ax.set_title("De-duplication narrows the deficit on 2 of 3 splits, not uniformly", fontsize=12.5, pad=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Custom legend: solid/dashed = baseline, color = split
from matplotlib.lines import Line2D
split_handles = [Line2D([0], [0], color=c, lw=2.4, label=s.split(" (")[0]) for s, c in SPLIT_COLORS.items()]
style_handles = [
    Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="vs. static-GA"),
    Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="vs. random search"),
]
leg1 = ax.legend(handles=split_handles, loc="lower left", fontsize=8.5, frameon=False, title="Split", title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc="upper right", fontsize=8.5, frameon=False, title="Baseline", title_fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig_dedup_check.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "fig_dedup_check.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_dedup_check.png'} / .pdf")
