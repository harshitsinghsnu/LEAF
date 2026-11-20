"""Fig 5: comprehensive forest plot of every real (model, split, baseline)
comparison in the paper, recomputed directly from raw per-seed result.json
files on disk (not from the agentic_value_add_*_stats.json files, which
overwrite the same filename for both baselines within one split -- only
the most recently computed comparison survives on disk there).

Run: python scripts/make_forest_figure.py
Writes figures/paper/fig5_forest.{png,pdf}
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

COLOR_SIG_NEG = "#d94801"
COLOR_NS = "#969696"
COLOR_SIG_POS = "#2c7fb8"


def hv(experiment: str, method: str, split: str, seeds: list[int]) -> list[float] | None:
    vals = []
    for s in seeds:
        p = RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json"
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text())["final_hypervolume"])
    return vals


# (row label, experiment, method, splits-to-include, seeds)
MODELS = [
    ("Llama 3.1 8B",       "e2_llm_full",         "leaf_llm",            ["random", "scaffold_cold", "chemotype_cold"], range(8)),
    ("Qwen2.5 7B",         "e2_llm_qwen_full",    "leaf_llm_qwen",       ["random", "scaffold_cold", "chemotype_cold"], range(8)),
    ("Qwen2.5 14B",        "e2_llm_qwen14b_full", "leaf_llm_qwen14b",    ["random", "scaffold_cold", "chemotype_cold"], range(8)),
    ("Gemini Flash-Lite",  "e2_llm_gemini_full",  "leaf_llm_gemini",     ["random", "scaffold_cold"], range(8)),
    ("Llama 3.3 70B (Groq)","e2_llm_groq70b_daily","leaf_llm_groq70b",   ["random"], range(3)),
    ("Diversity-prompt",   "e6_llm_v2diversity_full", "leaf_llm_v2diversity", ["random", "scaffold_cold", "chemotype_cold"], range(8)),
    ("Bandit (Thompson)",  "e2_bandit_baseline",  "bandit",              ["random", "scaffold_cold", "chemotype_cold"], range(8)),
]
SPLIT_MARK = {"random": "o", "scaffold_cold": "s", "chemotype_cold": "^"}
SPLIT_LABEL = {"random": "random", "scaffold_cold": "scaffold-cold", "chemotype_cold": "chemotype-cold"}


def main() -> None:
    rows = []  # (label, split, baseline, median, lo, hi, n, sig)
    for label, experiment, method, splits, seeds in MODELS:
        seeds = list(seeds)
        for split in splits:
            a = hv(experiment, method, split, seeds)
            if a is None:
                continue
            for baseline in ("static_ga", "random_search"):
                b = hv(experiment, baseline, split, seeds)
                if b is None:
                    continue
                diffs = np.array(a) - np.array(b)
                sig = paired_significance(diffs)
                is_sig = bool(sig.wilcoxon_p is not None and sig.wilcoxon_reliable and sig.wilcoxon_p < 0.05)
                rows.append((label, split, baseline, sig.median_diff, sig.ci_low, sig.ci_high, len(a), is_sig))

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.5))
    y_labels = []
    for i, (label, split, baseline, med, lo, hi, n, is_sig) in enumerate(rows):
        y = len(rows) - i
        color = COLOR_SIG_NEG if (is_sig and med < 0) else (COLOR_SIG_POS if (is_sig and med > 0) else COLOR_NS)
        ax.plot([lo, hi], [y, y], color=color, lw=1.8, zorder=1)
        ax.scatter([med], [y], color=color, marker=SPLIT_MARK[split], s=55, zorder=2,
                   edgecolor="black", linewidth=0.4)
        baseline_short = "static-GA" if baseline == "static_ga" else "random search"
        y_labels.append(f"{label}  vs. {baseline_short}  ({SPLIT_LABEL[split]}, n={n})")

    ax.axvline(0, color="black", lw=1.0, linestyle="--", zorder=0)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Median paired hypervolume difference (95% bootstrap CI)")
    ax.set_title("Every real agentic-vs-baseline comparison in this paper\n"
                  "(orange = significant negative, blue = significant positive, gray = not significant)",
                  fontsize=10.5)
    ax.set_ylim(0.3, len(rows) + 0.7)

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NS, markeredgecolor="black", label="random split"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_NS, markeredgecolor="black", label="scaffold-cold split"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_NS, markeredgecolor="black", label="chemotype-cold split"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig5_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig5_forest.png/.pdf  ({len(rows)} comparisons)")


if __name__ == "__main__":
    main()
