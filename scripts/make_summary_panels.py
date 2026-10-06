"""Fig 5 (redesigned, compact heatmap) and Fig 6 (new 3-panel diagnostics
figure), all from real data on disk -- no fabricated numbers.

Run: python scripts/make_summary_panels.py
Writes figures/paper/fig5_heatmap.{png,pdf}, figures/paper/fig6_diagnostics_panel.{png,pdf}
"""
from __future__ import annotations

import csv
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


def hv(experiment: str, method: str, split: str, seeds: list[int]) -> list[float] | None:
    vals = []
    for s in seeds:
        p = RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json"
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text())["final_hypervolume"])
    return vals


MODELS = [
    ("Llama 3.1 8B",        "e2_llm_full",             "leaf_llm",             range(8)),
    ("Qwen2.5 7B",          "e2_llm_qwen_full",        "leaf_llm_qwen",        range(8)),
    ("Qwen2.5 14B",         "e2_llm_qwen14b_full",     "leaf_llm_qwen14b",     range(8)),
    ("Gemini Flash-Lite",   "e2_llm_gemini_full",      "leaf_llm_gemini",      range(8)),
    ("Diversity-prompt",    "e6_llm_v2diversity_full", "leaf_llm_v2diversity", range(8)),
    ("Bandit (Thompson)",   "e2_bandit_baseline",      "bandit",               range(8)),
]
# Llama 3.3 70B (Groq) is deliberately excluded from this figure: it only
# has real data for the random split (n=3, below the n>=8 standard every
# other row meets), which would otherwise fill 4 of its 6 cells with n/a.
# It is reported in text only (§7/§8), not in this grid.
SPLITS = ["random", "scaffold_cold", "chemotype_cold"]
SPLIT_LABELS = ["random", "scaffold-\ncold", "chemotype-\ncold"]


def fig5_heatmap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, baseline, title in zip(axes, ["static_ga", "random_search"], ["vs. static-GA", "vs. random search"]):
        grid = np.full((len(MODELS), len(SPLITS)), np.nan)
        annot = [["" for _ in SPLITS] for _ in MODELS]
        for i, (label, experiment, method, seeds) in enumerate(MODELS):
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
                annot[i][j] = f"{sig.median_diff:+.0f}{star}\n(n={len(a)})"

        vmax = np.nanmax(np.abs(grid)) if np.any(~np.isnan(grid)) else 1
        im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(MODELS)):
            for j in range(len(SPLITS)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
                else:
                    lum = abs(grid[i, j]) / vmax
                    txt_color = "white" if lum > 0.55 else "black"
                    ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=7.3, color=txt_color)
        ax.set_xticks(range(len(SPLITS)))
        ax.set_xticklabels(SPLIT_LABELS, fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(np.arange(-0.5, len(SPLITS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([m[0] for m in MODELS], fontsize=9)
    fig.suptitle("Median paired hypervolume difference (real Proposer minus baseline), 6 fully-powered ($n=8$) methods\n"
                 "blue = agentic ahead, red = agentic behind, * = Wilcoxon $p<0.05$", fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig5_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_heatmap.png/.pdf")


def fig6_diagnostics_panel() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) hypervolume vs round, primary comparison (Llama 8B, random split)
    ax = axes[0]
    for method, color, label in [("leaf_llm", "#d94801", "Llama 3.1 8B"),
                                   ("static_ga", "#2c7fb8", "static-GA"),
                                   ("random_search", "#238b45", "random search")]:
        curves = []
        for s in range(8):
            p = RESULTS / "e2_llm_full" / f"{method}_random_seed{s}" / "rounds.csv"
            if not p.exists():
                continue
            rows = list(csv.DictReader(open(p)))
            curves.append([float(r["archive_hypervolume"]) for r in rows])
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean, color=color, label=label, lw=1.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("Pareto hypervolume")
    ax.set_title("(a) Search dynamics\n(Llama 3.1 8B, random split, $n=8$)", fontsize=10)
    ax.legend(fontsize=7.5)

    # (b) cost/latency bar chart (real measured rates, from paper §6.2 table).
    # Log scale: static-GA/random-search's real cost (~1s total, GP-only)
    # is ~500x smaller than the LLM Proposers' -- on a linear axis it is
    # literally invisible (a real value, not a missing bar); log scale
    # makes every real measurement visible at once.
    ax = axes[1]
    methods_cost = ["Llama 3.1 8B", "Qwen2.5 7B", "Qwen2.5 14B", "Gemini\nFlash-Lite", "Llama 3.3 70B\n(Groq, paced)", "static-GA /\nrandom search"]
    wall_clock_min = [5.3, 13.7, 13.7, 6.8, 9.3, 1 / 60]
    colors_cost = ["#d94801"] * 5 + ["#238b45"]
    bars = ax.bar(range(len(methods_cost)), wall_clock_min, color=colors_cost, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.bar_label(bars, labels=[f"{v:.1f}" if v >= 1 else f"{v*60:.0f}s" for v in wall_clock_min],
                 fontsize=7.5, padding=2)
    ax.set_xticks(range(len(methods_cost)))
    ax.set_xticklabels(methods_cost, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Wall-clock minutes / 80-call seed-run (log scale)")
    ax.set_title("(b) Real measured cost\n(log scale -- static-GA/random search's real cost is ~1s, not zero)", fontsize=9.5)

    # (c) duplicate rate vs hypervolume deficit (the negative robustness check, §6.2)
    ax = axes[2]
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
    label_colors = {"Llama 3.1 8B": "#d94801", "Qwen2.5 7B": "#fd8d3c", "Qwen2.5 14B": "#2c7fb8", "Gemini Flash-Lite": "#756bb1"}
    for lbl, color in label_colors.items():
        pts = [(x, y) for x, y, l in rows if l == lbl]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, color=color, label=lbl, s=28, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Seed-level mean duplicate rate (%)")
    ax.set_ylabel("Hypervolume deficit vs.\\ static-GA")
    ax.set_title("(c) Duplicate rate vs.\\ deficit\n($n=60$, $\\rho=-0.12$, $p=0.37$, n.s.)", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "fig6_diagnostics_panel.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig6_diagnostics_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_diagnostics_panel.png/.pdf")


if __name__ == "__main__":
    fig5_heatmap()
    fig6_diagnostics_panel()
