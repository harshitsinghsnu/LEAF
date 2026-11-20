"""Composite, publication-ready figure panels for the paper, assembled
from real per-seed results already on disk under results/ (never
hand-typed numbers) -- see leaf/eval/plotting.py for the per-experiment
figures these summarize across experiments.

Run: python scripts/make_paper_figures.py
Writes figures/paper/fig1_oracle_validity.{png,pdf},
       figures/paper/fig2_agentic_value_add.{png,pdf},
       figures/paper/fig3_diversity_mechanism.{png,pdf}
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

COLOR_POS = "#89D385"
COLOR_NEG = "#F2C7DB"
COLOR_NEUTRAL = "#B8E8EE"

# Same exact palette (and same model->color mapping) as
# scripts/make_fig2_styled_panel.py / make_fig3_styled_panel.py, so a
# given model reads as the same color everywhere in the paper.
C_TEAL = "#BFE8C8"
C_LAVENDER = "#A1A1F7"
C_PINK = "#F2C7DB"
C_CYAN = "#7ADDF2"
C_PURPLE = "#CBB7F5"
C_GREEN = "#89D385"
C_PALE_CYAN = "#B8E8EE"
MODEL_COLORS = {
    "Llama 3.1 8B": C_GREEN,
    "Qwen2.5 7B": C_CYAN,
    "Qwen2.5 14B": C_PURPLE,
    "Gemini Fl-Lite": C_PINK,
    "Gemini Flash-Lite": C_PINK,
    "static-GA": C_LAVENDER,
    "random search": C_PALE_CYAN,
}


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png/.pdf")


def _hv(experiment: str, method: str, split: str, seeds: list[int]) -> list[float]:
    vals = []
    for s in seeds:
        p = RESULTS / experiment / f"{method}_{split}_seed{s}" / "result.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["final_hypervolume"])
    return vals


def _diag(experiment: str, method: str, split_glob: str = "*") -> dict:
    dup, dist, cross = [], [], []
    for d in RESULTS.glob(f"{experiment}/{method}_{split_glob}_seed*"):
        f = d / "rounds.csv"
        if not f.exists():
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                v = r.get("frac_duplicate_of_archive")
                if v not in (None, "", "None"):
                    dup.append(float(v))
                v = r.get("mean_min_edit_distance")
                if v not in (None, "", "None"):
                    dist.append(float(v))
                v = r.get("frac_crossover")
                if v not in (None, "", "None"):
                    cross.append(float(v))
    return {
        "duplicate_pct": 100 * np.mean(dup) if dup else np.nan,
        "edit_distance": np.mean(dist) if dist else np.nan,
        "n_rounds": len(dup),
    }


# ---------------------------------------------------------------- Fig 1 ---
def fig1_oracle_validity() -> None:
    """E1: O_GT Spearman validity per split/head, with the pre-registered
    go/no-go threshold marked -- real numbers from
    results/e1/split_validation_table.csv."""
    rows = list(csv.DictReader(open(RESULTS / "e1" / "split_validation_table.csv")))
    splits = ["random", "scaffold_cold", "chemotype_cold"]
    heads = ["potency", "potency_raw"]
    threshold = 0.3

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(splits))
    width = 0.35
    for i, head in enumerate(heads):
        vals, interp = [], []
        for split in splits:
            row = next(r for r in rows if r["split_name"] == split and r["head"] == head)
            vals.append(float(row["spearman"]))
            interp.append(row["interpretable"] == "True")
        colors = [C_TEAL if ok else C_LAVENDER for ok in interp]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=head, color=colors,
                       hatch=None if i == 0 else "//", edgecolor="black", linewidth=0.5)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"go/no-go threshold ({threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Spearman correlation (held-out)")
    ax.set_title(r"$\mathcal{O}_{GT}$ validity across splits (E1)")
    # legend: head hatch pattern + interpretability color, built manually
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=C_TEAL, edgecolor="black", label="interpretable (>= threshold)"),
        Patch(facecolor=C_LAVENDER, edgecolor="black", label="uninterpretable (< threshold)"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="potency_raw (hatched)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower left")
    ax.set_ylim(0, 0.85)
    _save(fig, "fig1_oracle_validity")


# ---------------------------------------------------------------- Fig 2 ---
def fig2_agentic_value_add() -> None:
    """E2: real-LLM agentic value-add across every model tested, vs. both
    baselines, on the random split (the split every model has full or
    partial n>=4 data for) -- median paired hypervolume diff with 95%
    bootstrap CI, real per-seed data pulled from results/.

    Horizontal dot-and-whisker (point-range), not bars: with every real
    value negative or near-zero, bars from a zero baseline mostly just
    reproduce the axis line and waste vertical space; a point-range plot
    puts all the visual weight on the CI itself."""
    models = [
        ("Llama 3.1 8B",   "e2_llm_full",          "leaf_llm",            list(range(8))),
        ("Qwen2.5 7B",     "e2_llm_qwen_full",     "leaf_llm_qwen",       list(range(8))),
        ("Qwen2.5 14B",    "e2_llm_qwen14b_full",  "leaf_llm_qwen14b",    list(range(8))),
        ("Gemini Fl-Lite", "e2_llm_gemini_full",   "leaf_llm_gemini",     list(range(8))),
    ]
    comparisons = [("static_ga", "vs static-GA", -0.15, "o", C_PINK), ("random_search", "vs random search", 0.15, "s", C_LAVENDER)]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    n_models = len(models)
    y = np.arange(n_models)[::-1]  # top-to-bottom in listed order

    for baseline, label, offset, marker, color in comparisons:
        for yi, (name, experiment, method, seeds) in zip(y, models):
            hv_a = _hv(experiment, method, "random", seeds)
            hv_b = _hv(experiment, baseline, "random", seeds)
            n = min(len(hv_a), len(hv_b))
            if n < 2:
                continue
            diffs = np.array(hv_a[:n]) - np.array(hv_b[:n])
            sig = paired_significance(diffs)
            ypos = yi + offset
            ax.plot([sig.ci_low, sig.ci_high], [ypos, ypos], color=color, lw=1.8, zorder=1,
                    label=label if yi == y[0] else None)
            ax.scatter([sig.median_diff], [ypos], color=color, marker=marker, s=60, zorder=2,
                       edgecolor="black", linewidth=0.5)
            star = "*" if (sig.wilcoxon_p is not None and sig.wilcoxon_reliable and sig.wilcoxon_p < 0.05) else ""
            ax.text(sig.ci_high + 3, ypos, f"{sig.median_diff:+.1f}{star} (n={n})",
                    va="center", fontsize=7.5, color=color)

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in models], fontsize=9.5)
    ax.set_xlabel("Median paired hypervolume difference (95% bootstrap CI)\nagentic LLM Proposer minus baseline")
    ax.set_title("Real-LLM agentic value-add vs. matched-budget baselines\n(random split; * = Wilcoxon p<0.05; left of dashed line = underperformance)",
                 fontsize=10.5)
    ax.set_xlim(right=ax.get_xlim()[1] + 35)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "fig2_agentic_value_add")


# ---------------------------------------------------------------- Fig 3 ---
def fig3_diversity_mechanism() -> None:
    """Mechanistic diagnostics across every real Proposer tested: %
    exact-duplicate-of-archive proposals and mean edit distance to the
    nearest archive member -- the mechanism behind Fig. 2, real numbers
    pulled from rounds.csv logs across every completed experiment."""
    entries = [
        # leaf_llm's diagnostics (frac_duplicate etc.) were only logged in
        # the e2_diagnostics pass, not the full e2_llm_full sweep -- same
        # reason static-GA/random-search pull from e2_diagnostics below.
        ("Llama 3.1 8B",   "e2_diagnostics",      "leaf_llm"),
        ("Qwen2.5 7B",     "e2_llm_qwen_full",    "leaf_llm_qwen"),
        ("Qwen2.5 14B",    "e2_llm_qwen14b_full", "leaf_llm_qwen14b"),
        ("Gemini Fl-Lite", "e2_llm_gemini_full",  "leaf_llm_gemini"),
        # log_diagnostics was only turned on for these baselines in the
        # e2_diagnostics experiment (single split, 4 seeds); e2_llm_full's
        # own static_ga/random_search runs predate that config field and
        # have no diagnostics columns to pull from.
        ("static-GA",      "e2_diagnostics",      "static_ga"),
        ("random search",  "e2_diagnostics",      "random_search"),
    ]
    labels, dup_pcts, edit_dists = [], [], []
    for name, experiment, method in entries:
        d = _diag(experiment, method)
        if d["n_rounds"] == 0:
            continue
        labels.append(name)
        dup_pcts.append(d["duplicate_pct"])
        edit_dists.append(d["edit_distance"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(labels))
    colors = [MODEL_COLORS.get(l, C_PALE_CYAN) for l in labels]

    # random search's real, measured duplicate rate is exactly 0.0% -- a
    # genuine zero, not a missing bar. Label every bar with its numeric
    # value so a zero-height bar reads as "measured zero" rather than
    # "no data".
    bars1 = ax1.bar(x, dup_pcts, color=colors, edgecolor="black", linewidth=0.5)
    ax1.bar_label(bars1, fmt="%.1f%%", fontsize=7.5, padding=2)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("% exact duplicate of archive")
    ax1.set_title("Wasted-query rate\n(labeled exactly -- 0.0% for random search is a real measured zero)", fontsize=9.5)
    ax1.set_ylim(top=max(dup_pcts) * 1.18)

    bars2 = ax2.bar(x, edit_dists, color=colors, edgecolor="black", linewidth=0.5)
    ax2.bar_label(bars2, fmt="%.2f", fontsize=7.5, padding=2)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Mean edit distance to nearest\narchive member (0-3 slots)")
    ax2.set_title("Proposal novelty", fontsize=9.5)
    ax2.set_ylim(top=max(edit_dists) * 1.15)

    fig.suptitle("Mechanism: proposal duplication and novelty across all real Proposers tested", y=1.03)
    _save(fig, "fig3_diversity_mechanism")


if __name__ == "__main__":
    fig1_oracle_validity()
    fig2_agentic_value_add()
    fig3_diversity_mechanism()
    print("\nAll paper figures written to figures/paper/")
