"""Fig 4: LEAF closed-loop round architecture -- a precise schematic of
Algorithm 1 in the paper (Method §3.6), not an artistic rendering. Every
box/arrow corresponds to an actual step in leaf/orchestrator/loop.py, so
this is accurate documentation, not decorative -- built with matplotlib
patches for exact reproducibility rather than an AI image generator.

Run: python scripts/make_architecture_figure.py
Writes figures/paper/fig4_architecture.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

C_PROPOSER = "#d94801"   # the component the primary claim (H1) is about
C_FREE = "#2c7fb8"       # free steps (no budget cost)
C_BUDGET = "#238b45"     # steps that charge the budget
C_SECONDARY = "#756bb1"  # secondary/ablated components (Verifier, Hypothesis)
C_NEUTRAL = "#525252"


def box(ax, xy, w, h, text, color, fontsize=9.5, text_color="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.3, edgecolor=color, facecolor=color, alpha=0.92, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, zorder=3, wrap=True)
    return (x + w / 2, y), (x + w / 2, y + h), (x, y + h / 2), (x + w, y + h / 2)


def arrow(ax, p1, p2, color=C_NEUTRAL, style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0", label=None):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=7.5, color=color, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Row 1: archive / ledger state (inputs)
    _, top_a, _, right_a = box(ax, (0.3, 8.7), 2.2, 0.9, "Archive $A_{t-1}$\n(Pareto frontier)", C_NEUTRAL, 8.5)
    _, top_l, _, right_l = box(ax, (2.9, 8.7), 2.2, 0.9, "Ledger $H_{t-1}$", C_NEUTRAL, 8.5)
    _, top_x, _, right_x = box(ax, (5.5, 8.7), 2.2, 0.9, "Vocabulary $\\mathcal{X}$\n(29$\\times$74$\\times$14)", C_NEUTRAL, 8.5)

    # Row 2: Proposer (the primary-claim component)
    bot_p, top_p, left_p, right_p = box(ax, (0.5, 7.0), 4.0, 1.0,
        "PROPOSER AGENT\nLLM crossover/mutation over $\\mathcal{X}$\n(H1: the paper's primary claim)",
        C_PROPOSER, 9.5)
    arrow(ax, (top_a[0], 8.7), (1.5, 8.0))
    arrow(ax, (top_l[0], 8.7), (3.5, 8.0))
    arrow(ax, (top_x[0], 8.7), (6.6, 8.0), connectionstyle="arc3,rad=-0.3")

    ax.text(2.5, 6.75, "Pool $P_t$ ($m \\gg q$, free)", ha="center", va="center",
            fontsize=7.5, color=C_NEUTRAL, style="italic", zorder=3)

    # Row 3: O_AL posterior (free) + Verifier (secondary, ablated)
    bot_al, top_al, left_al, right_al = box(ax, (0.5, 5.4), 3.6, 1.0,
        "$\\mathcal{O}_{AL}$ posterior\n(surrogate, free scoring)", C_FREE, 9)
    bot_v, top_v, left_v, right_v = box(ax, (4.5, 5.4), 3.8, 1.0,
        "VERIFIER AGENTS\nDebate/conformal toxicity screen\n(secondary, H3, \\S6.3: null)", C_SECONDARY, 8.7)
    arrow(ax, bot_p, top_al)
    arrow(ax, (right_al[0], right_al[1]), left_v)

    # Row 4: Acquisition (selects q)
    bot_acq, top_acq, left_acq, right_acq = box(ax, (1.5, 3.8), 4.0, 1.0,
        "ACQUISITION\nmulti-objective UCB, selects $C_t$\n($|C_t|=q$, IDENTICAL across all methods)",
        C_FREE, 9)
    arrow(ax, bot_al, (2.3, 4.8))
    arrow(ax, bot_v, (5.7, 4.8))

    # Row 5: O_GT query (charges budget) + O_AL refit
    bot_gt, top_gt, left_gt, right_gt = box(ax, (0.5, 2.2), 4.0, 1.0,
        "$\\mathcal{O}_{GT}$ QUERY\n(frozen oracle; CHARGES budget $B$)", C_BUDGET, 9)
    bot_refit, top_refit, left_refit, right_refit = box(ax, (5.2, 2.2), 3.8, 1.0,
        "$\\mathcal{O}_{AL}$ refit\non $C_t$, $f(C_t)$", C_FREE, 9)
    arrow(ax, bot_acq, top_gt)
    arrow(ax, right_gt, left_refit, label="labels")

    # Row 6: archive update + hypothesis branch
    bot_arch, top_arch, left_arch, right_arch = box(ax, (0.5, 0.6), 3.6, 1.0,
        "NON-DOMINATED SORT\n$A_t \\leftarrow$ sort$(A_{t-1}\\cup C_t)$", C_NEUTRAL, 9)
    bot_hyp, top_hyp, left_hyp, right_hyp = box(ax, (4.5, 0.6), 4.4, 1.0,
        "HYPOTHESIS AGENT (every $k$ rounds)\ncounterfactual pairs vs.\\ $\\mathcal{O}_{GT}$ ONLY\n(secondary, H2, \\S6.3: 0/72 validated)",
        C_SECONDARY, 8.7)
    arrow(ax, bot_gt, top_arch, connectionstyle="arc3,rad=0.25")
    arrow(ax, bot_refit, top_hyp, connectionstyle="arc3,rad=-0.25")
    arrow(ax, right_arch, left_hyp, style="<->", color=C_NEUTRAL, lw=1.0)

    # Loop-back arrow: archive/ledger feed the next round's Proposer
    arrow(ax, (right_hyp[0]+0.1, 1.1), (9.4, 1.1), style="-", lw=1.3, color=C_NEUTRAL)
    arrow(ax, (9.4, 1.1), (9.4, 9.15), style="-", lw=1.3, color=C_NEUTRAL)
    arrow(ax, (9.4, 9.15), (4.6, 9.15), style="-|>", lw=1.3, color=C_NEUTRAL)
    ax.text(9.55, 5.0, "repeat until $B_t \\leq 0$", rotation=90, ha="center", va="center",
            fontsize=8.5, color=C_NEUTRAL, style="italic")

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_PROPOSER, markersize=12,
               label="Primary claim (H1): Proposer"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_SECONDARY, markersize=12,
               label="Secondary/ablated (H2, H3): null in \\S6.3"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BUDGET, markersize=12,
               label="Charges the query budget $B$"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_FREE, markersize=12,
               label="Free (no budget cost)"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=8.5, frameon=False)

    ax.set_title("LEAF closed-loop round (Algorithm 1) --- exact schematic, not illustrative",
                 fontsize=11, pad=10)
    fig.savefig(OUT / "fig4_architecture.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig4_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig4_architecture.png/.pdf")


if __name__ == "__main__":
    main()
