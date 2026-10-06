"""Figure 6: post-hoc fragment-importance diagnostic (NOT the Hypothesis
Ledger -- that mechanism is documented as empty/inactive throughout this
paper). This figure visualizes, for each reagent slot (A=amine,
B=aldehyde, C=isocyanide), the mean O_GT-predicted potency+potency_raw
across every full-vocabulary candidate containing each fragment value,
restricted to values appearing in >=20 candidates. Source data:
results/e_fragment_importance_posthoc.json, produced by a standalone
script scoring the full 30,044-candidate space with the same frozen
random-split oracle used throughout the paper -- no new search, no
budget charged, no Ledger involvement.

Run: python scripts/make_fig6_fragment_importance.py
Writes figures/paper/fig6_fragment_importance.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

C_TEAL = "#BFE8C8"
C_LAVENDER = "#A1A1F7"
C_PINK = "#F2C7DB"

data = json.loads(Path("results/e_fragment_importance_posthoc.json").read_text())

fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
slot_titles = {"A": "Amine (slot A)", "B": "Aldehyde (slot B)", "C": "Isocyanide (slot C)"}

for ax, slot in zip(axes, ["A", "B", "C"]):
    top = data[slot]["top5"]
    bottom = data[slot]["bottom5"][::-1]
    labels = [f"{v[:22]}{'…' if len(v) > 22 else ''}" for v, _ in top] + \
              [f"{v[:22]}{'…' if len(v) > 22 else ''}" for v, _ in bottom]
    vals = [s for _, s in top] + [s for _, s in bottom]
    colors = [C_TEAL] * len(top) + [C_PINK] * len(bottom)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="#333", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5, family="monospace")
    ax.invert_yaxis()
    ax.axvline(data[slot]["overall_mean"], color="#555", lw=1.0, linestyle="--")
    ax.set_xlabel("Mean predicted HeLa+RAW264.7\nacross containing candidates")
    ax.set_title(slot_titles[slot], fontsize=11)
    ax.set_facecolor("white")

axes[0].text(-0.02, 1.14, "Top 5 (teal) / bottom 5 (pink) fragment values by mean predicted potency",
             transform=axes[0].transAxes, fontsize=9, ha="left")

fig.suptitle(
    "Post-hoc fragment-importance diagnostic (full 30,044-candidate space, frozen oracle; not the Hypothesis Ledger)",
    fontsize=11.5, y=1.03,
)
fig.tight_layout()
fig.savefig(OUT / "fig6_fragment_importance.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig6_fragment_importance.pdf", bbox_inches="tight")
print("Wrote", OUT / "fig6_fragment_importance.png", "and .pdf")
