"""Figure 5: 3D ball-and-stick structures of the real best-found lipid (by
predicted potency+potency_raw under the frozen O_GT, random split) for
each of the three methods compared in Section 3.3's objective-scored-
prompt check (seed 1). Structures are assembled from the real queried
A/B/C triples on disk (results/e2_llm_v3scored_check/*_random_seed1/
queries.csv); potency predictions are produced by find_best_lipids.py
(cached at results/e_best_lipid_structures.json). 3D conformers are
generated with RDKit ETKDG embedding + MMFF94 optimization
(scratchpad/embed_3d.py -> results/pdb_structures/*.pdb) and rendered
with PyMOL (open-source, isolated conda env `pymol_env` so its numpy 2.x
dependency never touches the project's pinned numpy 1.26 stack used by
every GP/statistics script) via scratchpad/render_pymol.py. No
illustrative or hand-drawn structures -- every atom here was proposed or
baseline-generated under the matched budget and scored by the same
frozen GP used throughout the paper.

This script only composites the three PyMOL-rendered PNGs already on
disk (figures/paper/fig5_pymol_*.png) into the final multi-panel figure.

Run: python scripts/make_fig5_lipid_structures.py
Writes figures/paper/fig5_lipid_structures.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

data = json.loads(Path("results/e_best_lipid_structures.json").read_text())

labels = {
    "leaf_llm_v3scored": "Llama 8B (scored prompt)",
    "static_ga": "Static-GA",
    "random_search": "Random search",
}
pymol_pngs = {
    "leaf_llm_v3scored": OUT / "fig5_pymol_llama8b_scored.png",
    "static_ga": OUT / "fig5_pymol_static_ga.png",
    "random_search": OUT / "fig5_pymol_random_search.png",
}
order = ["leaf_llm_v3scored", "static_ga", "random_search"]

fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
for ax, key in zip(axes, order):
    entry = data[key]
    img = mpimg.imread(pymol_pngs[key])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"{labels[key]}\npredicted HeLa={entry['pred_potency']:.1f}, "
        f"RAW264.7={entry['pred_potency_raw']:.1f}",
        fontsize=10,
    )
    ax.text(0.5, -0.06, f"A={entry['A']}\nB={entry['B']}\nC={entry['C']}",
             transform=ax.transAxes, ha="center", va="top", fontsize=6, family="monospace")

fig.suptitle(
    "Real best-found lipid per method, random split, scored-prompt seed 1 (n=40 queries each)\n"
    "3D conformers: RDKit ETKDG + MMFF94, rendered in PyMOL",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0.08, 1, 0.94])
fig.savefig(OUT / "fig5_lipid_structures.png", dpi=200)
fig.savefig(OUT / "fig5_lipid_structures.pdf")
print("Wrote", OUT / "fig5_lipid_structures.png", "and .pdf")
