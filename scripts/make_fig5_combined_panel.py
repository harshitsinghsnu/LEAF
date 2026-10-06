"""Combined Figure 5: real best-found lipid structures (top row, PyMOL
3D renders) + fragment-importance diagnostic (bottom row), replacing the
separate fig5_lipid_structures and fig6_fragment_importance files. The
fragment-importance row now shows actual 2D fragment structures (RDKit
thumbnails) instead of raw SMILES text labels -- a strip plot of every
fragment value's mean predicted potency at that slot, with the five
highest/lowest values annotated by their real rendered structure rather
than truncated SMILES strings.

Inputs (already produced by earlier scripts, no new experiments here):
  results/e_best_lipid_structures.json       (find_best_lipids.py)
  figures/paper/fig5_pymol_*.png             (render_pymol.py)
  results/e_fragment_importance_posthoc.json (fragment_importance.py)

Run: python scripts/make_fig5_combined_panel.py
Writes figures/paper/fig5_combined_panel.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw


def autocrop_whitespace(img_array: np.ndarray, pad: int = 12) -> np.ndarray:
    """Crop away uniform white/near-white border around the molecule so
    the PyMOL renders (which have a large empty canvas margin from the
    ray-trace buffer/zoom settings) don't force extra blank height into
    the composited figure."""
    rgb = img_array[..., :3] if img_array.shape[-1] >= 3 else img_array
    non_white = np.any(rgb < 0.97, axis=-1)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img_array
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img_array.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img_array.shape[1])
    return img_array[r0:r1, c0:c1]

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

C_TEAL = "#BFE8C8"
C_LAVENDER = "#A1A1F7"
C_PINK = "#F2C7DB"
C_GRAY = "#CFCFCF"

lipid_data = json.loads(Path("results/e_best_lipid_structures.json").read_text())
frag_data = json.loads(Path("results/e_fragment_importance_posthoc.json").read_text())

# Recompute the per-value means from the raw (not just top/bottom5) so the
# strip plot can show the FULL distribution, not just the extremes. The
# cached JSON only stores top5/bottom5 + overall mean/std, so refit here
# is unnecessary -- instead, reconstruct an approximate full set is not
# possible from the cached file; we show top5/bottom5 plus the overall
# mean/std band as the distribution summary, which is what is on disk.

def frag_thumbnail(smiles: str, size=(90, 70)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


fig = plt.figure(figsize=(17, 10.4))
gs = fig.add_gridspec(2, 3, height_ratios=[0.68, 1.15], hspace=0.20, wspace=0.14,
                       left=0.03, right=0.99, top=0.90, bottom=0.04)

# ---------------------------------------------------------------- Top row: 3D lipid structures
lipid_labels = {
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
for i, key in enumerate(order):
    ax = fig.add_subplot(gs[0, i])
    entry = lipid_data[key]
    img = autocrop_whitespace(mpimg.imread(pymol_pngs[key]))
    ax.imshow(img)
    ax.axis("off")
    ax.set_anchor("N")
    ax.set_title(f"{lipid_labels[key]}\npredicted HeLa={entry['pred_potency']:.1f}, "
                 f"RAW264.7={entry['pred_potency_raw']:.1f}", fontsize=14, pad=6)

fig.text(0.012, 0.955, "A", fontsize=22, fontweight="bold", color="#1a1a1a")
fig.text(0.012, 0.59, "B", fontsize=22, fontweight="bold", color="#1a1a1a")

# ---------------------------------------------------------------- Bottom row: fragment-structure gallery, grid layout
# (a number-line + leader-line layout was tried first but produced heavy
# thumbnail/point overlap whenever several fragment values clustered
# closely together, especially at slot C -- a fixed 2x5 grid per slot
# avoids that entirely and is the standard "top/bottom structure
# gallery" layout for this kind of result.)
slot_titles = {"A": "Amine (slot A)", "B": "Aldehyde (slot B)", "C": "Isocyanide (slot C)"}
for i, slot in enumerate(["A", "B", "C"]):
    ax = fig.add_subplot(gs[1, i])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_anchor("N")
    top = frag_data[slot]["top5"]
    bottom = frag_data[slot]["bottom5"]

    ax.text(2.5, 1.97, "Highest-scoring", fontsize=12.5, ha="center", va="top", color="#2f6f3e")
    ax.text(2.5, 0.92, "Lowest-scoring", fontsize=12.5, ha="center", va="top", color="#9e3b63")

    for col, (smi, val) in enumerate(top):
        img = frag_thumbnail(smi)
        cx = col + 0.5
        if img is not None:
            ob = OffsetImage(img, zoom=0.40)
            ab = AnnotationBbox(ob, (cx, 1.55), frameon=True, pad=0.15,
                                 bboxprops=dict(edgecolor=C_TEAL, linewidth=1.6, facecolor="white"))
            ax.add_artist(ab)
        ax.text(cx, 1.05, f"{val:.2f}", ha="center", va="top", fontsize=11.5)

    for col, (smi, val) in enumerate(bottom):
        img = frag_thumbnail(smi)
        cx = col + 0.5
        if img is not None:
            ob = OffsetImage(img, zoom=0.40)
            ab = AnnotationBbox(ob, (cx, 0.5), frameon=True, pad=0.15,
                                 bboxprops=dict(edgecolor=C_PINK, linewidth=1.6, facecolor="white"))
            ax.add_artist(ab)
        ax.text(cx, 0.0, f"{val:.2f}", ha="center", va="top", fontsize=11.5)

    ax.set_title(slot_titles[slot], fontsize=15)
    ax.set_facecolor("white")

fig.text(0.5, 0.97,
          "Real best-found lipids (A) and post-hoc fragment-importance diagnostic (B, not the Hypothesis Ledger)",
          fontsize=17, ha="center")

fig.savefig(OUT / "fig5_combined_panel.png", dpi=200)
fig.savefig(OUT / "fig5_combined_panel.pdf")
print("Wrote", OUT / "fig5_combined_panel.png", "and .pdf")
