"""Real active-learning Gaussian-process illustration: fits the SAME
Tanimoto-kernel GP class used everywhere in this paper (leaf/oracle/gp_model.py
MultiHeadGP) on the real, vendored AGILE random-split training data, at an
early (small-n) and late (full-n) label count, and shows the posterior mean
+/- 2 std narrowing along the top principal component of the real Morgan
fingerprint embeddings. This is a genuine model-on-real-data plot -- the
GP, kernel, embeddings, and labels are all the paper's actual objects, not
a synthetic textbook illustration -- projected to 1D purely for
visualization (the GP itself is fit in the real high-dimensional
fingerprint space, never in the PCA space).

Run: python scripts/make_gp_active_learning_plot.py
Writes figures/paper/fig_gp_active_learning.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from leaf.data.agile_loader import load_agile_labeled_set
from leaf.fragments.embed import FragmentEmbedder
from leaf.oracle.eval_splits import build_splits
from leaf.oracle.gp_model import MultiHeadGP

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

COLOR_EARLY = "#d94801"
COLOR_LATE = "#2c7fb8"
COLOR_TRUE = "#252525"

frags, labels = load_agile_labeled_set()
embedder = FragmentEmbedder()
splits = build_splits(frags, seed=0)
train_idx, test_idx = splits["random"]
train_frags = [frags[i] for i in train_idx]
test_frags = [frags[i] for i in test_idx]
X_train = embedder.embed_batch(train_frags)
y_train = labels["potency"][train_idx]
X_test = embedder.embed_batch(test_frags)
y_test = labels["potency"][test_idx]

# Real 1D projection of the real fingerprint space (fit on train, applied to
# both), for visualization only -- the GP itself is always fit/evaluated in
# the real high-dimensional fingerprint space, never in this PCA space.
pca = PCA(n_components=1, random_state=0).fit(X_train)
pc1_test = pca.transform(X_test).ravel()
order = np.argsort(pc1_test)

rng = np.random.default_rng(0)
n_total = len(train_idx)
n_early = 8  # first real "round" of active learning: a handful of labels
perm = rng.permutation(n_total)
early_idx = perm[:n_early]
late_idx = perm  # all real training labels

# Real GP fits -- identical class/kernel to every O_AL/O_GT in this paper.
gp_early = MultiHeadGP(heads=["potency"]).fit(X_train[early_idx], {"potency": y_train[early_idx]})
gp_late = MultiHeadGP(heads=["potency"]).fit(X_train[late_idx], {"potency": y_train[late_idx]})

# Real posterior mean/std evaluated on the genuinely HELD-OUT real test
# partition (never used for fitting either GP) -- a fair, non-circular
# comparison of how much the posterior narrows as real labels accumulate,
# sorted by PC1 for a readable 1D curve.
mean_early, std_early = gp_early.predict(X_test[order], head="potency", return_std=True)
mean_late, std_late = gp_late.predict(X_test[order], head="potency", return_std=True)
pc1_sorted = pc1_test[order]
y_sorted = y_test[order]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

for ax, mean, std, n_labels, color, title in [
    (axes[0], mean_early, std_early, n_early, COLOR_EARLY, f"Early round: {n_early} real labels"),
    (axes[1], mean_late, std_late, n_total, COLOR_LATE, f"Full budget: {n_total} real labels"),
]:
    ax.fill_between(pc1_sorted, mean - 2 * std, mean + 2 * std, color=color, alpha=0.18, label="±2 std (posterior)")
    ax.plot(pc1_sorted, mean, color=color, lw=2.2, label="GP posterior mean")
    ax.scatter(pc1_sorted, y_sorted, color=COLOR_TRUE, s=16, alpha=0.6, zorder=5,
               edgecolor="none", label="Real held-out label (test partition)")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Top principal component of real Morgan fingerprints", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Real potency (expt_Hela)", fontsize=9.5)
handles, leg_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(r"Active-learning Gaussian process: real $\mathcal{O}$ posterior on held-out data narrows as labels accumulate",
             fontsize=12.5, y=1.02)

fig.tight_layout()
fig.savefig(OUT / "fig_gp_active_learning.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "fig_gp_active_learning.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_gp_active_learning.png'} / .pdf")
print(f"early std range: {std_early.min():.3f}-{std_early.max():.3f}, late std range: {std_late.min():.3f}-{std_late.max():.3f}")
