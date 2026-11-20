"""Regression tests for two real findings about the fragment embedding.

1. A bare radius-2 Morgan fingerprint collapses most of the real AGILE
   vocabulary to a small number of unique embeddings (measured: 1200 ->
   220 unique at both 1024 and 2048 bits -- not a hash-folding issue, a
   genuine blind spot of circular fingerprints against this chemical
   series' tail-length variation).
2. Concatenating normalized real descriptors (MolWt, HeavyAtomCount,
   LogP) recovers most of that lost resolution (1200 -> 1078 unique) but
   was tested end-to-end via E1 oracle validation and found to make real
   generalization dramatically WORSE (TanimotoKernel is a Jaccard-style
   similarity for sparse binary vectors; naive concatenation of dense
   continuous features distorts it). `use_descriptors` therefore defaults
   to False -- this is a documented, tested-and-reverted decision, not an
   oversight. See fragments/embed.py's module docstring for the full
   writeup and LEAF_Research_Plan.md.
"""
import numpy as np
import pytest

from leaf.data.agile_loader import load_agile_labeled_set
from leaf.fragments.embed import FragmentEmbedder

pytest.importorskip("rdkit")


def _n_unique(X: np.ndarray) -> int:
    return len({tuple(row) for row in X.tolist()})


def test_bare_fingerprint_default_and_severely_collision_prone():
    """Documents the real collision severity of the DEFAULT embedding
    (use_descriptors=False) -- this is a known, disclosed limitation, not
    something to be silently 'fixed' by re-enabling descriptors, which
    empirically hurts real oracle generalization (see module docstring)."""
    frags, _ = load_agile_labeled_set()
    X = FragmentEmbedder().embed_batch(frags)  # default: no descriptors
    n_unique = _n_unique(X)
    assert n_unique < len(frags) * 0.3


def test_descriptors_reduce_collisions_when_explicitly_enabled():
    frags, _ = load_agile_labeled_set()
    bare = FragmentEmbedder(use_descriptors=False).embed_batch(frags)
    with_desc = FragmentEmbedder(use_descriptors=True).embed_batch(frags)
    assert _n_unique(with_desc) > _n_unique(bare) * 3


def test_descriptor_block_is_bounded_for_tanimoto_kernel():
    """TanimotoKernel assumes ~[0,1]-scale features; the descriptor block
    must never blow past that regardless of input, or it can dominate the
    fingerprint bits and destabilize the kernel (research plan §3.3)."""
    frags, _ = load_agile_labeled_set()
    X = FragmentEmbedder(use_descriptors=True).embed_batch(frags[:50])
    descriptor_cols = X[:, -3:]
    assert descriptor_cols.min() >= 0.0
    assert descriptor_cols.max() <= 1.0
