"""Evaluate O_GT generalization across random / scaffold-cold / chemotype-cold
splits (RQ5), and apply the pre-registered go/no-go threshold (§5.11) that
decides whether a split's downstream results are reportable as findings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr, pearsonr


@dataclass
class SplitResult:
    split_name: str
    head: str
    spearman: float
    pearson: float
    n_test: int
    interpretable: bool


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, split_name: str,
                    head: str, min_spearman: float) -> SplitResult:
    rho, _ = spearmanr(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return SplitResult(
        split_name=split_name, head=head, spearman=float(rho), pearson=float(r),
        n_test=len(y_true), interpretable=bool(rho >= min_spearman),
    )


def build_splits(fragment_tuples: list, seed: int = 0, frac_test: float = 0.2):
    """Return (random_idx, scaffold_cold_idx, chemotype_cold_idx) train/test
    index splits, for the real (A, B, C) Ugi-3CR fragment schema
    (fragments/vocabulary.py). Scaffold-cold groups by (A, B) -- the
    amine+aldehyde backbone, analogous to "scaffold" in standard molecular
    OOD splitting -- and holds out entire backbone groups. Chemotype-cold
    groups by C alone -- the isocyanide tail, the only real per-molecule
    structural axis with just 5 distinct values in the current AGILE
    vocabulary, so this split necessarily holds out very few, very large
    groups; interpret its results with that in mind.
    """
    rng = np.random.default_rng(seed)
    n = len(fragment_tuples)
    idx = np.arange(n)

    rng.shuffle(idx)
    n_test = int(n * frac_test)
    random_split = (idx[n_test:], idx[:n_test])

    def group_split(key_fn):
        groups = {}
        for i, frag in enumerate(fragment_tuples):
            groups.setdefault(key_fn(frag), []).append(i)
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        test_idx, train_idx = [], []
        target = int(n * frac_test)
        # Reserve the first group for train unconditionally so a single
        # oversized group (or too few groups relative to n, easy to hit
        # with a small smoke-test vocabulary) can't leave train empty.
        if group_keys:
            train_idx.extend(groups[group_keys[0]])
            remaining_keys = group_keys[1:]
        else:
            remaining_keys = []
        for k in remaining_keys:
            if len(test_idx) < target:
                test_idx.extend(groups[k])
            else:
                train_idx.extend(groups[k])
        if not test_idx and train_idx:
            # Still empty (e.g. only one group total): fall back to a
            # random subsample of train for test rather than returning an
            # empty split.
            k = max(1, min(target, len(train_idx) - 1))
            test_idx, train_idx = train_idx[:k], train_idx[k:]
        # np.array([]) defaults to float64, which breaks fancy indexing --
        # pin dtype explicitly.
        return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

    scaffold_split = group_split(lambda f: (f.A, f.B))
    chemotype_split = group_split(lambda f: f.C)

    return {"random": random_split, "scaffold_cold": scaffold_split, "chemotype_cold": chemotype_split}
