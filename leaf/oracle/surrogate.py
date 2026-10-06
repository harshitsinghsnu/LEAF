"""The active-learning surrogate, O_AL.

Refit every round on ONLY the labels acquired so far within the budget
(i.e. the subset of O_GT's outputs actually "queried"). Its posterior
mean/variance drive: (1) the acquisition function, (2) the Verifier's
toxicity prior, (3) the Hypothesis Agent's attribution signal. It is never
treated as ground truth -- see research plan §3.1.
"""
from __future__ import annotations

import numpy as np

from leaf.oracle.gp_model import MultiHeadGP


class ALSurrogate:
    def __init__(self, heads: list[str] = ("potency", "toxicity")):
        self.heads = list(heads)
        self._gp = MultiHeadGP(heads)
        self.X_acquired: list[np.ndarray] = []
        self.Y_acquired: dict[str, list[float]] = {h: [] for h in heads}

    def add_labels(self, X_new: np.ndarray, Y_new: dict[str, np.ndarray]) -> None:
        """Record newly acquired ground-truth labels; does not refit yet."""
        self.X_acquired.append(X_new)
        for h in self.heads:
            if h in Y_new:
                self.Y_acquired[h].extend(np.atleast_1d(Y_new[h]).tolist())

    def refit(self) -> "ALSurrogate":
        """Refit on all labels acquired so far. Called once per round."""
        if not self.X_acquired:
            return self
        X = np.concatenate(self.X_acquired, axis=0)
        Y = {h: np.array(v) for h, v in self.Y_acquired.items() if len(v) == X.shape[0]}
        self._gp.fit(X, Y)
        return self

    def posterior(self, X: np.ndarray, head: str) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self._gp.predict(X, head, return_std=True)
        return mean, std

    @property
    def n_labels(self) -> int:
        return sum(len(v) for v in self.Y_acquired.values()) // max(len(self.heads), 1)
