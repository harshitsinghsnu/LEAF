"""The frozen ground-truth oracle, O_GT.

Fit ONCE on the full harmonized labeled dataset (Xu et al. + AGILE, see
data harmonization in data/processed/), then frozen before any round-loop
experiment runs. This is the object that is *queried* -- it stands in for
a real lab. It must NOT be refit inside the orchestrator loop; only
oracle/surrogate.py (O_AL) is refit per round. Conflating the two was the
central bug in Revision 1 -- see research plan §3.1.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from leaf.oracle.gp_model import MultiHeadGP


class GroundTruthOracle:
    def __init__(self, heads: list[str] = ("potency", "toxicity")):
        self.heads = list(heads)
        self._gp: MultiHeadGP | None = None
        self._frozen = False

    def fit(self, X: np.ndarray, Y: dict[str, np.ndarray]) -> "GroundTruthOracle":
        if self._frozen:
            raise RuntimeError(
                "O_GT is frozen and must not be refit -- refitting the ground "
                "truth inside the loop reintroduces the O_GT/O_AL conflation bug."
            )
        self._gp = MultiHeadGP(self.heads).fit(X, Y)
        return self

    def freeze(self) -> "GroundTruthOracle":
        self._frozen = True
        return self

    def query(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return point-label predictions (mean only) as if querying a real
        lab -- callers should not use the O_GT posterior variance as an
        acquisition signal (that is O_AL's job)."""
        if self._gp is None:
            raise RuntimeError("O_GT has not been fit.")
        return {h: self._gp.predict(X, h, return_std=False) for h in self.heads}

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "GroundTruthOracle":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not obj._frozen:
            raise RuntimeError("Loaded O_GT was not frozen before saving.")
        return obj
