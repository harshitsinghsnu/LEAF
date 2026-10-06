"""Multi-head GP with a continuous Tanimoto kernel.

k(x, x') = x.x' / (||x||^2 + ||x'||^2 - x.x')

Shared architecture used by both the frozen ground-truth oracle (O_GT) and
the per-round-refit active-learning surrogate (O_AL) -- see oracle/ground_truth.py
and oracle/surrogate.py. This module only implements the kernel + a thin
sklearn GaussianProcessRegressor wrapper per head so it stays runnable
without a GPU / GPyTorch dependency.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter


class TanimotoKernel(Kernel):
    """Continuous Tanimoto kernel, eq. (1) in the LEAF paper."""

    def __init__(self, variance: float = 1.0):
        self.variance = variance

    @property
    def hyperparameter_variance(self):
        return Hyperparameter("variance", "numeric", (1e-3, 1e3))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        xx = np.sum(X ** 2, axis=1)[:, None]
        yy = np.sum(Y ** 2, axis=1)[None, :]
        xy = X @ Y.T
        denom = xx + yy - xy
        denom = np.where(denom <= 0, 1e-12, denom)
        K = self.variance * (xy / denom)
        if eval_gradient:
            # sklearn optimizes kernel hyperparameters in log-space
            # (theta = log(variance)), so the gradient LBFGS needs is
            # dK/d(theta), not dK/d(variance). K is linear in variance
            # (K = variance * xy/denom), so dK/d(variance) = K/variance,
            # and by the chain rule dK/d(theta) = variance * dK/d(variance)
            # = K. An earlier version of this returned K/variance instead
            # of K -- wrong by a factor of variance^2, which produced a
            # degenerate optimization landscape (variance could run away
            # towards its 1e-3/1e3 bounds), generating NaN/Inf that
            # crashed scipy's Fortran-level L-BFGS-B routine outright on
            # Windows (a hard "Windows fatal exception", not a catchable
            # Python error) rather than just failing to converge cleanly.
            return K, K[:, :, None]
        return K

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def is_stationary(self):
        return False


class MultiHeadGP:
    """One independent Tanimoto-kernel GP per objective head.

    heads: e.g. ["potency", "toxicity"] (tropism only included if a genuine
    organ-selectivity dataset is confirmed -- see research plan §5.6/§5.9).
    """

    def __init__(self, heads: list[str], alpha: float = 1e-6, random_state: int = 0):
        self.heads = list(heads)
        self.models = {
            h: GaussianProcessRegressor(kernel=TanimotoKernel(), alpha=alpha,
                                         normalize_y=True, random_state=random_state)
            for h in heads
        }
        self._fitted = set()

    def fit(self, X: np.ndarray, Y: dict[str, np.ndarray]) -> "MultiHeadGP":
        # TanimotoKernel has exactly one hyperparameter (a scale factor);
        # against near-uninformative labels (e.g. the synthetic placeholder
        # data used before real harmonized labels land, research plan §5.8)
        # its LBFGS optimization has nothing meaningful to converge to and
        # emits an ABNORMAL_TERMINATION_IN_LNSRCH ConvergenceWarning on
        # every fit. sklearn already falls back to its best iterate, so the
        # fit is still usable -- this only silences a cosmetic warning that
        # would otherwise drown out actual run output across hundreds of
        # per-round refits. It does NOT mask a data problem: real signal
        # quality is checked by the E1 go/no-go gate (§5.11), not by
        # whether the optimizer's line search converged.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for h in self.heads:
                if h not in Y:
                    continue
                self.models[h].fit(X, Y[h])
                self._fitted.add(h)
        return self

    def predict(self, X: np.ndarray, head: str, return_std: bool = True):
        if head not in self._fitted:
            raise RuntimeError(f"Head '{head}' has not been fit yet.")
        return self.models[head].predict(X, return_std=return_std)

    def predict_all(self, X: np.ndarray, return_std: bool = True) -> dict:
        return {h: self.predict(X, h, return_std=return_std) for h in self._fitted}
