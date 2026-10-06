"""Non-dominated sorting and hypervolume computation.

Objective count (3 vs. 4) is decided in Step 1 based on f_syn variance
(§5.6) -- tropism is included only if a genuine dataset is confirmed,
otherwise reported separately as a heuristic filter, never inside the
primary hypervolume. Reference point and normalization are fixed and
logged BEFORE any experiment runs (§5.1) -- never tuned post hoc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a dominates b under maximization on all objectives."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(points: np.ndarray) -> np.ndarray:
    """Return indices of the non-dominated (Pareto-optimal) points, under
    maximization on every objective (sign-adjust e.g. -toxicity beforehand)."""
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and keep[j] and dominates(points[j], points[i]):
                keep[i] = False
                break
    return np.where(keep)[0]


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume dominated by `front` relative to `ref_point`, maximization
    on every axis. Exact O(n log n) sweep for d == 2 (the common case here:
    potency, -toxicity); Monte-Carlo estimate for d >= 3 to avoid the
    exponential-blowup and floating-point cancellation failure mode of a
    naive inclusion-exclusion implementation (an earlier version of this
    function used inclusion-exclusion for all d and both grew intractably
    slow and produced non-monotonic HV estimates as the archive grew, due
    to catastrophic cancellation in the alternating sum)."""
    front = np.atleast_2d(front)
    if front.size == 0:
        return 0.0
    d = front.shape[1]
    if d == 2:
        return _hypervolume_2d(front, ref_point)
    return _hypervolume_mc(front, ref_point)


def _hypervolume_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 2D hypervolume via the standard sort-and-sweep algorithm."""
    pts = front[np.all(front > ref_point, axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    total = 0.0
    prev_y = ref_point[1]
    for x, y in pts:
        if y > prev_y:
            total += (x - ref_point[0]) * (y - prev_y)
            prev_y = y
    return float(total)


def _hypervolume_mc(front: np.ndarray, ref_point: np.ndarray, n_samples: int = 20000,
                     random_state: int = 0) -> float:
    rng = np.random.default_rng(random_state)
    upper = front.max(axis=0)
    box_vol = np.prod(np.maximum(upper - ref_point, 1e-12))
    samples = rng.uniform(ref_point, upper, size=(n_samples, front.shape[1]))
    dominated = np.any(np.all(front[:, None, :] >= samples[None, :, :], axis=2), axis=0)
    return float(box_vol * dominated.mean())


@dataclass
class ParetoArchive:
    objective_names: list[str]
    reference_point: np.ndarray
    points: list[np.ndarray] = field(default_factory=list)
    fragments: list = field(default_factory=list)
    hypervolume_history: list[float] = field(default_factory=list)

    def update(self, new_points: np.ndarray, new_fragments: list) -> None:
        if self.points:
            all_points = np.vstack([np.stack(self.points), new_points])
            all_fragments = self.fragments + new_fragments
        else:
            all_points = new_points
            all_fragments = new_fragments
        keep_idx = non_dominated_sort(all_points)
        self.points = [all_points[i] for i in keep_idx]
        self.fragments = [all_fragments[i] for i in keep_idx]
        hv = hypervolume(np.stack(self.points), self.reference_point)
        # Monotonicity check: archive HV must never decrease round over round
        # (research plan §6.1 item 28).
        if self.hypervolume_history and hv < self.hypervolume_history[-1] - 1e-9:
            raise RuntimeError(f"Archive hypervolume decreased: {self.hypervolume_history[-1]} -> {hv}")
        self.hypervolume_history.append(hv)

    @property
    def current_hypervolume(self) -> float:
        return self.hypervolume_history[-1] if self.hypervolume_history else 0.0
