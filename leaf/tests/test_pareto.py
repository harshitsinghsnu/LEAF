"""Non-dominated sort + hypervolume monotonicity (research plan §6.1 item 28)."""
import numpy as np

from leaf.archive.pareto import non_dominated_sort, hypervolume, ParetoArchive


def test_non_dominated_sort_basic():
    points = np.array([[1, 1], [2, 2], [0, 0], [2, 1]])
    idx = non_dominated_sort(points)
    # (2,2) dominates every other point here, including (2,1); only (2,2) survives.
    assert set(idx.tolist()) == {1}


def test_hypervolume_positive_for_nonempty_front():
    front = np.array([[2, 2], [3, 1]])
    ref = np.array([0.0, 0.0])
    assert hypervolume(front, ref) > 0


def test_archive_update_never_decreases_hypervolume():
    archive = ParetoArchive(objective_names=["a", "b"], reference_point=np.array([0.0, 0.0]))
    archive.update(np.array([[1.0, 1.0]]), ["fragA"])
    archive.update(np.array([[0.5, 0.5]]), ["fragB"])  # dominated, should not lower HV
    assert archive.hypervolume_history[-1] >= archive.hypervolume_history[0]
