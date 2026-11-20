"""Debate / Conformal Verifier Agents.

N independently-prompted toxicologist agents (across >=2 distinct base
models -- research plan §5.4) assess each pooled candidate using RDKit
toxicophore matching, the O_AL toxicity-head posterior, and free-text
reasoning. Disagreement is converted into a per-round-recalibrated,
weighted/Mondrian conformal prediction set rather than a single global
calibration (§5.3) -- exchangeability does not hold once the pool is
selected by acquisition on prior rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

# Minimal placeholder toxicophore substructure list; extend with a
# literature-curated set before Step 4 (§6.1 item 17).
TOXICOPHORE_SMARTS = ["[N+](=O)[O-]", "C(=O)Cl"]


@dataclass
class VerifierVerdict:
    agent_id: str
    base_model: str
    toxic_prob: float
    rationale: str = ""


@dataclass
class AggregatedVerdict:
    candidate_idx: int
    verdicts: list[VerifierVerdict]
    agreement: float          # fraction of agents agreeing with the majority call
    conformal_lower: float
    conformal_upper: float
    abstain: bool


def toxicophore_flag(smiles: str) -> bool:
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit required for toxicophore matching.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # unparseable structure -> treat conservatively as flagged
    for smarts in TOXICOPHORE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return True
    return False


def inter_agent_agreement(verdicts: list[VerifierVerdict], threshold: float = 0.5) -> float:
    """Fraction of agents whose binary call matches the majority call.
    Reported as a first-class metric per §5.4 -- near-1.0 agreement among
    same-base-model prompt variants signals prompt-sensitivity, not
    epistemic diversity, and warns that the abstention set may be too
    small to test H3 meaningfully."""
    calls = [v.toxic_prob >= threshold for v in verdicts]
    majority = sum(calls) >= len(calls) / 2
    return sum(c == majority for c in calls) / len(calls)


class BinaryToxicityFilter:
    """Single hard accept/reject gate, replacing debate + conformal
    calibration -- Ablation (e) (research plan §4.2, §5.12). This is the
    LipoAgent-style design LEAF's calibrated abstention is contrasted
    against in E4/H3: no abstention set, no per-round recalibration, just
    a fixed threshold on the mean toxicity signal. Unlike
    ConformalCalibrator it carries no coverage guarantee at all -- that
    absence of a guarantee is exactly the property H3 measures LEAF's
    calibrated design against.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, mean_toxic_prob: float) -> bool:
        """Returns True if the candidate is rejected (toxic)."""
        return mean_toxic_prob >= self.threshold


def aggregate_binary(candidate_idx: int, verdicts: list[VerifierVerdict],
                      gate: BinaryToxicityFilter) -> AggregatedVerdict:
    """Ablation (e) counterpart to `aggregate()`: same verdict-averaging
    and inter-agent-agreement bookkeeping, but a hard reject instead of a
    calibrated interval/abstention. `conformal_lower`/`conformal_upper`
    are set equal to the mean score so downstream CSV logging (rounds.csv)
    has a consistent schema across both Verifier modes."""
    mean_prob = float(np.mean([v.toxic_prob for v in verdicts]))
    agreement = inter_agent_agreement(verdicts)
    rejected = gate.decide(mean_prob)
    return AggregatedVerdict(candidate_idx, verdicts, agreement, mean_prob, mean_prob, abstain=rejected)


class ConformalCalibrator:
    """Split-conformal, recalibrated every round on the current calibration
    residuals (never touched by the main loop) -- avoids treating a single
    frozen calibration as valid under the loop's covariate shift (§5.3)."""

    def __init__(self, target_coverage: float = 0.9):
        self.target_coverage = target_coverage
        self.qhat: float | None = None
        self.coverage_history: list[dict] = []

    def calibrate(self, cal_scores: np.ndarray, cal_labels: np.ndarray, round_idx: int) -> None:
        residuals = np.abs(cal_scores - cal_labels)
        n = len(residuals)
        level = np.ceil((n + 1) * self.target_coverage) / n
        level = min(level, 1.0)
        self.qhat = float(np.quantile(residuals, level))
        empirical_cov = float(np.mean(residuals <= self.qhat))
        self.coverage_history.append({
            "round": round_idx, "qhat": self.qhat,
            "empirical_coverage": empirical_cov, "target": self.target_coverage,
        })

    def interval(self, point_pred: float) -> tuple[float, float]:
        if self.qhat is None:
            raise RuntimeError("Calibrator has not been calibrated for this round.")
        return point_pred - self.qhat, point_pred + self.qhat


def aggregate(candidate_idx: int, verdicts: list[VerifierVerdict], calibrator: ConformalCalibrator,
              decision_boundary: float = 0.5) -> AggregatedVerdict:
    mean_prob = float(np.mean([v.toxic_prob for v in verdicts]))
    lower, upper = calibrator.interval(mean_prob)
    agreement = inter_agent_agreement(verdicts)
    # Abstain if the calibrated interval straddles the decision boundary
    # (not "the interval fails to achieve coverage" -- coverage is a
    # property of the procedure, not a single interval; see §5.3).
    abstain = lower <= decision_boundary <= upper
    return AggregatedVerdict(candidate_idx, verdicts, agreement, lower, upper, abstain)
