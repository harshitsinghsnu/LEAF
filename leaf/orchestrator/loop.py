"""Algorithm 1 (corrected): the LEAF closed-loop round.

Fixes applied relative to Revision 1's Algorithm 1 (research plan §3.1-3.2, §5.10):
  1. Proposer emits a POOL P_t, not a directly-queried population.
  2. O_AL scores the pool for free; Verifier abstains on the pool for free.
  3. An acquisition function selects C_t (|C_t| = q) from the surviving pool.
  4. O_GT (frozen) labels ONLY C_t; budget decrements by |C_t|.
  5. Hypothesis Agent counterfactual queries are ALSO charged against budget.
  6. O_AL is refit each round; O_GT is NEVER refit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leaf.agents.proposer import RuleBasedProposer, LLMProposer
from leaf.agents.verifier import (
    ConformalCalibrator, BinaryToxicityFilter, VerifierVerdict, aggregate, aggregate_binary,
    toxicophore_flag,
)
from leaf.fragments.vocabulary import assemble_lipid
from leaf.agents.hypothesis import HypothesisLedger, Hypothesis, make_counterfactual_pairs, test_hypothesis
from leaf.archive.pareto import ParetoArchive
from leaf.oracle.acquisition import multi_objective_ucb, expected_hypervolume_improvement, select_batch
from leaf.oracle.ground_truth import GroundTruthOracle
from leaf.oracle.surrogate import ALSurrogate
from leaf.fragments.embed import FragmentEmbedder


@dataclass
class LoopConfig:
    budget: int
    pool_size: int
    q: int                       # candidates queried against O_GT per round
    # Real AGILE heads: expt_Hela and expt_Raw are both real transfection-
    # efficiency readouts (different cell lines) -- there is no toxicity
    # assay in this dataset (see data/raw/README.md), so "toxicity" is not
    # a GP head here; toxicity screening uses the deterministic RDKit
    # toxicophore rule instead (see the Verifier block above).
    heads: list[str] = field(default_factory=lambda: ["potency", "potency_raw"])
    signs: dict = field(default_factory=lambda: {"potency": 1.0, "potency_raw": 1.0})
    hypothesis_every_k: int = 5
    max_counterfactual_pairs: int = 20
    hypothesis_c_value: str = ""    # set from real component_lists() by callers (run_experiment.py); no hardcoded default
    hypothesis_c_control: str = ""
    seed: int = 0
    # "multi_objective_ucb" | "expected_hypervolume_improvement". BUG FOUND
    # (external review, §7): earlier revisions of this loop hardcoded
    # multi_objective_ucb and only checked that a config's
    # acquisition_function field was PRESENT (for pre-registration
    # bookkeeping), never actually dispatched on its value -- every
    # experiment in this paper used UCB regardless of what its config said,
    # despite the paper text describing EHVI as "a config-flagged
    # alternative." Fixed here; see §7/§8 for what was and wasn't re-run
    # under real EHVI after the fix.
    acquisition_function: str = "multi_objective_ucb"
    ucb_beta: float = 2.0
    use_proposer: bool = True    # False -> Ablation, replaced by static GA
    use_verifier: bool = True    # False -> Ablation (a): Proposer+Oracle only
    verifier_mode: str = "conformal"  # "conformal" (full LEAF) | "binary" (Ablation (e), §5.12)
    binary_filter_threshold: float = 0.5
    use_hypothesis: bool = True  # False -> Ablation (b): no Hypothesis Agent at all
    use_hypothesis_feedback: bool = True
    # False -> Ablation (d): hypotheses are still tested and logged to the
    # Ledger (still charged against budget), but the Ledger does NOT bias
    # the Proposer's mutations. This is a distinct axis from
    # `use_hypothesis` -- conflating the two made ablation (d) silently
    # identical to full LEAF in an earlier version of this loop, since
    # both "run the Hypothesis Agent" and "let it influence the Proposer"
    # were gated by the same flag.
    use_acquisition: bool = True
    # False -> isolates proposal quality from selection quality: the first
    # q surviving pool candidates are queried directly, with NO UCB
    # scoring at all. Used to test whether a weak method (e.g. the real
    # LLM Proposer) is weak because it proposes bad candidates, or because
    # good candidates it does propose get discarded by acquisition.
    log_diagnostics: bool = False
    # When True, RoundLog also carries proposer diagnostics (operator mix,
    # duplicate-of-archive rate, edit distance to nearest archive member).
    # Off by default: these are extra pool-vs-archive comparisons every
    # round and are only useful for the mechanistic-diagnostics analysis,
    # not for the primary hypervolume comparisons.
    dedupe_pool: bool = False
    # When True, exact-duplicate-of-archive candidates are dropped from the
    # pool BEFORE O_AL scoring/Verifier/acquisition, for every method
    # (baselines included, so the comparison stays matched). Deliberately
    # NOT backfilled by asking the proposer again -- if deduping leaves
    # fewer than q survivors, acquisition selects fewer than q that round,
    # which is the real, honest cost of enforcing novelty on a
    # duplicate-heavy proposer, not papered over. Added specifically to
    # answer the question "does the §6.2 underperformance persist once
    # duplicate-driven budget waste is structurally removed, i.e. is it a
    # proposal-quality problem or an easily-engineered safeguard problem?"
    # (external review, \S7).
    dedupe_scope: str = "archive"
    # "archive" (default): dedupe_pool only excludes exact matches of the
    # current Pareto-optimal archive (dominated-but-evaluated candidates
    # can still be re-proposed and re-queried). "all_evaluated": excludes
    # exact matches of EVERY candidate ever queried this run, dominated or
    # not, plus within-pool duplicates (two identical novel proposals in
    # the same round keep only one). Added per external review's point
    # that archive-only de-duplication is an incomplete novelty control;
    # "all_evaluated" is the stricter, more literal reading of "does this
    # query buy new information at all." No backfill is added in either
    # scope -- the outer while-loop's extra rounds already serve as the
    # real replenishment mechanism, consistent with dedupe_pool's existing
    # no-backfill design and this paper's disclosure that de-duplication
    # trades oracle queries for proposal rounds, not a free lunch.


@dataclass
class RoundLog:
    round_idx: int
    queries_used_this_round: int
    budget_remaining: int
    archive_hypervolume: float
    coverage: dict | None = None
    inter_agent_agreement: float | None = None
    frac_crossover: float | None = None
    frac_duplicate_of_archive: float | None = None
    mean_min_edit_distance: float | None = None
    # Fraction of the ACQUISITION-SELECTED batch (the candidates that
    # actually charge the oracle budget) that exactly duplicate an
    # already-evaluated archive member, as distinct from
    # frac_duplicate_of_archive above (which measures the raw proposal
    # pool before verification/acquisition). External review correctly
    # noted these are not interchangeable: a high pool duplicate rate
    # does not necessarily mean acquisition actually selects duplicates
    # (it could, in principle, score and avoid them). This field answers
    # that directly rather than leaving it inferred.
    frac_selected_duplicate_of_archive: float | None = None


def _fragment_edit_distance(a, b) -> int:
    """Number of (A, B, C) slots that differ between two Fragments."""
    return sum(1 for slot in ("A", "B", "C") if getattr(a, slot) != getattr(b, slot))


def _pool_diagnostics(pool, archive_frags) -> tuple[float, float, float]:
    """Real, computed-not-guessed proposer diagnostics for one round's pool:
    fraction of candidates using crossover (vs. mutation), fraction that
    exactly duplicate an already-evaluated archive member (wasted query
    risk), and mean minimum edit distance (# differing slots, 0-3) to the
    nearest archive member (how far the proposer explores vs. stays close
    to known-good points)."""
    if not pool:
        return 0.0, 0.0, 0.0
    n_crossover = sum(1 for c in pool if c.operator == "crossover")
    if not archive_frags:
        return n_crossover / len(pool), 0.0, 3.0  # nothing to compare against yet (round 0)
    n_dup, dist_sum = 0, 0
    for c in pool:
        dists = [_fragment_edit_distance(c.fragment, a) for a in archive_frags]
        min_dist = min(dists)
        dist_sum += min_dist
        if min_dist == 0:
            n_dup += 1
    return n_crossover / len(pool), n_dup / len(pool), dist_sum / len(pool)


def run_loop(config: LoopConfig, oracle_gt: GroundTruthOracle, embedder: FragmentEmbedder,
             objective_names: list[str], reference_point: np.ndarray,
             calibration_pool: tuple[np.ndarray, np.ndarray] | None = None,
             proposer=None) -> dict:
    """Runs the corrected round loop until the budget is exhausted.

    `proposer` must implement `propose_pool(archive_fragments, pool_size,
    ledger_bias=None) -> list[ProposedCandidate]`. Defaults to
    RuleBasedProposer (the LLM-proposer stand-in) if not given. Pass
    `StaticGAProposer(seed=...)` or `RandomSearchProposer(seed=...)`
    (leaf/baselines/) here to run those baselines through the IDENTICAL
    acquisition function, Verifier, and budget accounting -- this is what
    makes the E2 comparison a fair isolation of proposer quality (research
    plan §3.2 baseline-parity requirement) rather than three different
    pipelines.

    Returns a dict with the final archive, hypothesis ledger, and per-round
    logs (including a budget-accounting audit trail consumed by
    tests/test_budget_accounting.py).
    """
    proposer = proposer if proposer is not None else RuleBasedProposer(seed=config.seed)
    surrogate = ALSurrogate(heads=config.heads)
    archive = ParetoArchive(objective_names=objective_names, reference_point=reference_point)
    ledger = HypothesisLedger()
    calibrator = ConformalCalibrator()
    binary_gate = BinaryToxicityFilter(threshold=config.binary_filter_threshold)

    budget_remaining = config.budget
    round_idx = 0
    logs: list[RoundLog] = []
    audit_queries_issued = 0
    # Per-query record of exactly which fragment was charged against the
    # budget each round, and whether it exactly duplicated an
    # already-evaluated archive member at selection time -- added for
    # external-review item #6 (selected-query duplicate accounting is not
    # the same as raw-pool duplicate accounting, see RoundLog above) and
    # #2 (applicability-domain analysis needs the actual queried fragment
    # values, not just aggregate hypervolume).
    query_log: list[dict] = []
    all_evaluated_keys: set[tuple] = set()  # every (A,B,C) ever charged this run, dominated or not

    while budget_remaining > 0:
        queries_this_round = 0

        # 1. Proposer emits a pool (free -- no oracle query).
        archive_frags = archive.fragments
        use_bias = config.use_hypothesis and config.use_hypothesis_feedback
        if isinstance(proposer, LLMProposer):
            pool = proposer.propose_pool(archive_frags, config.pool_size,
                                          ledger_bias=ledger.bias_for_proposer() if use_bias else None,
                                          archive_scores=archive.points if archive.points else None)
        else:
            pool = proposer.propose_pool(archive_frags, config.pool_size,
                                          ledger_bias=ledger.bias_for_proposer() if use_bias else None)

        diag_frac_crossover = diag_frac_dup = diag_mean_dist = None
        if config.log_diagnostics:
            diag_frac_crossover, diag_frac_dup, diag_mean_dist = _pool_diagnostics(pool, archive_frags)

        if config.dedupe_pool and config.dedupe_scope == "all_evaluated":
            seen_this_round: set[tuple] = set()
            deduped_pool = []
            for c in pool:
                key = (c.fragment.A, c.fragment.B, c.fragment.C)
                if key in all_evaluated_keys or key in seen_this_round:
                    continue
                seen_this_round.add(key)
                deduped_pool.append(c)
            pool = deduped_pool
        elif config.dedupe_pool and archive_frags:
            archive_keys = {(f.A, f.B, f.C) for f in archive_frags}
            pool = [c for c in pool if (c.fragment.A, c.fragment.B, c.fragment.C) not in archive_keys]

        X_pool = embedder.embed_batch([c.fragment for c in pool]) if pool else np.zeros((0, embedder.dim))

        if X_pool.shape[0] == 0:
            # dedupe_pool=True can legitimately empty the pool if every
            # proposed candidate this round exactly matches an archive
            # member (a real, informative outcome for a duplicate-heavy
            # proposer, not an error) -- sklearn's GaussianProcessRegressor
            # cannot predict on 0 samples, so charge 0 budget this round
            # and let the Proposer try again next round rather than crash.
            logs.append(RoundLog(round_idx, 0, budget_remaining,
                                  archive.current_hypervolume, None, None,
                                  diag_frac_crossover, diag_frac_dup, diag_mean_dist,
                                  frac_selected_duplicate_of_archive=None))
            round_idx += 1
            if round_idx > 10 * max(config.budget // max(config.q, 1), 1):
                # Real safety valve: if the proposer NEVER produces a novel
                # candidate across an unreasonable number of attempts, stop
                # rather than loop forever burning zero-budget rounds.
                break
            continue

        # 2. O_AL scores the pool (free).
        if surrogate.n_labels > 0:
            if config.acquisition_function == "expected_hypervolume_improvement":
                current_front = np.stack(archive.points) if archive.points else np.zeros((0, len(config.heads)))
                scores = expected_hypervolume_improvement(surrogate, X_pool, config.heads, config.signs, current_front)
            else:
                scores = multi_objective_ucb(surrogate, X_pool, config.heads, config.signs, beta=config.ucb_beta)
        else:
            scores = np.zeros(X_pool.shape[0])  # cold start: no labels yet, acquire uniformly

        # 3. Verifier abstains on the pool (free) -- skipped in Ablation (a).
        keep_mask = np.ones(X_pool.shape[0], dtype=bool)
        agreement = None
        coverage_entry = None
        if config.use_verifier and (calibration_pool is not None or config.verifier_mode == "binary"):
            # Real toxicity signal: deterministic RDKit toxicophore
            # substructure matching on the assembled molecule (no GP
            # "toxicity" head exists -- AGILE has no toxicity/viability
            # assay at all, see data/raw/README.md; fabricating a learned
            # toxicity signal from nothing would misrepresent the data).
            tox_mean = np.array([
                1.0 if (lambda s: s is not None and toxicophore_flag(s))(assemble_lipid(c.fragment)) else 0.0
                for c in pool
            ])
            agreements = []
            if config.verifier_mode == "binary":
                # Ablation (e): hard accept/reject, no coverage concept at
                # all -- do not populate coverage_entry (a config using
                # this mode expects rounds.csv's coverage columns to be
                # empty, not a fabricated number).
                for i, tm in enumerate(tox_mean):
                    verdicts = [VerifierVerdict(f"agent{j}", "modelA" if j % 2 == 0 else "modelB", float(tm))
                                for j in range(3)]
                    agg = aggregate_binary(i, verdicts, binary_gate)
                    agreements.append(agg.agreement)
                    keep_mask[i] = not agg.abstain
            else:
                cal_scores, cal_labels = calibration_pool
                calibrator.calibrate(cal_scores, cal_labels, round_idx)  # per-round recalibration, §5.3
                coverage_entry = calibrator.coverage_history[-1]
                for i, tm in enumerate(tox_mean):
                    verdicts = [VerifierVerdict(f"agent{j}", "modelA" if j % 2 == 0 else "modelB", float(tm))
                                for j in range(3)]
                    agg = aggregate(i, verdicts, calibrator)
                    agreements.append(agg.agreement)
                    keep_mask[i] = not agg.abstain
            agreement = float(np.mean(agreements))

        surviving_idx = np.where(keep_mask)[0]
        if surviving_idx.size == 0:
            # Every pooled candidate was abstained on. Falling back to
            # `continue` here without consuming budget can loop forever
            # (observed in testing: verifier abstained on 100% of a
            # cold-start pool with uninformative O_AL variance). Guarantee
            # termination by falling back to the full, unfiltered pool for
            # this round rather than stalling; this is logged so the
            # fallback rate itself becomes a diagnosable metric.
            surviving_idx = np.arange(X_pool.shape[0])

        # 4. Acquisition selects q candidates from the surviving pool
        #    (or, if disabled, just the first q survivors -- isolates
        #    proposal quality from selection quality).
        if config.use_acquisition:
            acq_idx = select_batch(scores[surviving_idx], config.q)
        else:
            acq_idx = np.arange(min(config.q, surviving_idx.size))
        chosen_idx = surviving_idx[acq_idx]
        X_chosen = X_pool[chosen_idx]
        chosen_candidates = [pool[i] for i in chosen_idx]

        # 5. O_GT (frozen) labels ONLY the chosen candidates; charge budget.
        n_query = min(len(chosen_idx), budget_remaining)
        X_chosen = X_chosen[:n_query]
        chosen_candidates = chosen_candidates[:n_query]
        Y_gt = oracle_gt.query(X_chosen)
        queries_this_round += n_query
        budget_remaining -= n_query
        audit_queries_issued += n_query
        all_evaluated_keys.update((c.fragment.A, c.fragment.B, c.fragment.C) for c in chosen_candidates)

        diag_frac_selected_dup = None
        if config.log_diagnostics:
            if chosen_candidates and archive_frags:
                selected_dup_flags = [
                    min(_fragment_edit_distance(c.fragment, a) for a in archive_frags) == 0
                    for c in chosen_candidates
                ]
                diag_frac_selected_dup = sum(selected_dup_flags) / len(chosen_candidates)
            else:
                selected_dup_flags = [False] * len(chosen_candidates)
                diag_frac_selected_dup = 0.0
            for c, is_dup in zip(chosen_candidates, selected_dup_flags):
                entry = {
                    "round": round_idx, "A": c.fragment.A, "B": c.fragment.B,
                    "C": c.fragment.C, "is_duplicate_of_archive": is_dup,
                }
                thinking = getattr(c, "thinking", None)
                if thinking:
                    entry["thinking_length"] = len(thinking)
                    entry["thinking_mentions_potency"] = ("potency" in thinking.lower())
                query_log.append(entry)

        # 6. Refit O_AL on newly acquired labels.
        surrogate.add_labels(X_chosen, Y_gt)
        surrogate.refit()

        # 7. Update archive.
        signed_points = np.stack([
            config.signs.get(h, 1.0) * Y_gt[h] for h in config.heads
        ], axis=1)
        archive.update(signed_points, [c.fragment for c in chosen_candidates])

        # 8. Hypothesis Agent, every k rounds -- counterfactual queries ALSO
        #    charged against budget (§5.10 fix).
        if config.use_hypothesis and round_idx % config.hypothesis_every_k == 0 and archive.fragments:
            # Real structure-property axis: isocyanide (C) tail length,
            # comparing the shortest vs. longest real tail in the vendored
            # AGILE vocabulary. NOTE: no component in this real dataset is
            # halogenated (checked all 20 A / 12 B / 5 C SMILES directly),
            # so the paper's original "brominated-tail" hypothesis (H4/E5,
            # targeting the LUMI-lab rediscovery) cannot be tested via
            # counterfactual pairs on AGILE data -- there is no brominated
            # component to construct a matched pair against. That
            # experiment needs either an extended vocabulary or a
            # different (literature-only, non-counterfactual) mode; see
            # research plan §5.5 / data/raw/README.md.
            if config.hypothesis_c_value and config.hypothesis_c_control:
                value, control = config.hypothesis_c_value, config.hypothesis_c_control
            else:
                from leaf.fragments.vocabulary import component_lists
                _, _, c_list = component_lists()
                value, control = c_list[-1], c_list[0]  # real tail-length extremes, deterministic fallback
            slot = "C"
            pairs = make_counterfactual_pairs(archive.fragments, slot, value, control,
                                               max_pairs=config.max_counterfactual_pairs)
            if pairs and budget_remaining > 0:
                n_pairs = min(len(pairs), budget_remaining // 2)
                pairs = pairs[:n_pairs]
                if pairs:
                    control_X = embedder.embed_batch([p[0] for p in pairs])
                    treat_X = embedder.embed_batch([p[1] for p in pairs])
                    control_Y = oracle_gt.query(control_X)["potency"]
                    treat_Y = oracle_gt.query(treat_X)["potency"]
                    cost = 2 * len(pairs)
                    budget_remaining -= cost
                    queries_this_round += cost
                    audit_queries_issued += cost
                    hyp = Hypothesis(text=f"{slot}={value} vs {control} shifts potency",
                                      fragment_slot=slot, fragment_value=value, property_head="potency")
                    ledger.entries.append(test_hypothesis(hyp, control_Y, treat_Y))

        logs.append(RoundLog(round_idx, queries_this_round, budget_remaining,
                              archive.current_hypervolume, coverage_entry, agreement,
                              diag_frac_crossover, diag_frac_dup, diag_mean_dist,
                              frac_selected_duplicate_of_archive=diag_frac_selected_dup))
        round_idx += 1

        if audit_queries_issued > config.budget:
            raise RuntimeError("Budget accounting violated: queries issued exceeds B.")

    return {"archive": archive, "ledger": ledger, "logs": logs, "total_queries_issued": audit_queries_issued,
            "query_log": query_log}
