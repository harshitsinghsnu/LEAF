# Response to External Review (round 3: confound analysis, fairness questions)

## 1. The core confound: does v3_scored's gain reflect memorization or genuine use of injected scores?

**Status: addressed with two new real experiments, both now in the paper (§3.3, Appendix G).**

- **Leakage probe.** `llama3.1:8b` (the exact Proposer model), zero-shot, temperature 0, bare SMILES only — no archive, no vocabulary, no scores — was asked to score 50 real held-out random-split AGILE lipids' transfection efficiency. It returned the **identical score (8.0/10) for all 50 structurally distinct molecules**. This is a degenerate, non-discriminating output — stronger evidence against the memorization explanation than a low-but-nonzero correlation would have been, since the model has *no* exploitable zero-shot signal about real potency to reactivate.
- **Rank-only prompt (v4_rank).** A new prompt variant shows only each sampled archive member's relative rank (1 = best), never raw values (verified directly that no magnitude leaks into the rendered prompt). At n=4, random split, matching v3_scored's seeds exactly: median +17.2 vs. static-GA and −17.7 vs. random search — **matching or exceeding** v3_scored's raw-values result (+13.9 / −32.7). Relative preference alone accounts for at least as much of the effect as exact magnitudes.

**Conclusion:** the data weighs against both alternative explanations you raised (memorization, magnitude-anchoring). The more likely account is that the model is using the injected signal as a genuine preference ordering — though with n=4 throughout, this remains directional, not proven, and a full n≥8 sweep of both variants is the natural next step (not yet run, given the per-seed compute cost already established in this project — dedup checks at n=8 took up to ~10 hours for one split).

One honest caveat volunteered rather than hidden: seed 1 has now independently produced an unusually strong candidate under three different conditions (stricter dedup, v3_scored, v4_rank) — 232.263, 232.263, and 227.271 respectively. We verified via raw per-query fragment logs that these are not a caching bug (the actual search trajectories differ), but this does mean seed 1 has outsized, non-independent leverage across these n=4 results, and they should be read with that in mind.

## 2. "Is this a lot to ask of an LLM?"

**We agree with the review's framing, and it should be stated more explicitly in the paper than it currently is** (see change #1 in the chat summary). To restate LEAF's actual setup against this concern:

- The Proposer is genuinely zero-shot: no fine-tuning, no RL/reward signal, no persistent memory across rounds beyond what's re-stuffed into the prompt (Appendix E confirms single-turn, stateless calls).
- It must reproduce SMILES strings character-for-character while also being asked to reason about recombination — a real tension between precise instruction-following and exploration, and the paper's own diagnostic data (67-72% archive-duplicate rate for the two ≤8B models) is consistent with the model defaulting to safe, low-risk copying under that tension.
- Static-GA is a fixed, near-zero-cost, purpose-built local-search operator; asking a general-purpose LLM to beat it with no domain fine-tuning, no tool access, and (in the default prompt) no explicit objective description at all is a demanding bar.

The paper's own evidence supports this reading directly: the single intervention that moved medians the most (giving the LLM real scores, then just rank) is exactly a "give it more information" fix, not an appeal to different reasoning capacity — consistent with "the deficit is partly a prompt-information problem, not a fundamental LLM-reasoning problem" being the fairer conclusion than a blanket "LLMs can't do chemistry" framing, which the paper (correctly) never claims but also doesn't state as clearly as it now can, given the v3_scored/v4_rank results.

## 3. Standard ways to merge candidates before acquisition — does LEAF's GA match the field?

**Mostly yes, with one honest gap worth naming.** Checking the review's description against LEAF's actual, real implementation (`leaf/baselines/static_ga.py`, Table 4/Appendix D):

- LEAF's static-GA does exactly the "uniform/single-point crossover on structured slots" pattern the review names as standard: each of the three fragment slots (A/B/C) is copied from one of two sampled parents, independently, then one slot is mutated with probability 0.5.
- This is a legitimate, if intentionally simple, instantiation of the standard toolkit — LEAF's contribution is isolating the proposer, not building a state-of-the-art GA, so a textbook operator set is the right choice for a clean baseline.
- **The gap the review implicitly surfaces:** LEAF's GA selects its two parents **uniformly at random without replacement** from the archive, not via fitness-weighted or Pareto-front-biased selection, which is the more common selection pressure in molecular GA implementations (e.g., preferentially sampling from the non-dominated front). This is a real, disclosable simplification — it makes LEAF's GA baseline weaker on the selection-pressure axis than some published molecular GAs, which if anything makes the LLM Proposer's failure to beat it a *more*, not less, striking result. This should be added as an explicit limitation (see chat summary #2).

## 4. Is comparing across a fixed acquisition function fair?

**Yes, for the question the paper actually asks — but there's a real, distinct question it doesn't answer.**

- Holding UCB (β=2) fixed across every proposer is the correct controlled-variable design for isolating "does proposer identity matter," and the paper is explicit about why (§2.2: "All methods share acquisition and q").
- The genuinely open question the review raises is different: **does the interaction between proposer and acquisition function matter** — specifically, would a novelty-rewarding acquisition function like EHVI treat the LLM's low-diversity, high-duplication proposals differently than UCB does (which has no novelty term)? The paper's existing EHVI check (n=4, one proposer, one split) tests "does switching acquisition change the LLM's own result" — it does **not** test "does the LLM vs. GA gap change under EHVI," because GA/random were not re-run under EHVI in that check.
- This is a real, currently-unrun experiment: a 2×2 (UCB vs. EHVI) × (LLM vs. GA) design on the same split/seeds, specifically to test the interaction rather than just the LLM's own acquisition sensitivity. Recommended as explicit future work (see chat summary #3), not something the current data can speak to either way.

## 5. On the review's own citations

Several of the sources listed (arXiv IDs like 2603.25857, 2605.12784, 2511.22311, and multiple "2026" conference papers) could not be verified as real, resolvable publications — some carry dates or ID patterns inconsistent with arXiv's actual numbering/timeline. Per this project's standing policy (already applied once this session, when three similarly-sourced citations were removed from the bibliography rather than kept with a `[verify]` tag), we recommend not incorporating these into LEAF's reference list. The methodological ideas themselves (blinding levels for in-context property prediction, CoT rationale requirements) are sound and worth pursuing experimentally regardless of whether a specific citation can be verified — the experiment design doesn't depend on the citation being real.

## 6. Remaining recommended experiments (not yet run)

In rough order of implementation cost:

1. **Attribution-flags prompt** — requires wiring real SHAP/Integrated-Gradients output (already used for the dormant Hypothesis Agent) into a new prompt variant that names which fragment slot is driving current archive performance, without revealing values. Moderate code change + new n=4 run.
2. **Always-active Hypothesis Ledger** — loosen the counterfactual-pair construction to draw from all evaluated candidates (not only the Pareto-restricted archive) or run every round instead of every k=4 rounds. This directly targets the paper's own documented 0/72 empty-Ledger finding. Moderate-to-large change since it touches the counterfactual-pair matching logic (§3.4/Appendix A.2).
3. **CoT structural-rationale requirement** — require a short mechanistic justification before each candidate, still without showing scores. Prompt-only change, cheap to implement, but harder to score/verify systematically (would need a rubric or a second LLM judge, which introduces its own reliability question).
4. **Proposer × acquisition-function interaction sweep** (2×2 UCB/EHVI × LLM/GA) — addresses the acquisition-fairness question above directly. Moderate cost, roughly double the existing EHVI check's compute.

None of these are run yet. Let me know which to prioritize.
