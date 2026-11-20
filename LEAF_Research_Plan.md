# LEAF — Lipid Evolutionary Agentic Framework
### Full Research & Engineering Plan (Revision 2)
*In-silico, multi-agent, budget-constrained active-learning simulation for ionizable lipid design; the GP/AL machinery here is built from scratch.*

> **Revision note.** This revision resolves a set of issues raised in review of Revision 1: an underspecified oracle (surrogate vs. ground truth conflated), a missing acquisition function, an underpowered significance-testing plan, a conformal-calibration scheme that does not account for the loop's adaptive sampling, a rediscovery experiment (H4) that both misdescribed its target finding and was leaked by pretraining, and several other gaps listed in §9. Every section below is the corrected version; nothing here is meant to be read alongside Revision 1.

---

## 0. Framing (reconsidered)

Revision 1 framed LEAF primarily as a *tropism-aware lipid design system*. That framing invites direct comparison against LUMI-lab and LipoAgent on a dimension — real biological data — that is not this project's focus. The defensible and more interesting claim is narrower and methodological:

> **Does LLM-driven evolutionary proposal, under a matched oracle-query budget and a matched acquisition function, outperform a well-tuned non-LLM genetic algorithm on multi-objective molecular search?**

Lipids are the testbed, not the point. Tropism becomes a secondary, clearly-labeled heuristic axis rather than a headline claim (see §5.6). RQ1/H1 (agentic value-add) and RQ7 (reproducibility/variance) are the primary contributions; RQ2–RQ4 are secondary and explicitly allowed to come back negative per the pre-committed fallback in §8.3.

---

## 1. Objectives

| # | Objective | Success looks like |
|---|---|---|
| O1 | Build a closed-loop, multi-agent, in-silico lipid design system | End-to-end loop runs on public data, agent roles clearly separable and independently testable |
| O2 | Demonstrate that agentic (LLM-driven) search adds *measurable* value over non-agentic evolutionary search, under an identical acquisition function | Bootstrap-CI-backed hypervolume gap vs. static-GA ablation across ≥8 seeds, not just narrative plausibility |
| O3 | Replace binary safety-gating (the design pattern LipoAgent's own predictor comparisons motivated; see §5.12 for the corrected characterization of LipoAgent itself) with calibrated, abstention-aware toxicity screening that accounts for the loop's adaptive sampling | Conformal coverage is tracked and reported per round; degradation under distribution shift is disclosed, not assumed away |
| O4 | Turn "agent reasoning" into falsifiable computational experiments rather than free-text claims | Every Hypothesis-Agent claim is backed by a counterfactual ground-truth query with a significance test, and a pretraining-leakage probe precedes any rediscovery claim |
| O5 | Produce a reproducible, fully public, in-silico alternative to robotic self-driving-lab pipelines (LUMI-lab) for groups without robotics access | Full code + data pipeline runnable by a third party with only public datasets |
| O6 | Produce a submittable workshop paper whose primary claim is methodological (agentic value-add under budget), with lipids as a rigorously ablated testbed | Every claim in the paper is backed by a specific experiment in §8, and every ambiguity flagged in §9 is closed before submission |

---

## 2. Research Questions

**Core methodological RQs**

- **RQ1 (agentic value-add — primary).** Does an LLM-driven evolutionary Proposer, paired with the *same* acquisition function, achieve higher Pareto hypervolume (potency, −toxicity, synthesizability, [tropism-heuristic]) than a fixed, non-LLM genetic algorithm under the same ground-truth oracle and the same query budget? *(→ H1)*
- **RQ2 (hypothesis feedback value).** Does biasing the Proposer's mutation operators via the Hypothesis Ledger improve sample efficiency (queries-to-target-hypervolume) relative to an ablation where hypotheses are logged but never used? *(→ H2)*
- **RQ3 (calibration value under drift).** Does debate + conformal calibration — recalibrated every round to account for the loop's adaptive, non-exchangeable sampling — reduce false-accept rate at matched coverage relative to a single binary toxicity filter? *(→ H3)*
- **RQ4 (external validity of the reasoning loop, secondary).** Can the Hypothesis Agent, using a literature corpus that is both frozen *and* pretraining-leakage-probed, independently surface a genuine structure–property association it was not told, without prior exposure via the base model? *(→ H4, now scoped to structure–**property**, not structure–tropism — see §5.5)*

**Engineering-validity RQs (must pass before RQ1–RQ4 are trustworthy)**

- **RQ5 (oracle validity).** Does the frozen ground-truth Oracle generalize acceptably across random / scaffold-cold / chemotype-cold splits? A pre-registered minimum chemotype-cold Spearman correlation (§8.4) gates whether that split's results are reported as findings or as "uninterpretable — oracle below threshold."
- **RQ6 (search-space validity).** Is the fragment vocabulary large enough, relative to the query budget B, that search is a real combinatorial problem (B ≪ |𝒳|) rather than something exhaustively scoreable by the surrogate alone? Report |𝒳| and B/|𝒳| explicitly (§5.7).
- **RQ7 (agent determinism / reproducibility).** How sensitive are LEAF's results to LLM sampling temperature, prompt phrasing, base model choice, and random seed? What is the variance budget that must be reported alongside every RQ1–RQ4 result?

---

## 3. Methods (implementation-oriented summary)

### 3.1 The oracle is two objects, not one

Revision 1 left the oracle's identity ambiguous — a single GP object was simultaneously described as "refit each round" (active-learning surrogate) and as the thing being queried for ground truth. If those are the same object, querying it returns no new information and every downstream metric is measured in the surrogate's own units.

**Resolved design: frozen ground truth + adaptive surrogate.**

| Object | Role | Fit on |
|---|---|---|
| **𝒪_GT** (ground-truth oracle) | The thing that is *queried* — stands in for a real lab | Fit once, on the **full** available labeled data (Xu et al. + AGILE, harmonized — see §5.8), frozen before any round-loop experiment runs. Validated on cold splits per RQ5. |
| **𝒪_AL** (active-learning surrogate) | The thing the loop *reasons with* — posterior mean/variance drive acquisition, Verifier toxicity prior, and Hypothesis Agent attribution | Refit every round on **only the labels acquired so far** (i.e., the subset of 𝒪_GT's outputs that have actually been "queried" within the budget) |

This is the standard retrospective-AL-simulation convention (Settles, 2012; Gessner et al., 2024) applied to lipids. It is a model-on-model evaluation, and the paper says so explicitly rather than implying real-world validation — see the reframed Limitations in §8.6.

**Consequence for candidate generation.** The Proposer can generate fragment tuples absent from the ~1,080-lipid labeled library. 𝒪_GT must therefore itself be a fitted model (not a lookup table) capable of scoring any point in the fragment space — this is why RQ5's cold-split validation of 𝒪_GT specifically (not just 𝒪_AL) is load-bearing: if 𝒪_GT cannot generalize off the exact library it was fit on, none of RQ1–RQ4 mean anything, since every "ground truth" label used for evaluation is itself an extrapolation.

### 3.2 Acquisition function (previously missing)

Revision 1's Algorithm 1 queried every proposed candidate against the ground truth, which is not an active-learning loop — it is exhaustive labeling of whatever the Proposer emits, with 𝒪_AL computed but never used for a decision. Fixed as follows:

1. Proposer generates a **pool** P_t of candidates, |P_t| = m ≫ q (q = per-round query allowance).
2. 𝒪_AL scores all of P_t (free — no budget cost).
3. Verifier agents act on the pool using 𝒪_AL's toxicity posterior (also free); abstentions are removed from the pool.
4. An **acquisition function** over the surviving pool selects C_t ⊂ P_t with |C_t| = q. Default: multi-objective UCB matching the objectives in §5.1; EHVI (expected hypervolume improvement) as a config-flagged alternative, since hypervolume is the primary metric.
5. 𝒪_GT labels **only** C_t; budget decrements by q (plus any Hypothesis Agent counterfactual queries — see §3.5 fix below); 𝒪_AL refits on the newly labeled points.

**Baseline parity requirement.** The static-GA and random-search baselines (§6.2) must use the **identical acquisition function and identical q**. Without this, RQ1 stops isolating proposal quality and starts conflating it with "has an acquisition step vs. doesn't," which is not a fair test of agentic value-add. This constraint is stated explicitly in every baseline config.

### 3.3 Component table

| Component | What it is | What it needs to run |
|---|---|---|
| Fragment library | Fixed vocabulary: head groups, linker chemistry, tail 1/2, unsaturation, halogenation | Enumerated from Xu et al. combinatorial routes; stored as a structured lookup table. |𝒳| computed and reported (RQ6). |
| Proposer Agent | LLM performing crossover/mutation over fragment tuples, generating a pool P_t | Prompt template + JSON-schema-constrained output + fragment-validity checker |
| Ground-Truth Oracle (𝒪_GT) | Multi-head GP, Tanimoto kernel, fit once on full harmonized data, frozen | Embedding function, GP library (GPyTorch), harmonized label sources (§5.8) |
| AL Surrogate (𝒪_AL) | Same architecture as 𝒪_GT, refit every round on acquired labels only | Same infra as above, refit routine, warm-started each round |
| Acquisition function | Selects q candidates from pool P_t for querying against 𝒪_GT | Multi-objective UCB / EHVI over 𝒪_AL's posterior |
| Debate/Conformal Verifier | N independently-prompted toxicologist agents (≥2 distinct base models — see §5.4), RDKit tool calls, per-round-recalibrated conformal prediction | RDKit substructure matcher, held-out calibration set, weighted/Mondrian conformal wrapper, per-round coverage logging |
| Hypothesis Agent | RAG over literature (leakage-probed) + IG/SHAP attribution mining + counterfactual generation against 𝒪_GT | Literature index (vector store) + pretraining-leakage probe (§5.5), IG/SHAP (Captum), counterfactual pair generator, significance test |
| Pareto Archive | Non-dominated sorting across objectives (3 or 4 — see §5.6 on synthesizability/tropism degeneracy) | NSGA-II-style sort (`pymoo`) |
| Orchestrator | Round loop, budget tracking (now including Hypothesis Agent query cost) | Pure Python control flow, no ML |

---

## 4. Repository structure

```
leaf/
├── data/
│   ├── raw/                     # downloaded public datasets (Xu et al., AGILE, LUMI-lab supp. if released)
│   ├── processed/                # cleaned, harmonized (§5.8), split (random / scaffold-cold / chemotype-cold)
│   └── calibration/               # held-out set reserved ONLY for conformal calibration
├── fragments/
│   ├── vocabulary.py              # fragment enumeration + validity rules; reports |X|
│   └── embed.py                   # fragment tuple -> vector embedding
├── oracle/
│   ├── gp_model.py                # Tanimoto-kernel GP, multi-head; shared by GT and AL instances
│   ├── ground_truth.py            # fits/freezes O_GT once on full harmonized data
│   ├── surrogate.py                # O_AL: refit-per-round on acquired labels only
│   ├── acquisition.py             # multi-objective UCB / EHVI over O_AL posterior
│   └── eval_splits.py             # random / scaffold-cold / chemotype-cold evaluation of O_GT
├── agents/
│   ├── proposer.py                # LLM crossover/mutation agent, emits pool P_t
│   ├── verifier.py                # debate ensemble (multi-model) + per-round conformal calibration
│   ├── hypothesis.py              # RAG + attribution mining + counterfactual testing against O_GT
│   ├── leakage_probe.py           # pretraining-leakage probe run before any RQ4 claim
│   └── prompts/                   # versioned prompt templates (critical for RQ7 reproducibility)
├── archive/
│   └── pareto.py                  # non-dominated sorting, hypervolume computation
├── orchestrator/
│   └── loop.py                    # Algorithm 1 (corrected), round-by-round, budget includes hypothesis queries
├── baselines/
│   ├── static_ga.py               # non-LLM genetic algorithm, SAME acquisition function + q as LEAF
│   ├── random_search.py
│   └── lipoagent_runner.py        # real LipoAgent efficiency-score client (NOT a toxicity/calibration comparator -- see §5.12)
├── eval/
│   ├── metrics.py                 # hypervolume, Recall@k, hypothesis precision, per-round coverage, inter-agent agreement
│   ├── significance.py            # bootstrap CI (primary) + Wilcoxon (secondary, ≥8 seeds — see §7.2)
│   └── run_experiment.py          # CLI entrypoint for any (method, split, seed) triple
├── tests/                          # unit + integration tests, one per module above
└── configs/                        # YAML configs per experiment (method, split, budget, seed, acquisition fn)
```

---

## 5. Fixes to specific gaps (indexed to what they resolve)

### 5.1 Objective harmonization and normalization
f(x) = (f_pot(x), −f_tox(x), f_syn(x), [f_trop(x)]). Before any hypervolume computation, each axis is normalized (min–max or rank-based) against the frozen 𝒪_GT training distribution, and the reference point is fixed and logged **before** any experiment runs (unchanged from Revision 1, now made concrete: normalization method stated in the pre-registered config).

### 5.2 Seed count and statistical test (was: 3 seeds + Wilcoxon)
With n = 3 paired seeds, Wilcoxon signed-rank cannot reach p < 0.05 (minimum achievable two-sided p = 0.25 at n=3; 8 sign patterns total). **Fixed:** run **≥8 seeds** for E2 (the primary RQ1 claim). Report the **bootstrap CI on the median paired hypervolume difference** as the primary statistic; report Wilcoxon signed-rank as a secondary, corroborating statistic where seed count allows. Do not claim significance from Wilcoxon alone below n=8.

### 5.3 Conformal calibration under loop-induced distribution shift
Split conformal assumes exchangeability between calibration and test points. The loop violates this by construction: candidates at round t are selected via acquisition on rounds 1..t−1, and 𝒪_AL is refit each round, so nonconformity scores drift. **Fixed:**
- Recalibrate conformal thresholds every round using the most recent 𝒪_AL fit.
- Report **coverage as a function of round**, not a single number — expect and disclose degradation.
- Use weighted or Mondrian conformal prediction to partially correct for the covariate shift toward the Pareto front.
- Correct the category error in Revision 1's phrasing: coverage is a property of the *procedure* across many points, not of a single candidate's interval. Use "interval width" or "interval straddles the decision boundary" when describing a single candidate's fate.

### 5.4 Debate ensemble diversity
N prompts of a single base model measure prompt sensitivity, not epistemic uncertainty, and correlate heavily. **Fixed:**
- Report **inter-agent agreement** as a first-class metric (not just aggregated verdicts).
- Run at least one configuration with ≥2 distinct base models in the debate ensemble, and report whether abstention behavior changes.
- If single-model agreement is ≥95%, flag the abstention set as too small to test H3 meaningfully in that configuration.

### 5.5 H4 / rediscovery experiment — corrected target and leakage probe
Revision 1 described the target finding as a "halogenated-tail tropism association." The actual LUMI-lab result is that brominated lipid tails improve **potency** (transfection efficiency) in vitro, with the lung-editing result following from **intratracheal route of administration** in vivo — not a structure→organ-tropism rule. **Fixed:**
- RQ4/H4 is rescoped to **structure–property** (not structure–tropism) rediscovery, matching what was actually published.
- **Mandatory pretraining-leakage probe before E5 runs**: query the bare LLM (no RAG, no archive, no LEAF context) — "what tail chemical modifications enhance mRNA transfection efficiency in ionizable lipids?" If the model names bromination/halogenation unprompted, E5 is downgraded from "external validation" to "consistency check" per the pre-committed fallback in §8.5, and this is disclosed in the paper regardless of outcome.
- Freezing the RAG corpus (Revision 1's only leakage safeguard) is necessary but insufficient — it does not address base-model pretraining knowledge, which the LUMI-lab preprint (bioRxiv, Feb 2025) has been part of for well over a year.
- Where feasible, prefer a rediscovery target constructed from held-out structure–property relationships in the project's own data (a relationship deliberately withheld from the Hypothesis Agent's training-time archive access) as a cleaner leakage-free alternative.

### 5.6 Tropism proxy and synthesizability degeneracy
- **Tropism proxy**: none of the three candidate data sources (Xu et al., AGILE, conditional LUMI-lab release) carries organ-selectivity labels; the realistic outcome is a heuristic proxy encoding known structure→organ rules. If so, the Hypothesis Agent "discovering" structure–tropism relationships is recovering the proxy's own construction — a tautology. **Fixed:** tropism is demoted to a clearly-labeled heuristic filter reported **outside** the primary hypervolume, not as a fourth optimization axis, unless a genuine organ-selectivity dataset is confirmed available (tracked per §5.9). The title and abstract are updated accordingly (§8.7).
- **Synthesizability**: since the fragment vocabulary is restricted to literature-precedented synthetic routes (per the Ethics section), f_syn may be near-constant across the whole space, contributing negligible hypervolume while adding reference-point sensitivity. **Fixed:** compute and report the variance of f_syn across the full enumerated vocabulary *before* committing it as an optimization axis (Step 1, gate below); drop to a 2-objective (potency, −toxicity) or 3-objective hypervolume if variance is negligible.

### 5.7 Search-space size (RQ6 — resolved)
Real fragment slots are (A, B, C) — the three Ugi-3CR reactants — not the earlier illustrative head/linker/tail/unsaturation/halogenation placeholder (superseded once the real AGILE data was vendored, §5.9). Initial real vocabulary was AGILE's own labeled set: a full factorial 20×12×5 = 1,200 with zero missing combinations, meaning |𝒳| exactly equaled the labeled set and B ≪ |𝒳| could not hold — disclosed as an open limitation, then fixed: AGILE's own repository also publishes a real, unlabeled 12,276-candidate virtual library from the same reaction. `fragments/vocabulary.py::decompose_lipid` (exact inverse of the assembler, validated 1200/1200 round-trip) recovered each candidate's (A,B,C) triple with 100% success, yielding 9/62/9 new real A/B/C values and an extended |𝒳| = 29×74×14 = **30,044** (validated: 100% RDKit round-trip on the full enumerated space). Only the original 1,200 have real labels; 𝒪_GT must genuinely extrapolate to the other 28,844 real, chemically legitimate combinations. B/|𝒳| ≈ 0.27% at E2's reduced budget (80) — see `leaf/data/raw/README.md` for full provenance and checksums.

### 5.8 Label harmonization
Xu et al. and AGILE use different assays/cell lines. Before merging into a single potency head: rank-normalize within source and/or add a source indicator feature to 𝒪_GT/𝒪_AL. This harmonization step is implemented and unit-tested in `data/processed/`, and documented in the paper rather than left implicit.

### 5.9 Toxicity/tropism label provenance
Name explicitly, in `data/raw/README.md` and in the paper, exactly which dataset provides the toxic/non-toxic calibration labels for the Verifier (§3.4/RQ3) and whether any tropism label source is genuine data vs. heuristic proxy. If toxicity labels come from a different chemotype (e.g., a general small-molecule tox dataset), the resulting domain gap is stated as a limitation, not left implicit.

### 5.10 Budget accounting bug
Revision 1's Algorithm 1 decremented the budget only by |C_t|, but the Hypothesis Agent's counterfactual pairs also consume ground-truth queries. **Fixed:** Algorithm 1 line "B_t ← B_t − |C_t|" becomes "B_t ← B_t − |C_t| − (counterfactual queries issued this round, if t mod k = 0)". This is now unit-tested (`tests/test_budget_accounting.py`): total queries issued across the full run must never exceed B, counting both acquisition and hypothesis-testing queries.

### 5.11 Oracle validity go/no-go threshold
Pre-register a minimum chemotype-cold Spearman correlation for 𝒪_GT (e.g., ρ ≥ 0.3, exact value fixed before E1 runs and logged in the experiment config). If 𝒪_GT falls below threshold on a split, that split's RQ1–RQ4 results are reported as **uninterpretable**, not as findings — this replaces the vaguer "known before it is trusted" language from Revision 1.

### 5.12 LipoAgent baseline correction (revised after inspecting the vendored code)
- Correct repository: **github.com/SAI-Lab-NYU/LipoAgent** (confirmed reachable and matching the paper's general description — multi-agent LLM coordination via TxGemma-27B-Chat and Qwen3-8B agents with a Predict Agent + Verify Agent per base model). Prior citation details (arXiv:2605.25250, "Y. Li et al.") were carried from an unverified literature-search summary and should be re-checked against the repository's actual associated publication (its README states "accepted to ACL 2026," not the arXiv ID previously assumed).
- **LipoAgent is a de novo/toxicity-gate role was based on a mischaracterization.** Direct inspection of `txgemma/predict_agent.py` and `txgemma/verify_agent.py` shows both scripts produce **only an `efficiency_score` (1–10, predicted mRNA transfection performance)** — there is no toxicity, safety, or gating field anywhere in either script's input, prompt, or output. The "Verify Agent" is a second efficiency-scoring pass (with more detailed prompting for extreme scores), not a safety check.
- **Fixed:** LipoAgent is **removed as the E4 binary-gate comparator**. Fabricating a toxicity threshold on top of an efficiency score would misrepresent a capability the tool does not have. E4 now compares full LEAF against Ablation (e) (`BinaryToxicityFilter`, implemented natively in `agents/verifier.py`, no external dependency) only.
- LipoAgent's real, correctly-scoped role (if used at all) is as a **potency/efficiency baseline** — e.g. contributing to Recall@k — not as a toxicity or calibration comparator. This is out of scope for the current experimental design (§7) and would be a distinct, separately-justified addition.
- E7 ("public baseline comparison" against LipoAgent-as-searcher, Revision 1) remains **removed** from the experimental design (§6.2) for the original reason (task mismatch — LipoAgent predicts over an enumerated library, it does not search).

---

## 6. Detailed Flow of Code Writing & Execution

### 6.1 Build order (each step gated on the previous step's tests passing)

**Step 1 — Data & fragment vocabulary (no ML yet)**
1. Acquire and clean public datasets into `data/raw/` → `data/processed/`, including the label-harmonization step (§5.8).
2. Implement `fragments/vocabulary.py`: enumerate valid (head, linker, tail1, tail2, unsaturation, halogenation) tuples; compute and log |𝒳| (§5.7).
3. **Test:** every enumerated fragment tuple must round-trip to a valid RDKit molecule (100% of the vocabulary).
4. Compute f_syn variance across the full vocabulary (§5.6) — decide 3- vs. 4-objective hypervolume **before** Step 5.
5. **Gate:** fragment validity = 100%, |𝒳| and B/|𝒳| logged, dataset splits frozen and checksum-logged, synthesizability-axis decision made and logged.

**Step 2 — Ground-Truth Oracle (RQ5, RQ11 live here)**
6. Implement `fragments/embed.py`.
7. Implement `oracle/gp_model.py` with the Tanimoto kernel; single-objective potency-only GP first.
8. **Test:** sane potency-prediction Pearson/Spearman on random split before touching cold splits.
9. Extend to multi-head (potency, toxicity, [tropism if a genuine dataset is confirmed — §5.9]).
10. Fit and **freeze** `oracle/ground_truth.py` on the full harmonized dataset. This frozen object is 𝒪_GT for the rest of the project.
11. **Test (RQ5):** run all three splits, log the generalization gap; compare against the pre-registered go/no-go threshold (§5.11).
12. **Gate:** 𝒪_GT split-wise performance table exists, is committed, and each split is marked interpretable/uninterpretable per §5.11 before proceeding.

**Step 3 — AL surrogate + acquisition function**
13. Implement `oracle/surrogate.py` (𝒪_AL, refit-per-round on acquired labels only).
14. Implement `oracle/acquisition.py` (multi-objective UCB default, EHVI config option).
15. **Test:** on a toy synthetic problem, verify acquisition concentrates queries near the Pareto front faster than random selection from the pool.
16. **Gate:** do not proceed to the Verifier or Proposer until acquisition demonstrably outperforms random pool selection on the toy problem.

**Step 4 — Verifier Agent (can build in parallel with Step 5)**
17. RDKit toxicophore substructure rules (`agents/verifier.py`).
18. Wire N independently-prompted toxicologist agent calls across **≥2 distinct base models** (§5.4); log raw per-agent verdicts and pairwise inter-agent agreement before any aggregation logic.
19. Implement **per-round-recalibrated**, weighted/Mondrian split-conformal calibration using `data/calibration/` (never touched by training or the main loop).
20. **Test (RQ3 precondition):** verify conformal coverage as a function of round on a synthetic held-out slice with induced distribution shift, before running it inside the real loop.
21. **Gate:** log coverage-vs-round curves and inter-agent agreement; do not integrate into the orchestrator until both are reviewed.

**Step 5 — Proposer Agent**
22. Define the strict JSON schema for a proposed candidate (fragment tuple + provenance).
23. Implement crossover/mutation prompt templates generating a **pool** P_t (§3.2), not a directly-queried population; version every prompt (RQ7).
24. **Test:** 100% of LLM outputs parse to valid fragment tuples (reject/retry logic); log reject rate as a metric.
25. **Gate:** run Proposer + acquisition + 𝒪_GT only (no Verifier, no Hypothesis Agent) as **Ablation (a)**.

**Step 6 — Pareto Archive + Orchestrator loop**
26. Implement non-dominated sorting and hypervolume computation, using the objective set and normalization decided in Step 1 (`archive/pareto.py`).
27. Wire the corrected round loop (`orchestrator/loop.py`, Algorithm 1 with the budget fix in §5.10), Verifier + Proposer + Acquisition + Archive only = **Ablation (b)**.
28. **Test:** budget accounting is exact (queries used, including any hypothesis-agent queries once Step 7 lands, never exceeds B); archive is monotonically non-dominated across rounds.

**Step 7 — Hypothesis Agent (last)**
29. Implement IG/SHAP attribution extraction over 𝒪_AL and the current archive.
30. Build the literature index (RAG); freeze the corpus before running RQ4; implement and run `agents/leakage_probe.py` (§5.5) **before** any rediscovery claim is made.
31. Implement counterfactual pair generation + significance testing against 𝒪_GT (not 𝒪_AL); charge these queries against the budget (§5.10). A hypothesis is written to the Ledger **only** if it passes the significance threshold.
32. Wire Ledger → Proposer mutation-bias feedback.
33. **Test:** run the full loop = **full LEAF**, i.e. Ablation (c).

**Step 8 — Baselines & ablations (can build in parallel with Steps 5–7 once 𝒪_GT + acquisition exist)**
34. `baselines/static_ga.py`: same fragment vocabulary, same 𝒪_GT, **same acquisition function and q** as LEAF (§3.2 parity requirement).
35. `baselines/random_search.py`.
36. `baselines/lipoagent_runner.py`: real LipoAgent efficiency-score client (vendoring + GPU inference required to actually run it); **not used as the E4 comparator** — LipoAgent's real code has no toxicity signal to gate on (§5.12).
37. Ablations (d) [no Ledger feedback] and (e) [binary filter instead of debate/conformal] are config flags on the already-built system.

### 6.2 Testing discipline
- Every module in `agents/`, `oracle/`, `archive/` gets a unit test file before it is wired into `orchestrator/loop.py`.
- Integration test: a tiny end-to-end run (budget = 20 queries, toy fragment vocabulary subset) completes in under a few minutes — this is what CI runs on every commit.
- Full experiments run manually / on a schedule only, given LLM API cost.

---

## 7. Experimental Design

| Experiment | RQ(s) | Method(s) compared | Splits | Seeds |
|---|---|---|---|---|
| E1 — Oracle validation | RQ5, RQ6 | 𝒪_GT only, all splits, go/no-go threshold applied | random, scaffold-cold, chemotype-cold | 3 |
| E2 — Agentic value-add (**primary**) | RQ1 | Full LEAF vs. static GA vs. random search, **identical acquisition fn** | all 3 splits | ≥8 |
| E3 — Hypothesis feedback ablation | RQ2 | Full LEAF vs. Ablation (d) | chemotype-cold | ≥8 |
| E4 — Verifier calibration ablation | RQ3 | Full LEAF vs. Ablation (e) (`BinaryToxicityFilter`) | held-out calibration set, per-round coverage tracked | ≥8 |
| E5 — Hypothesis rediscovery (secondary, leakage-probed) | RQ4 | Hypothesis Agent only, frozen + leakage-probed corpus | N/A (case study) | report per-run |
| E6 — Prompt/temperature/base-model sensitivity | RQ7 | Full LEAF across prompt variants × temperatures × ≥2 base models | one representative split | ≥3 per configuration |
| E8 — External validation (conditional) | RQ1, RQ4 | Full LEAF vs. LUMI-lab supplementary data, *if released* | external | 1 (fixed) |

**E7 (LipoAgent-as-searcher comparator) is removed**, and **LipoAgent is also removed from E4** (its real code has no toxicity signal to gate on) — see §5.12.

### 7.2 Significance testing (corrected)
- E2–E4 use **≥8 seeds** and report the **bootstrap CI on the median paired difference** as the primary statistic.
- Wilcoxon signed-rank is reported as a secondary corroborating statistic where seed count and rank distribution support it; it is never used as the sole basis for a significance claim below n = 8, and the specific rank distribution required to hit p < 0.05 at a given n is checked before it is relied on (do not assume "≥8 seeds" alone guarantees p < 0.05 — it depends on which seed is discordant and its rank).

---

## 8. Rigorous Evaluation Protocol

1. **Pre-registration discipline.** Before running E2–E4, freeze and log in a committed config: hypervolume reference point + normalization method (§5.1), the significance threshold for Hypothesis-Agent claims, the literature corpus for RAG (post leakage-probe), the acquisition function and q (identical across LEAF and baselines), and the 𝒪_GT go/no-go threshold (§5.11).
2. **Multi-seed, paired statistics.** ≥8 seeds, bootstrap CI primary, per §7.2.
3. **The load-bearing negative-result plan.** If E2 shows static GA statistically matches full LEAF's hypervolume under identical acquisition, the paper's contribution claim shifts from "agentic reasoning improves design efficiency" to "agentic reasoning improves interpretability and hypothesis generation without efficiency loss" — decided in advance, not after seeing results.
4. **No leakage in E5.** Corpus frozen and logged; **pretraining-leakage probe run and disclosed regardless of outcome** (§5.5); if leaked, E5 is downgraded to "consistency check," never silently reported as external validation.
5. **Reproducibility artifact.** All prompts (versioned), configs, random seeds, acquisition function code, and the frozen+probed literature corpus snapshot are released alongside the code.
6. **Reporting checklist before submission:**
   - [ ] E1 split-wise 𝒪_GT table including the honest generalization gap and go/no-go outcome per split
   - [ ] E2 hypervolume comparison with bootstrap CI (≥8 seeds), identical-acquisition-function statement
   - [ ] E3 ablation isolating Hypothesis-Ledger feedback value
   - [ ] E4 calibration curves (coverage **vs. round**) + inter-agent agreement table
   - [ ] E5 rediscovery case study with leakage-probe result disclosed, retitled structure–property (not structure–tropism)
   - [ ] E6 prompt/temperature/base-model sensitivity table
   - [ ] E8 external validation, marked conditional on data release
   - [ ] |𝒳| and B/|𝒳| reported (RQ6)
   - [ ] f_syn variance reported; objective count (3 vs. 4) justified
   - [ ] Label harmonization and toxicity/tropism data provenance stated explicitly
   - [ ] Limitations section states this is a model-on-model (𝒪_AL vs. 𝒪_GT) evaluation, not real-world validation
   - [ ] Limitations section states which RQs were *not* answered affirmatively, if any

---

## 9. Scope Boundary

- No claim of biological ground truth for any tropism signal — it is explicitly a research instrument, demoted outside the primary hypervolume unless a genuine organ-selectivity dataset is confirmed (§5.6).
- A trained generative Proposer (flow-matching / diffusion) is deferred to Future Work (§6 item 3), contingent on H1 showing headroom the evolutionary Proposer cannot close.

---

## 10. Log of issues closed in this revision

1. Oracle GT/AL conflation → §3.1
2. Missing acquisition function → §3.2, build Step 3
3. Underpowered significance plan (n=3, Wilcoxon) → §5.2, §7.2
4. Conformal calibration ignoring loop-induced distribution shift → §5.3
5. Debate ensemble = single-model prompt variants → §5.4
6. H4 misdescribed target finding + pretraining leakage unaddressed → §5.5
7. Tropism proxy circularity / synthesizability degeneracy → §5.6
8. Search-space size never stated → §5.7
9. Label harmonization across heterogeneous assays unstated → §5.8
10. Toxicity/tropism label provenance unnamed → §5.9
11. Budget accounting omitted Hypothesis Agent queries → §5.10
12. No pre-registered oracle validity threshold → §5.11
13. LipoAgent citation/year error, task-mismatch baseline (E7) → §5.12
14b. LipoAgent mischaracterized as toxicity-gated; its real code (inspected directly) has no toxicity signal, so it is also removed from E4 → §5.12
14. Step numbering breaks in Revision 1's build order → §6.1 renumbered
15. Framing over-claimed tropism-awareness and novelty of retrospective AL simulation on lipids → §0, §8.7 (title/abstract narrowed)
