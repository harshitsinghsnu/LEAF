"""CLI entrypoint for any (experiment, method, split, seed) run.

Writes results to results/<experiment>/<method>_<split>_seed<seed>/:
  - manifest.json   (git hash, config, package versions -- §8 reproducibility)
  - result.json     (summary metrics)
  - rounds.csv       (per-round log: hypervolume, coverage, agreement, budget)

Usage:
    python -m leaf.eval.run_experiment --config leaf/configs/e2_agentic_value_add.yaml \
        --experiment e2 --method leaf --split random --seed 0

See scripts/run_all_experiments.py to sweep every (method, split, seed)
triple a config specifies and generate the paper's figures/tables in one
command.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from leaf.agents.proposer import RuleBasedProposer, LLMProposer
from leaf.baselines.static_ga import StaticGAProposer
from leaf.baselines.random_search import RandomSearchProposer
from leaf.baselines.bandit_proposer import BanditProposer
from leaf.data.agile_loader import load_agile_labeled_set
from leaf.eval.reproducibility import RunManifest, warn_if_dirty
from leaf.fragments.embed import FragmentEmbedder
from leaf.oracle.eval_splits import build_splits
from leaf.oracle.ground_truth import GroundTruthOracle
from leaf.orchestrator.loop import LoopConfig, run_loop

# method name -> (proposer class, loop-config overrides). Config files
# (leaf/configs/e*.yaml) name methods by their ablation letter (research
# plan §4.2); every name a config can use must be registered here or the
# sweep driver fails with a bare KeyError instead of a clear message.
METHOD_SPECS = {
    "leaf": (RuleBasedProposer, {}),
    # The REAL agentic proposer -- an actual LLM (local Ollama, llama3.1:8b
    # by default) performing crossover/mutation, per research plan §3.2.
    # "leaf" above remains RuleBasedProposer (a deterministic non-LLM
    # stand-in) for E3/E4/E6, which ablate the Verifier/Hypothesis Agent,
    # not the Proposer -- swapping in the LLM there would only add ~4s/
    # candidate of latency without testing anything those experiments are
    # actually about. "leaf_llm" is what the paper's H1 (agentic value-add)
    # claim actually requires.
    "leaf_llm": (LLMProposer, {}),
    "leaf_llm_ablation_a": (LLMProposer, {"use_verifier": False, "use_hypothesis": False}),
    "leaf_llm_ablation_b": (LLMProposer, {"use_hypothesis": False}),
    "leaf_llm_ablation_d": (LLMProposer, {"use_hypothesis_feedback": False}),
    "leaf_llm_ablation_e": (LLMProposer, {"verifier_mode": "binary"}),
    # Isolates proposal quality from selection quality: same LLM, but the
    # acquisition function is disabled (first q survivors queried
    # directly, no UCB scoring) -- see LoopConfig.use_acquisition.
    "leaf_llm_no_acquisition": (LLMProposer, {"use_acquisition": False}),
    # Same experimental design (budget, acquisition, vocabulary, schema-
    # constrained JSON generation), different underlying local model --
    # tests prompt/architecture sensitivity at the SAME parameter scale as
    # leaf_llm (Llama 3.1 8B); NOT a scale comparison (that needs a much
    # larger model than this environment can run at usable speed).
    "leaf_llm_qwen": (lambda seed: LLMProposer(seed=seed, model="qwen2.5:7b"), {}),
    # A genuinely larger local model (14B vs. 7-8B for the two above),
    # made possible by an NVIDIA RTX A4000 (16GB VRAM) available in this
    # environment -- tests whether the §6 underperformance is specific to
    # small (<=8B) models or persists at a real, if modest, step up in
    # scale. Not a 40B+/hosted-model test (still blocked by VRAM/API
    # access), but a real, non-trivial scale comparison, not a stand-in.
    "leaf_llm_qwen14b": (lambda seed: LLMProposer(seed=seed, model="qwen2.5:14b"), {}),
    # A biomedical-domain-adapted model (Meditron 7B, Llama-2-based,
    # continued-pretrained on PubMed/clinical guideline text). This is the
    # closest genuinely domain-adapted model available as a pre-built
    # Ollama pull in this environment -- no chemistry-specific LLM
    # (ChemLLM, ChemDFM, etc.) is published on the Ollama registry, and
    # TxGemma requires HuggingFace gated access + more VRAM than
    # available (\S7 item 6). Meditron is medical/biomedical, not
    # chemistry-specific -- reported as such, not relabeled to fit.
    "leaf_llm_meditron": (lambda seed: LLMProposer(seed=seed, model="meditron"), {}),
    # Same model/architecture as leaf_llm, prompt_version v2_diversity:
    # explicitly names the duplicate-heavy failure mode (§6.2) and lists
    # already-evaluated triples to avoid, testing whether the
    # underperformance is fixable by prompting alone (research plan §7
    # item 9) rather than a property of the model/setup.
    "leaf_llm_v2diversity": (lambda seed: LLMProposer(seed=seed, prompt_version="v2_diversity"), {}),
    # v3_scored: adds real per-member potency/potency_raw scores and an
    # objective description to the archive sample shown in the prompt
    # (external review item #2 -- the default prompt shows only fragment
    # identities, never quantitative feedback about what is being
    # optimized). Same model/architecture as leaf_llm otherwise.
    "leaf_llm_v3scored": (lambda seed: LLMProposer(seed=seed, prompt_version="v3_scored"), {}),
    # v4_rank: shows only each archive member's relative rank (1=best),
    # never the underlying scores or scale -- isolates whether relative
    # preference alone (not absolute magnitude/anchoring) drives any
    # effect seen with v3_scored (external review follow-up).
    "leaf_llm_v4rank": (lambda seed: LLMProposer(seed=seed, prompt_version="v4_rank"), {}),
    # v5_cot: requires a short structural rationale before committing to a
    # candidate, no scores shown -- tests whether forced explicit
    # mechanistic reasoning (not label access) changes proposal quality
    # (external review item #4).
    "leaf_llm_v5cot": (lambda seed: LLMProposer(seed=seed, prompt_version="v5_cot"), {}),
    # v6_shuffled: real score VALUES, randomly reassigned to the wrong
    # fragment -- a perturbation control (arXiv:2610.00898 methodology)
    # testing whether v3_scored's effect depends on the true
    # score-to-fragment mapping or just on numbers being present.
    "leaf_llm_v6shuffled": (lambda seed: LLMProposer(seed=seed, prompt_version="v6_shuffled"), {}),
    # Genuine reasoning-mode Proposer (external review: test an actual
    # reasoning-tuned model with native extended inference-time reasoning,
    # not just a prompted one-sentence rationale -- the distinction
    # arXiv:2610.00898 draws between models that reason over inputs and
    # models merely asked to). Qwen3-8B supports Ollama's native "think"
    # API; default prompt (v1), scores not shown.
    "leaf_llm_qwen3_reasoning": (lambda seed: LLMProposer(seed=seed, model="qwen3:8b", think=True, timeout=300.0), {}),
    # Same reasoning model + the real objective-scored prompt, to test
    # whether genuine extended reasoning (not just a prompted rationale)
    # makes better use of injected scores than the non-reasoning models did.
    "leaf_llm_qwen3_reasoning_scored": (lambda seed: LLMProposer(seed=seed, model="qwen3:8b", think=True, prompt_version="v3_scored", timeout=300.0), {}),
    # Second, independently-lineaged reasoning model (DeepSeek-R1 8B,
    # Llama-3.1-8B-distilled) to test whether the Qwen3 engagement-without-
    # performance finding generalizes beyond one model/architecture.
    "leaf_llm_deepseekr1_reasoning": (lambda seed: LLMProposer(seed=seed, model="deepseek-r1:8b", think=True, timeout=600.0), {}),
    "leaf_llm_deepseekr1_reasoning_scored": (lambda seed: LLMProposer(seed=seed, model="deepseek-r1:8b", think=True, prompt_version="v3_scored", timeout=600.0), {}),
    # A real 70B-class hosted model via Groq's free API tier, addressing
    # \S7 item 5 (previously blocked by no API credentials/insufficient
    # local VRAM for 40B+ models). Requires GROQ_API_KEY set in the
    # environment -- raises immediately if unset (LLMProposer.__init__).
    # max_retries=8: Groq's free tier is 12K tokens/min on this model, and
    # needs several backoff-and-retry cycles (LLMProposer.propose_pool)
    # before the window resets, not just 2-3 quick retries.
    # vocab_sample_size=8: the real, harder constraint is a 100,000
    # tokens/DAY cap. Measured: full vocabulary costs ~1,850 tokens/call
    # (a full seed-run needs ~148K tokens, exceeding the daily cap
    # outright); vocab_sample_size=8 cuts this to ~813 tokens/call
    # (~65K tokens/seed-run), fitting one seed-run/day with margin for
    # retries. See LEAF_Research_Plan.md and paper §6.2/§7 item 5 for the
    # full account of why this was necessary.
    "leaf_llm_groq70b": (lambda seed: LLMProposer(seed=seed, backend="groq",
                                                    model="llama-3.3-70b-versatile",
                                                    max_retries=8, vocab_sample_size=8), {}),
    # A real frontier-class hosted model via Google's free Gemini API
    # tier, the other half of \S7 item 5. Requires GEMINI_API_KEY set in
    # the environment.
    # gemini-flash-latest (aliases to gemini-3.6-flash) has a hard 20
    # requests/DAY free-tier cap (GenerateRequestsPerDayPerProjectPerModel
    # -FreeTier) -- confirmed by inspecting the actual 429 response body,
    # not assumed; a full sweep needs ~1,920 calls, so that model is
    # infeasible on the free tier regardless of retry/backoff logic.
    # gemini-flash-lite-latest is a separate quota bucket with headroom.
    "leaf_llm_gemini": (lambda seed: LLMProposer(seed=seed, backend="gemini",
                                                   model="gemini-flash-lite-latest",
                                                   max_retries=8), {}),
    "leaf_full": (RuleBasedProposer, {}),  # alias used by e3/e4 configs
    "leaf_ablation_a": (RuleBasedProposer, {"use_verifier": False, "use_hypothesis": False}),
    "leaf_ablation_b": (RuleBasedProposer, {"use_hypothesis": False}),
    "leaf_ablation_d": (RuleBasedProposer, {"use_hypothesis_feedback": False}),
    "leaf_ablation_e": (RuleBasedProposer, {"verifier_mode": "binary"}),
    "static_ga": (StaticGAProposer, {"use_hypothesis": False}),
    "random_search": (RandomSearchProposer, {"use_hypothesis": False}),
    # A real, fitted (not evolutionary) baseline: per-slot Beta-posterior
    # Thompson sampling over the fragment vocabulary, fit fresh from the
    # Pareto archive every round. Lighter than a trained generative
    # Proposer (still deferred, real training work) but genuinely
    # different from both static-GA (no crossover/mutation) and random
    # search (informed, not blind).
    "bandit": (BanditProposer, {"use_hypothesis": False}),
}
# "lipoagent_binary_gate" is intentionally NOT registered. The vendored,
# real LipoAgent code (github.com/SAI-Lab-NYU/LipoAgent) was inspected
# directly: neither its Predict nor Verify agent produces a toxicity
# signal -- both only output an efficiency_score (1-10). LipoAgent cannot
# serve as a toxicity binary-gate comparator without fabricating a
# capability it doesn't have, so it is out of scope for E4 rather than
# "not yet wired" -- see leaf/configs/e4_verifier_calibration_ablation.yaml
# and LEAF_Research_Plan.md §5.12 for the full explanation.
NOT_YET_WIRED: dict[str, str] = {}

RESULTS_ROOT = Path("results")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _fit_real_oracle_for_split(embedder: FragmentEmbedder, split: str,
                                dataset_seed: int = 0, heads: list[str] | None = None) -> GroundTruthOracle:
    """Fits O_GT on the REAL, vendored AGILE dataset (real Ugi-3CR
    chemistry, real Morgan-fingerprint embedding, real expt_Hela/expt_Raw
    labels -- see leaf/data/raw/README.md), restricted to the given
    split's train partition (via oracle/eval_splits.build_splits) so that
    `split` actually changes behavior (random/scaffold_cold/chemotype_cold
    previously produced byte-identical results when this function used
    synthetic placeholder labels that ignored the split argument).

    For a reportable result, prefer passing --oracle
    leaf/data/processed/oracle_gt.pkl (built by
    scripts/run_e1_oracle_validation.py --freeze-oracle, fit on the FULL
    real dataset and validated against the E1 go/no-go gate) instead of
    this per-split-train-only fit.

    `dataset_seed` is intentionally independent of the experiment's run
    seed: O_GT must be the SAME frozen object across all seeds of a given
    (experiment, method, split) -- only the loop's own stochastic parts
    (Proposer sampling, acquisition ties, etc.) should vary with the run
    seed, not the ground truth itself.
    """
    frags, labels = load_agile_labeled_set()
    splits = build_splits(frags, seed=dataset_seed)
    train_idx, _ = splits[split]
    train_frags = [frags[i] for i in train_idx]

    X = embedder.embed_batch(train_frags)
    heads = heads if heads is not None else ["potency", "potency_raw"]
    Y = {h: labels[h][train_idx] for h in heads}
    return GroundTruthOracle(heads=heads).fit(X, Y).freeze()


def run_one(config_path: str, experiment: str, method: str, split: str, seed: int,
            oracle_path: str | None = None) -> Path:
    if method in NOT_YET_WIRED:
        raise NotImplementedError(f"method '{method}' is not runnable yet: {NOT_YET_WIRED[method]}")
    if method not in METHOD_SPECS:
        raise KeyError(f"Unknown method '{method}'. Registered methods: {sorted(METHOD_SPECS)}. "
                        f"Not-yet-wired methods (raise NotImplementedError instead): {sorted(NOT_YET_WIRED)}.")

    cfg = load_config(config_path)
    if "acquisition_function" not in cfg and method in ("leaf", "leaf_llm", "leaf_full", "static_ga", "random_search", "bandit"):
        raise ValueError(
            "Config must pin an acquisition_function -- baseline parity requires the "
            "same acquisition fn/q across leaf/static_ga/random_search (research plan §3.2)."
        )

    out_dir = RESULTS_ROOT / experiment / f"{method}_{split}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(experiment=experiment, method=method, split=split, seed=seed, config=cfg)
    warn_if_dirty(manifest)
    manifest.save(out_dir / "manifest.json")

    heads = cfg.get("heads", ["potency", "potency_raw"])
    embedder = FragmentEmbedder()
    if oracle_path and Path(oracle_path).exists():
        oracle_gt = GroundTruthOracle.load(oracle_path)
    else:
        oracle_gt = _fit_real_oracle_for_split(embedder, split, heads=heads)

    proposer_cls, overrides = METHOD_SPECS[method]
    proposer = proposer_cls(seed=seed)
    # Real E6 temperature sweep support: a config's `temperature` field
    # overrides an LLM proposer's default post-construction, so the same
    # method entry (e.g. "leaf_llm") can be run at several temperatures
    # via separate config files without new METHOD_SPECS entries per
    # temperature (research plan §7 item 9).
    if isinstance(proposer, LLMProposer) and "temperature" in cfg:
        proposer.temperature = float(cfg["temperature"])
    loop_cfg_kwargs = dict(
        budget=cfg["budget"], pool_size=cfg["pool_size"], q=cfg["q"], seed=seed,
        hypothesis_every_k=cfg.get("hypothesis_every_k", 5),
        use_hypothesis=cfg.get("use_hypothesis", True),
        use_hypothesis_feedback=cfg.get("use_hypothesis_feedback", True),
        use_verifier=cfg.get("use_verifier", True),
        use_acquisition=cfg.get("use_acquisition", True),
        log_diagnostics=cfg.get("log_diagnostics", False),
        dedupe_pool=cfg.get("dedupe_pool", False),
        dedupe_scope=cfg.get("dedupe_scope", "archive"),
        acquisition_function=cfg.get("acquisition_function", "multi_objective_ucb"),
        ucb_beta=cfg.get("ucb_beta", 2.0),
        heads=heads,
        signs={h: 1.0 for h in heads},
    )
    loop_cfg_kwargs.update(overrides)  # method-specific ablation flags win over the shared config
    loop_cfg = LoopConfig(**loop_cfg_kwargs)

    # Calibration pool for the conformal Verifier: real toxicophore flags
    # (deterministic 0/1) on a random real sample -- not synthetic noise.
    # Because the underlying signal is a deterministic rule, score==label
    # exactly and the calibrated interval has zero width; this correctly
    # reflects that there is no residual uncertainty to calibrate against
    # a rule-based signal, not a bug. See leaf/data/raw/README.md for why
    # no probabilistic toxicity label exists to calibrate against instead.
    from leaf.agents.verifier import toxicophore_flag
    from leaf.fragments.vocabulary import assemble_lipid
    cal_frags, _ = load_agile_labeled_set()
    rng = np.random.default_rng(seed + 1000)
    cal_sample = rng.choice(len(cal_frags), size=min(50, len(cal_frags)), replace=False)
    cal_labels = np.array([
        1.0 if (lambda s: s is not None and toxicophore_flag(s))(assemble_lipid(cal_frags[i])) else 0.0
        for i in cal_sample
    ])
    cal_scores = cal_labels.copy()

    default_ref_point = [-5.0] * len(heads)
    result = run_loop(
        loop_cfg, oracle_gt, embedder,
        objective_names=heads,
        reference_point=np.array(cfg.get("reference_point", default_ref_point)),
        calibration_pool=(cal_scores, cal_labels),
        proposer=proposer,
    )

    with open(out_dir / "rounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "queries_this_round", "budget_remaining",
                          "archive_hypervolume", "empirical_coverage", "target_coverage",
                          "inter_agent_agreement", "frac_crossover", "frac_duplicate_of_archive",
                          "mean_min_edit_distance", "frac_selected_duplicate_of_archive"])
        for log in result["logs"]:
            cov = log.coverage or {}
            writer.writerow([log.round_idx, log.queries_used_this_round, log.budget_remaining,
                              log.archive_hypervolume, cov.get("empirical_coverage"),
                              cov.get("target"), log.inter_agent_agreement,
                              log.frac_crossover, log.frac_duplicate_of_archive, log.mean_min_edit_distance,
                              log.frac_selected_duplicate_of_archive])

    # Selected-query duplicate accounting (external review item #6) and
    # applicability-domain accounting (item #2): for every candidate that
    # actually charged the oracle budget, record whether it duplicated an
    # archive member at selection time and how many of its three fragment
    # values (A, B, C) were absent from this split's real oracle TRAINING
    # partition -- 0 means a fully interpolated query, 1-3 means the
    # oracle had to extrapolate to unseen reagent values on that query.
    if cfg.get("log_diagnostics", False) and result.get("query_log"):
        train_frags_for_domain, _ = load_agile_labeled_set()
        domain_splits = build_splits(train_frags_for_domain, seed=0)
        domain_train_idx, _ = domain_splits[split]
        seen_A = {train_frags_for_domain[i].A for i in domain_train_idx}
        seen_B = {train_frags_for_domain[i].B for i in domain_train_idx}
        seen_C = {train_frags_for_domain[i].C for i in domain_train_idx}
        with open(out_dir / "queries.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "A", "B", "C", "is_duplicate_of_archive", "n_unseen_fragment_values",
                              "thinking_length", "thinking_mentions_potency"])
            for q in result["query_log"]:
                n_unseen = int(q["A"] not in seen_A) + int(q["B"] not in seen_B) + int(q["C"] not in seen_C)
                writer.writerow([q["round"], q["A"], q["B"], q["C"], q["is_duplicate_of_archive"], n_unseen,
                                  q.get("thinking_length", ""), q.get("thinking_mentions_potency", "")])

    summary = {
        "experiment": experiment, "method": method, "split": split, "seed": seed,
        "final_hypervolume": result["archive"].current_hypervolume,
        "total_queries_issued": result["total_queries_issued"],
        "n_rounds": len(result["logs"]),
        # If this is an LLMProposer, reject_rate == fraction of the pool
        # that had to be backfilled by RuleBasedProposer because the model
        # returned malformed/hallucinated JSON. A high reject_rate means
        # the resulting hypervolume is NOT primarily a measurement of this
        # model's proposals -- flagged here rather than silently reported
        # as if the named method ran as intended.
        "proposer_reject_rate": getattr(proposer, "reject_rate", None),
        "hypothesis_precision": result["ledger"].precision(),
    }
    if (out_dir / "queries.csv").exists():
        import csv as _csv
        with open(out_dir / "queries.csv") as _f:
            rows = list(_csv.DictReader(_f))
        if rows:
            summary["frac_selected_duplicate_of_archive_overall"] = sum(
                r["is_duplicate_of_archive"] == "True" for r in rows) / len(rows)
            summary["mean_n_unseen_fragment_values"] = sum(
                int(r["n_unseen_fragment_values"]) for r in rows) / len(rows)
            summary["frac_queries_fully_out_of_domain"] = sum(
                int(r["n_unseen_fragment_values"]) == 3 for r in rows) / len(rows)
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{experiment}/{method}/{split}/seed{seed}] final HV={summary['final_hypervolume']:.3f} "
          f"queries={summary['total_queries_issued']} -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, help="e.g. e2 -- used as the results/ subdirectory name")
    parser.add_argument("--method", required=True, choices=sorted(list(METHOD_SPECS) + list(NOT_YET_WIRED)))
    parser.add_argument("--split", default="random")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--oracle", default=None, help="path to a frozen O_GT saved via GroundTruthOracle.save(...)")
    args = parser.parse_args()
    run_one(args.config, args.experiment, args.method, args.split, args.seed, oracle_path=args.oracle)


if __name__ == "__main__":
    main()
