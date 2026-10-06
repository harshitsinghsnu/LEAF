"""Builds a real manifest mapping every reported table/figure in the paper
to its exact source config file, results directory, and the script that
reproduces it -- external review item #6 (reproducibility). Run from the
repo root: python reproducibility_package/build_manifest.py
"""
import json
from pathlib import Path

MANIFEST = {
    "Table 1 (objective specification)": {
        "source": "leaf/orchestrator/loop.py (LoopConfig defaults) + every leaf/configs/*.yaml",
        "verification": "grep for 'heads', 'signs', 'reference_point' across all configs; none override the defaults",
    },
    "Table 2 / Figure 2B (primary four-model sweep, n=8, all splits)": {
        "configs": ["leaf/configs/e2_llm_full.yaml", "leaf/configs/e2_llm_qwen_full.yaml",
                    "leaf/configs/e2_llm_qwen14b_full.yaml", "leaf/configs/e2_llm_gemini_full.yaml"],
        "results_dirs": ["results/e2_llm_full", "results/e2_llm_qwen_full",
                          "results/e2_llm_qwen14b_full", "results/e2_llm_gemini_full"],
        "reproduce": "python -m leaf.eval.run_experiment --config <cfg> --experiment <name> --method <m> --split <s> --seed <n> for each (method, split, seed)",
    },
    "Table 3 (matched de-duplication, all splits)": {
        "configs": ["leaf/configs/e2_dedup_check_full8_control.yaml", "leaf/configs/e2_dedup_check_full8.yaml",
                    "leaf/configs/e2_dedup_check_otherSplits_control.yaml", "leaf/configs/e2_dedup_check_otherSplits.yaml"],
        "results_dirs": ["results/e2_dedup_check_full8_control", "results/e2_dedup_check_full8",
                          "results/e2_dedup_check_other_splits_control", "results/e2_dedup_check_other_splits"],
    },
    "EHVI exercise (Section 3.4)": {
        "configs": ["leaf/configs/e2_ehvi_check.yaml"],
        "results_dirs": ["results/e2_ehvi_check"],
    },
    "Selected-query duplicate + applicability-domain instrumentation (Section 3.3)": {
        "configs": ["leaf/configs/e2_llm_full_instrumented.yaml"],
        "results_dirs": ["results/e2_llm_full_instrumented"],
        "note": "per-query CSV at results/e2_llm_full_instrumented/<method>_random_seed<n>/queries.csv",
    },
    "Table 9 (de-duplication pilot, B=15, n=4)": {
        "configs": ["leaf/configs/e2_dedup_check.yaml", "leaf/configs/e2_dedup_check_control.yaml"],
        "results_dirs": ["results/e2_dedup_check", "results/e2_dedup_check_control"],
    },
    "Table 11 (Thompson bandit)": {
        "configs": ["leaf/configs/e2_bandit_baseline.yaml"],
        "results_dirs": ["results/e2_bandit_baseline"],
    },
    "Appendix G (normalized hypervolume, full-space HV per split)": {
        "script": "computed directly against every split's frozen O_GT over all 30,044 candidates; no search or LLM calls",
        "note": "no persisted script filename yet -- see conversation record; values: random=258.2, scaffold_cold=235.9, chemotype_cold=199.5",
    },
    "Split indices (random/scaffold_cold/chemotype_cold train/test partitions)": {
        "source": "leaf/oracle/eval_splits.py build_splits(frags, seed=0)",
        "exported_to": "reproducibility_package/split_indices.json",
    },
    "GP hyperparameters": {
        "source": "leaf/oracle/gp_model.py MultiHeadGP: TanimotoKernel, alpha=1e-6, normalize_y=True, random_state=0",
    },
}

Path("reproducibility_package").mkdir(exist_ok=True)
with open("reproducibility_package/manifest.json", "w") as f:
    json.dump(MANIFEST, f, indent=2)
print("wrote reproducibility_package/manifest.json")
