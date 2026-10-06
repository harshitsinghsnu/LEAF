"""Run exactly the next not-yet-completed seed of the Groq Llama 3.3 70B
comparison (leaf_llm_groq70b, random split), one per invocation -- meant
to be called once/day so the real daily token quota (100K/day on the free
tier, ~65K/seed-run at vocab_sample_size=8, see paper §6.2/§7 item 5)
naturally paces the accumulation of a full n>=8 sample without any
single run exceeding the day's budget.

Usage: python scripts/run_next_groq_seed.py
Idempotent: if today's seed already ran, does nothing (checks results/ on
disk rather than any separate state file).
"""
from __future__ import annotations

from pathlib import Path

from leaf.eval.run_experiment import run_one

EXPERIMENT = "e2_llm_groq70b_daily"
SPLIT = "random"
MAX_SEEDS = 8
RESULTS_ROOT = Path("results") / EXPERIMENT


def _completed_seeds(method: str) -> set[int]:
    done = set()
    for s in range(MAX_SEEDS):
        if (RESULTS_ROOT / f"{method}_{SPLIT}_seed{s}" / "result.json").exists():
            done.add(s)
    return done


def main() -> None:
    done = _completed_seeds("leaf_llm_groq70b")
    remaining = [s for s in range(MAX_SEEDS) if s not in done]
    if not remaining:
        print(f"All {MAX_SEEDS} seeds already completed for {EXPERIMENT}. Nothing to do.")
        return
    seed = remaining[0]
    print(f"Running seed {seed} ({len(done)}/{MAX_SEEDS} already done)...")
    run_one("leaf/configs/e2_llm_groq70b_pilot.yaml", EXPERIMENT, "leaf_llm_groq70b", SPLIT, seed)
    # Baselines are free/local -- run them alongside so every seed has a
    # matched static-GA/random-search comparison from the start.
    run_one("leaf/configs/e2_llm_groq70b_pilot.yaml", EXPERIMENT, "static_ga", SPLIT, seed)
    run_one("leaf/configs/e2_llm_groq70b_pilot.yaml", EXPERIMENT, "random_search", SPLIT, seed)
    print(f"Seed {seed} done. {len(done) + 1}/{MAX_SEEDS} complete.")
    if len(done) + 1 == MAX_SEEDS:
        print("All seeds complete -- ready for plotting.plot_agentic_value_add(...) at full n=8 power.")


if __name__ == "__main__":
    main()
