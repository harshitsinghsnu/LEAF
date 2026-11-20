"""Leakage probe (external review): does Llama 3.1 8B -- the SAME model
used as the Proposer in the v3_scored richer-prompt check -- already have
predictive skill about real AGILE potency, zero-shot, with NO archive
context, NO vocabulary, and NO scores? If it does, an improvement from
showing it real potency scores (v3_scored, §3.3) could plausibly be
reactivating pretrained/memorized chemistry knowledge rather than
genuinely reasoning over the injected scores. If it does not (matching
the paper's existing Qwen3-8B zero-shot result, rho=0.014), that is
real evidence against the memorization explanation.

This directly implements the leakage-probe control external review
asked for: run before trusting the v3_scored result further.

Protocol: on the SAME held-out random-split test partition E1 and the
LipoAgent comparison use, ask llama3.1:8b to score each real AGILE
lipid's transfection efficiency 1-10, zero-shot, no label access, no
archive, no vocabulary list -- bare SMILES string only.

Usage: python scripts/run_leakage_probe.py [--n 50]
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from leaf.data.agile_loader import load_agile_labeled_set
from leaf.fragments.vocabulary import assemble_lipid
from leaf.oracle.eval_splits import build_splits

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"
OUT_DIR = Path("results") / "e_leakage_probe_llama8b"


def score_one(smiles: str, temperature: float = 0.0) -> float | None:
    prompt = (
        "You are an expert in lipid nanoparticle (LNP) design for mRNA delivery. "
        "Score the following ionizable lipid's likely transfection efficiency on a scale "
        "of 1 (very poor) to 10 (excellent), based on its structure. "
        "Respond with ONLY a single JSON object: {\"efficiency_score\": <integer 1-10>}.\n\n"
        f"Lipid SMILES: {smiles}"
    )
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "format": "json", "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate", data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "leaf-research/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        raw = result["response"]
        m = re.search(r'"efficiency_score"\s*:\s*(\d+)', raw)
        if m:
            return float(m.group(1))
        m = re.search(r"\b([1-9]|10)\b", raw)
        return float(m.group(1)) if m else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of held-out test molecules to score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    frags, labels = load_agile_labeled_set()
    splits = build_splits(frags, seed=args.seed)
    _, test_idx = splits["random"]

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(test_idx, size=min(args.n, len(test_idx)), replace=False)

    scores, true_potency, reject_count = [], [], 0
    for i in sample_idx:
        smi = assemble_lipid(frags[i])
        if smi is None:
            continue
        s = score_one(smi, temperature=args.temperature)
        if s is None:
            reject_count += 1
            continue
        scores.append(s)
        true_potency.append(labels["potency"][i])

    n_used = len(scores)
    if n_used < 5:
        print(f"Only {n_used} usable scores -- aborting, not enough for a correlation.")
        return

    unique_scores = sorted(set(scores))
    degenerate = len(unique_scores) == 1
    if degenerate:
        rho, pval = float("nan"), float("nan")
    else:
        rho, pval = spearmanr(scores, true_potency)
    result = {
        "model": MODEL,
        "n_scored": n_used,
        "n_requested": len(sample_idx),
        "reject_count": reject_count,
        "degenerate_constant_output": degenerate,
        "unique_scores_emitted": unique_scores,
        "spearman_vs_real_potency": float(rho) if not degenerate else None,
        "p_value": float(pval),
        "comparison_O_GT_random_split_spearman": 0.6834639696547492,
        "comparison_qwen3_8b_zeroshot_spearman": 0.014,
        "temperature": args.temperature,
    }
    out_path = OUT_DIR / f"result_temp{args.temperature}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
