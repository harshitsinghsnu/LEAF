"""E-LipoAgent: does LipoAgent's own approach (zero-shot LLM efficiency
scoring, no training) predict real transfection potency as well as LEAF's
fitted O_GT (a Gaussian Process on Morgan fingerprints)?

LipoAgent (Li et al., ACL 2026; github.com/SAI-Lab-NYU/LipoAgent) scores
candidate lipids 1-10 for transfection efficiency using a Predict Agent
built on TxGemma-27B-Chat and Qwen3-8B, with no training on labeled data
-- the model's pretrained chemistry knowledge is the whole mechanism.
TxGemma is inaccessible here (HuggingFace gating + 16GB VRAM
insufficient, see paper §7 item 6); Qwen3-8B is not, and is one of
LipoAgent's own two base models, so this is a real, direct test of one
half of their actual approach, not a stand-in.

Protocol: on the SAME held-out random-split test partition E1 uses (so
the comparison is apples-to-apples with the paper's own O_GT numbers),
ask Qwen3:8B to score each real AGILE lipid's transfection efficiency
1-10, zero-shot, no training, no access to the label. Compute Spearman
correlation between its scores and real expt_Hela potency, and report it
directly against O_GT's real, already-measured E1 number (0.68, random
split, potency).

Usage: python scripts/run_lipoagent_comparison.py [--n 50]
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
MODEL = "qwen3:8b"
OUT_PATH = Path("results") / "e_lipoagent_comparison" / "result.json"


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
        # fall back to any bare integer in range if the model didn't follow the schema exactly
        m = re.search(r"\b([1-9]|10)\b", raw)
        return float(m.group(1)) if m else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of held-out test molecules to score")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frags, labels = load_agile_labeled_set()
    splits = build_splits(frags, seed=args.seed)
    _, test_idx = splits["random"]  # same partition E1 reports its 0.68 Spearman on

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(test_idx, size=min(args.n, len(test_idx)), replace=False)

    scores, true_potency, reject_count = [], [], 0
    for i in sample_idx:
        smi = assemble_lipid(frags[i])
        if smi is None:
            continue
        s = score_one(smi)
        if s is None:
            reject_count += 1
            continue
        scores.append(s)
        true_potency.append(labels["potency"][i])

    n_used = len(scores)
    if n_used < 5:
        print(f"Only {n_used} usable scores -- aborting, not enough for a correlation.")
        return

    rho, pval = spearmanr(scores, true_potency)
    result = {
        "model": MODEL,
        "n_scored": n_used,
        "n_requested": len(sample_idx),
        "reject_count": reject_count,
        "spearman_vs_real_potency": float(rho),
        "p_value": float(pval),
        "comparison_O_GT_random_split_spearman": 0.6834639696547492,  # real E1 number, results/e1/split_validation_table.csv
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
