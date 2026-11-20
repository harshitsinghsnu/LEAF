"""E1: Oracle validation (RQ5, RQ6) -- research plan §7.

Uses the REAL, vendored AGILE dataset (leaf/data/raw/agile/, see
leaf/data/raw/README.md for provenance) -- real Ugi-3CR fragment
assembly (validated 1200/1200 exact match against AGILE's own published
molecules), real Morgan-fingerprint embedding, and real experimental
labels (expt_Hela, expt_Raw). No synthetic data is used here.

E1 does not run the round loop; it fits O_GT and evaluates it on
random/scaffold-cold/chemotype-cold splits, applying the pre-registered
go/no-go threshold (§5.11). This MUST be run, and its gate satisfied,
before any E2-E8 result is treated as reportable (§6.1 Step 2 gate).

Usage:
    python scripts/run_e1_oracle_validation.py --config leaf/configs/e1_oracle_validation.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from leaf.data.agile_loader import load_agile_labeled_set
from leaf.eval.reproducibility import RunManifest, warn_if_dirty
from leaf.fragments.embed import FragmentEmbedder
from leaf.fragments.vocabulary import vocabulary_size
from leaf.oracle.eval_splits import build_splits, evaluate_split
from leaf.oracle.gp_model import MultiHeadGP
from leaf.oracle.ground_truth import GroundTruthOracle

RESULTS_ROOT = Path("results") / "e1"
FIGURES_ROOT = Path("figures") / "e1"
FROZEN_ORACLE_PATH = Path("leaf/data/processed/oracle_gt.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="leaf/configs/e1_oracle_validation.yaml")
    parser.add_argument("--freeze-oracle", action="store_true",
                         help="After split validation, also fit O_GT on the FULL real "
                              "dataset and freeze/save it to leaf/data/processed/oracle_gt.pkl "
                              "for use as --oracle in E2-E6.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(experiment="e1", method="oracle_only", split="all", seed=0, config=cfg)
    warn_if_dirty(manifest)
    manifest.save(RESULTS_ROOT / "manifest.json")

    print(f"|X| (full fragment vocabulary) = {vocabulary_size()}")

    frags, labels = load_agile_labeled_set()
    n_labeled = len(frags)
    n_unlabeled = vocabulary_size() - n_labeled
    print(f"Real labeled AGILE set: {n_labeled} lipids; "
          f"{n_unlabeled} additional real, assemblable, UNLABELED combinations "
          f"(B/|X| for a budget of e.g. 80 = {80 / vocabulary_size():.4f})")
    if vocabulary_size() <= n_labeled * 5:
        print("NOTE (RQ6): |X| is still not much larger than the labeled set -- "
              "B << |X| may not hold meaningfully. See LEAF_Research_Plan.md.")
    else:
        print(f"NOTE (RQ6, resolved): the vocabulary now includes real reagents from AGILE's "
              f"own published ~12,276-candidate virtual library (decomposed via decompose_lipid(), "
              f"validated 100% against the 1,200 known triples), extending |X| from 1,200 to "
              f"{vocabulary_size()} -- {n_unlabeled} real, chemically-legitimate combinations "
              f"(same Ugi-3CR reaction, real reagents) have NO experimental label. O_GT is fit "
              f"only on the original {n_labeled} labeled points and must genuinely extrapolate to "
              f"score anything else -- see leaf/data/raw/README.md for full provenance.")

    embedder = FragmentEmbedder()
    X = embedder.embed_batch(frags)
    heads = cfg.get("heads", ["potency", "potency_raw"])
    Y = {h: labels[h] for h in heads if h in labels}

    splits = build_splits(frags, seed=0)
    rows = []
    for split_name in cfg["splits"]:
        train_idx, test_idx = splits[split_name]
        for head in Y:
            gp = MultiHeadGP(heads=[head]).fit(X[train_idx], {head: Y[head][train_idx]})
            y_pred = gp.predict(X[test_idx], head, return_std=False)
            result = evaluate_split(Y[head][test_idx], y_pred, split_name, head,
                                     min_spearman=cfg["min_spearman_go_no_go"])
            rows.append(vars(result))
            status = "INTERPRETABLE" if result.interpretable else "BELOW THRESHOLD -- UNINTERPRETABLE"
            print(f"[{split_name}/{head}] Spearman={result.spearman:.3f} Pearson={result.pearson:.3f} "
                  f"n_train={len(train_idx)} n_test={result.n_test} -> {status}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "split_validation_table.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for head in Y:
        sub = df[df["head"] == head]
        ax.plot(sub["split_name"], sub["spearman"], marker="o", label=head)
    ax.axhline(cfg["min_spearman_go_no_go"], color="red", linestyle="--", label="go/no-go threshold")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("O_GT generalization across splits (E1 / RQ5) -- real AGILE data")
    ax.legend()
    fig.savefig(FIGURES_ROOT / "oracle_split_validation.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / "oracle_split_validation.pdf", bbox_inches="tight")

    (RESULTS_ROOT / "search_space_size.json").write_text(json.dumps({
        "vocabulary_size": vocabulary_size(),
        "n_real_labeled": n_labeled,
        "budget_vs_space_note": "vocabulary_size == n_real_labeled: no genuinely unlabeled "
                                 "search space with this single data source (RQ6)",
    }, indent=2))

    print(f"\nWrote {RESULTS_ROOT}/split_validation_table.csv and {FIGURES_ROOT}/oracle_split_validation.png")
    if not df["interpretable"].all():
        print("\nWARNING: at least one (split, head) combination is below the pre-registered "
              "go/no-go threshold. Per research plan §5.11, downstream results on that split "
              "must be reported as uninterpretable, not as findings.")

    if args.freeze_oracle:
        print("\nFitting final O_GT on the FULL real dataset (all "
              f"{n_labeled} labeled lipids) and freezing...")
        FROZEN_ORACLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        oracle_gt = GroundTruthOracle(heads=list(Y.keys())).fit(X, Y).freeze()
        oracle_gt.save(FROZEN_ORACLE_PATH)
        print(f"Saved frozen O_GT to {FROZEN_ORACLE_PATH}. Pass "
              f"--oracle {FROZEN_ORACLE_PATH} to run_all_experiments.py / run_experiment.py "
              "for E2-E6 to use the real, frozen oracle instead of a per-split refit.")


if __name__ == "__main__":
    main()
