"""Loader for the real, vendored AGILE dataset.

See leaf/data/raw/README.md for provenance (github.com/bowang-lab/AGILE,
MIT license), column definitions, and the checksum of the vendored CSV.
"""
from __future__ import annotations

import csv

import numpy as np

from leaf.fragments.vocabulary import Fragment, AGILE_CSV


def load_agile_labeled_set() -> tuple[list[Fragment], dict[str, np.ndarray]]:
    """Returns (fragments, labels) for all 1,200 real AGILE-labeled lipids
    (a full factorial 20 x 12 x 5 design -- every enumerable (A, B, C)
    combination in the current vocabulary already has a real label; see
    fragments/vocabulary.py and the RQ6 note in the research plan about
    what this means for the "budget << search space" framing).

    labels contains two REAL experimental transfection-efficiency arrays:
    'potency' (expt_Hela) and 'potency_raw' (expt_Raw). There is no
    toxicity/viability assay in this dataset -- see data/raw/README.md.
    """
    fragments: list[Fragment] = []
    hela, raw = [], []
    with open(AGILE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fragments.append(Fragment(row["A_smiles"], row["B_smiles"], row["C_smiles"]))
            hela.append(float(row["expt_Hela"]))
            raw.append(float(row["expt_Raw"]))
    return fragments, {"potency": np.array(hela), "potency_raw": np.array(raw)}
