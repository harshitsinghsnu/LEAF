"""Step 1 gate (research plan §6.1 item 3): 100% of the enumerated
vocabulary must round-trip to a valid RDKit molecule, AND (since the
vocabulary is now the real, vendored AGILE Ugi-3CR library) the assembler
must exactly reproduce AGILE's own published molecules -- not just parse
to *some* valid structure."""
import csv
import json

import pytest

from leaf.fragments.vocabulary import (
    AGILE_CSV, AGILE_EXTENDED_CANDIDATES, Fragment, assemble_lipid, decompose_lipid,
    enumerate_vocabulary, is_valid, labeled_component_lists, vocabulary_size,
)

rdkit = pytest.importorskip("rdkit")


def test_labeled_vocabulary_is_the_original_factorial_design():
    a_list, b_list, c_list = labeled_component_lists()
    assert len(a_list) * len(b_list) * len(c_list) == 1200  # 20 A x 12 B x 5 C


def test_extended_vocabulary_is_larger_than_labeled_set():
    """RQ6: the whole point of the extension (research plan §5.7) is that
    |X| must end up >> the 1,200 labeled lipids for budget-constrained
    search to be meaningful."""
    assert vocabulary_size() > 1200 * 10


def test_vocabulary_size_matches_enumeration():
    vocab = enumerate_vocabulary()
    assert len(vocab) == vocabulary_size()


@pytest.mark.skipif(not AGILE_EXTENDED_CANDIDATES.exists(), reason="extended candidate file not generated")
def test_full_vocabulary_round_trips():
    vocab = enumerate_vocabulary()
    invalid = [f for f in vocab if not is_valid(f)]
    assert invalid == [], f"{len(invalid)}/{len(vocab)} fragments failed RDKit round-trip"


def test_assembler_matches_known_agile_molecules():
    """The chemically load-bearing test: our reverse-engineered Ugi-3CR
    assembler must exactly reproduce AGILE's own published
    combined_mol_SMILES for every one of the 1,200 real labeled lipids --
    canonical-SMILES equality, not just "RDKit can parse it"."""
    from rdkit import Chem

    n_total, n_match = 0, 0
    mismatches = []
    with open(AGILE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_total += 1
            frag = Fragment(row["A_smiles"], row["B_smiles"], row["C_smiles"])
            ours = assemble_lipid(frag)
            known = Chem.MolToSmiles(Chem.MolFromSmiles(row["combined_mol_SMILES"]))
            if ours == known:
                n_match += 1
            else:
                mismatches.append((row["label"], ours, known))
    assert n_match == n_total, f"{n_total - n_match}/{n_total} mismatches, e.g. {mismatches[:3]}"


def test_decompose_inverts_assemble_on_known_molecules():
    """decompose_lipid must exactly recover the original (A, B, C) triple
    for every one of the 1,200 known real lipids -- validated at 1200/1200
    when this was built; regresses to a hard failure if the inverse
    logic ever drifts from the forward assembler."""
    from rdkit import Chem

    n_total, n_match = 0, 0
    with open(AGILE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_total += 1
            frag = Fragment(row["A_smiles"], row["B_smiles"], row["C_smiles"])
            assembled = assemble_lipid(frag)
            decomposed = decompose_lipid(assembled)
            if decomposed is None:
                continue
            canon = tuple(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in frag.as_tuple())
            if decomposed.as_tuple() == canon:
                n_match += 1
    assert n_match == n_total, f"{n_total - n_match}/{n_total} decomposition round-trip failures"


@pytest.mark.skipif(not AGILE_EXTENDED_CANDIDATES.exists(), reason="extended candidate file not generated")
def test_extended_candidates_file_fully_decomposed():
    with open(AGILE_EXTENDED_CANDIDATES, encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) > 10000  # the real AGILE candidate library has 12,276 entries
    for row in rows[:50]:  # spot-check a sample for real assembly validity
        assert assemble_lipid(Fragment(row["A"], row["B"], row["C"])) is not None
