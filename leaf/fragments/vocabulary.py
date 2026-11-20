"""Fragment vocabulary enumeration and validity checking.

A candidate lipid is a tuple x = (A, B, C) of the three Ugi
three-component-reaction reactants: A = primary amine (-NH2 head group),
B = aldehyde (-CHO, contributes the linker/backbone carbon), C =
isocyanide (-NC, contributes a lipid tail). This replaces the earlier
illustrative 6-slot placeholder (head/linker/tail1/tail2/unsaturation/
halogenation) with the REAL combinatorial structure of the AGILE dataset
(Xu et al., Nat. Commun. 2024), which is itself the full Ugi-3CR library
originally described in Xu et al. (2023) -- see data/raw/README.md.

The assembly logic in `assemble_lipid` below is reverse-engineered from
AGILE's own public notebook (`ugi_3CR_agile_reaction_virtual.ipynb`,
github.com/bowang-lab/AGILE) and validated against all 1,200 known
(A, B, C) -> combined_mol_SMILES triples in the real dataset --
see `leaf/tests/test_vocabulary.py::test_assembler_matches_known_agile_molecules`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolEnumerator

AGILE_CSV = Path(__file__).parent.parent / "data" / "raw" / "agile" / "AGILE_smiles_with_value_group.csv"
# Real, published AGILE virtual candidate library (12,276 lipids, MIT
# license, github.com/bowang-lab/AGILE's data.zip ->
# candidate_set_smiles_plus_features.csv), decomposed into (A,B,C) triples
# via decompose_lipid() below (100% decomposition success, validated against
# the 1,200 known triples with an exact round-trip -- see test_vocabulary.py).
# Provides 9 new real A values, 62 new B, 9 new C beyond the original
# 20/12/5 -- see leaf/data/raw/README.md for the full provenance writeup.
AGILE_EXTENDED_CANDIDATES = (
    Path(__file__).parent.parent / "data" / "raw" / "agile_extended" / "candidate_triples.json"
)

# Ugi-3CR core scaffold and attachment points, exactly as used by AGILE's
# own virtual-library generator: NCC(N)=O, with component A attaching at
# atom 0 (the free amine nitrogen), B at atom 1 (the alpha carbon), and C
# at atom 3 (the amide nitrogen).
CORE_SMILES = "NCC(N)=O"
CORE_A_POS = 0
CORE_B_POS = 1
CORE_C_POS = 3

_FRAG_NH2 = Chem.MolFromSmarts("[NH2]")
_FRAG_CHO = Chem.MolFromSmarts("[CX3H1](=O)")
_FRAG_NC = Chem.MolFromSmarts("N#C")
_STAR = Chem.MolFromSmiles("*")


@dataclass(frozen=True)
class Fragment:
    """A = amine SMILES, B = aldehyde SMILES, C = isocyanide SMILES."""
    A: str
    B: str
    C: str

    def as_tuple(self) -> tuple:
        return (self.A, self.B, self.C)


@lru_cache(maxsize=1)
def _load_agile_components() -> tuple[list[str], list[str], list[str]]:
    """Reads the unique A/B/C reactant SMILES directly from the real,
    vendored AGILE dataset -- this IS the literature-derived vocabulary
    (research plan §5.7), not a hand-picked illustrative subset."""
    if not AGILE_CSV.exists():
        raise FileNotFoundError(
            f"AGILE dataset not found at {AGILE_CSV}. See leaf/data/raw/README.md "
            "for provenance and how to re-vendor it (github.com/bowang-lab/AGILE, MIT license)."
        )
    a_set, b_set, c_set = [], [], []
    seen_a, seen_b, seen_c = set(), set(), set()
    with open(AGILE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["A_smiles"] not in seen_a:
                seen_a.add(row["A_smiles"]); a_set.append(row["A_smiles"])
            if row["B_smiles"] not in seen_b:
                seen_b.add(row["B_smiles"]); b_set.append(row["B_smiles"])
            if row["C_smiles"] not in seen_c:
                seen_c.add(row["C_smiles"]); c_set.append(row["C_smiles"])
    return a_set, b_set, c_set


@lru_cache(maxsize=1)
def _load_extended_components() -> tuple[list[str], list[str], list[str]]:
    """Merges the original 20/12/5 labeled-set components with the real,
    additional reagent values found by decomposing AGILE's own published
    12,276-lipid virtual candidate library (see AGILE_EXTENDED_CANDIDATES
    above). Design decision, disclosed rather than silently assumed: the
    Ugi three-component reaction applies uniformly to any (amine,
    aldehyde, isocyanide) triple, so freely recombining this extended,
    still entirely real reagent pool is chemically legitimate -- it is
    the same published reaction, not a restriction to only the specific
    cross-combinations AGILE happened to list. Only the ORIGINAL 1,200
    (the labeled factorial subset) have real experimental labels; every
    other combination in the resulting larger space is real chemistry
    with NO label, which is what finally makes B << |X| meaningful
    (research plan RQ6) -- see leaf/data/raw/README.md.
    """
    # Copy before mutating: _load_agile_components() is lru_cache'd and
    # returns the SAME list objects on every call. Appending directly to
    # them (an earlier version of this function did) silently corrupted
    # the cached "labeled-only" lists for every future caller, including
    # labeled_component_lists() -- caught by
    # test_labeled_vocabulary_is_the_original_factorial_design.
    orig_a, orig_b, orig_c = _load_agile_components()
    a_list, b_list, c_list = list(orig_a), list(orig_b), list(orig_c)
    a_set, b_set, c_set = set(a_list), set(b_list), set(c_list)
    if AGILE_EXTENDED_CANDIDATES.exists():
        import json
        with open(AGILE_EXTENDED_CANDIDATES, encoding="utf-8") as f:
            for row in json.load(f):
                if row["A"] not in a_set:
                    a_set.add(row["A"]); a_list.append(row["A"])
                if row["B"] not in b_set:
                    b_set.add(row["B"]); b_list.append(row["B"])
                if row["C"] not in c_set:
                    c_set.add(row["C"]); c_list.append(row["C"])
    return a_list, b_list, c_list


def component_lists() -> tuple[list[str], list[str], list[str]]:
    """Returns the EXTENDED component lists (original + real candidate-
    library reagents) when available, falling back to the original
    20/12/5 if the extended candidate file hasn't been generated. This is
    what Proposer/GA/RandomSearch draw from -- see _load_extended_components
    for why extending is chemically legitimate, not fabricated."""
    return _load_extended_components()


def labeled_component_lists() -> tuple[list[str], list[str], list[str]]:
    """The original 20/12/5 components from the labeled AGILE set only --
    use this (not component_lists()) anywhere that must stay grounded to
    what O_GT was actually fit on (e.g. data/agile_loader.py)."""
    return _load_agile_components()


def enumerate_vocabulary() -> list[Fragment]:
    """Enumerate the full combinatorial fragment space: every (A, B, C)
    triple from the EXTENDED real component lists (see
    _load_extended_components). Only the 1,200 lipids from the original
    factorial subset have real experimental labels; the rest of this
    (much larger) space is real, assemblable chemistry with no label --
    this is what makes RQ6's B << |X| condition meaningful."""
    a_list, b_list, c_list = component_lists()
    return [Fragment(a, b, c) for a in a_list for b in b_list for c in c_list]


def vocabulary_size() -> int:
    a_list, b_list, c_list = component_lists()
    return len(a_list) * len(b_list) * len(c_list)


def _strip_leaving_group(raw_smiles: str, smarts_frag) -> "Chem.Mol | None":
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None or not mol.HasSubstructMatch(smarts_frag):
        return None
    return Chem.ReplaceSubstructs(mol, smarts_frag, _STAR)[0]


def assemble_lipid(frag: Fragment) -> str | None:
    """Exact Ugi three-component-reaction assembly, replicating AGILE's
    own generation notebook. Returns the assembled molecule's canonical
    SMILES, or None if any component's reactive group can't be found/
    stripped, or the resulting attachment-point bookkeeping doesn't hold
    (mirrors the same assumptions AGILE's own code makes -- a component
    for which this fails is treated as unassemblable, not silently wrong).
    """
    core_mol = Chem.MolFromSmiles(CORE_SMILES)
    core_num_atoms = core_mol.GetNumAtoms()

    a_frag = _strip_leaving_group(frag.A, _FRAG_NH2)
    b_frag = _strip_leaving_group(frag.B, _FRAG_CHO)
    c_frag = _strip_leaving_group(frag.C, _FRAG_NC)
    if a_frag is None or b_frag is None or c_frag is None:
        return None

    a_smiles, b_smiles, c_smiles = (Chem.MolToSmiles(m) for m in (a_frag, b_frag, c_frag))
    if not (a_smiles.startswith("*") and b_smiles.startswith("*") and c_smiles.startswith("*")):
        # AGILE's own generator assumes the dummy atom canonicalizes to
        # the first atom of each fragment's SMILES; when that assumption
        # fails the atom-index bookkeeping below would be wrong, so treat
        # it as an unassemblable candidate rather than silently mis-assembling.
        return None

    star_a_pos = core_num_atoms
    star_b_pos = core_num_atoms + a_frag.GetNumAtoms()
    star_c_pos = core_num_atoms + a_frag.GetNumAtoms() + b_frag.GetNumAtoms()

    combo = Chem.MolFromSmiles(
        f"{CORE_SMILES}.{a_smiles}.{b_smiles}.{c_smiles} "
        f"|m:{star_a_pos}:{CORE_A_POS},{star_b_pos}:{CORE_B_POS},{star_c_pos}:{CORE_C_POS}|"
    )
    if combo is None:
        return None
    enumerated = rdMolEnumerator.Enumerate(combo)
    if len(enumerated) != 1:
        return None
    return Chem.MolToSmiles(enumerated[0])


def fragment_to_smiles(frag: Fragment) -> str | None:
    return assemble_lipid(frag)


_CORE_MOL = Chem.MolFromSmiles(CORE_SMILES)
_REATTACH_A = Chem.MolFromSmiles("N")             # inverse of stripping [NH2] (1 atom)
_REATTACH_B = Chem.MolFromSmiles("C=O")           # inverse of stripping [CX3H1](=O) (2 atoms)
_REATTACH_C = Chem.MolFromSmiles("[N+]#[C-]")     # inverse of stripping N#C (2 atoms), real isocyanide charge form


def decompose_lipid(smiles: str) -> Fragment | None:
    """Inverse of `assemble_lipid`: given an assembled Ugi-3CR product's
    SMILES, recover the (A, B, C) reactant triple that would produce it.

    Used to extend the real vocabulary beyond AGILE's original 1,200
    labeled lipids using AGILE's own real, published ~12,277-candidate
    virtual library (data/candidate_set_smiles_plus_features.csv in their
    repo) -- see research plan §5.7/RQ6. Returns None if the molecule
    doesn't contain the expected core scaffold in a decomposable form
    (treated as "not part of this Ugi-3CR series," not silently guessed).

    Validated to exactly round-trip all 1,200 known real (A,B,C) triples
    -- see leaf/tests/test_vocabulary.py::test_decompose_inverts_assemble.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    match = mol.GetSubstructMatch(_CORE_MOL)
    if not match:
        return None  # not a member of this Ugi-3CR core-scaffold series

    core_atom_ids = set(match)
    slot_attach_atoms = {"A": match[CORE_A_POS], "B": match[CORE_B_POS], "C": match[CORE_C_POS]}

    bond_ids = []
    slot_by_bond = {}
    for slot, attach_idx in slot_attach_atoms.items():
        atom = mol.GetAtomWithIdx(attach_idx)
        external_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in core_atom_ids]
        if len(external_neighbors) != 1:
            return None  # ambiguous or missing attachment -- not cleanly decomposable
        bond = mol.GetBondBetweenAtoms(attach_idx, external_neighbors[0])
        bond_ids.append(bond.GetIdx())
        slot_by_bond[bond.GetIdx()] = slot

    fragmented = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True)
    frags = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=None)
    piece_atom_maps = Chem.GetMolFrags(fragmented, asMols=False)  # tuples of original atom indices per fragment

    # Identify each fragment's slot by checking which external attachment
    # atom (the neighbor, not the core atom) it contains.
    slot_frag: dict[str, "Chem.Mol"] = {}
    for frag_mol, atom_ids in zip(frags, piece_atom_maps):
        atom_id_set = set(atom_ids)
        for slot, attach_idx in slot_attach_atoms.items():
            atom = mol.GetAtomWithIdx(attach_idx)
            external_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in core_atom_ids]
            if external_neighbors and external_neighbors[0] in atom_id_set and attach_idx not in atom_id_set:
                slot_frag[slot] = frag_mol
    if set(slot_frag) != {"A", "B", "C"}:
        return None

    reattach = {"A": _REATTACH_A, "B": _REATTACH_B, "C": _REATTACH_C}
    result = {}
    for slot, frag_mol in slot_frag.items():
        try:
            Chem.SanitizeMol(frag_mol, catchErrors=True)
            rebuilt = Chem.ReplaceSubstructs(frag_mol, _STAR, reattach[slot], replaceAll=True)[0]
            Chem.SanitizeMol(rebuilt)
            result[slot] = Chem.MolToSmiles(rebuilt)
        except (Chem.rdchem.MolSanitizeException, ValueError, IndexError):
            return None

    return Fragment(A=result["A"], B=result["B"], C=result["C"])


def is_valid(frag: Fragment) -> bool:
    """Round-trip a fragment tuple through the real Ugi-3CR assembler and
    RDKit parsing. Must be 100% for the full vocabulary before Step 1's
    gate is satisfied (research plan §6.1) -- unlike the earlier
    placeholder, failure here means the assembler's chemistry is wrong,
    not that RDKit is missing."""
    smiles = assemble_lipid(frag)
    if smiles is None:
        return False
    return Chem.MolFromSmiles(smiles) is not None
