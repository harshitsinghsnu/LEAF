"""Literature-lipid face-validity check: does the real decompose_lipid()
function (validated to round-trip all 1,200 AGILE lipids) recognize any
of the three most widely used clinical/literature ionizable lipids as
members of this Ugi-3CR series? If not (expected, since MC3/SM-102/
ALC-0315 are not Ugi-3CR products), this converts the paper's previously
"planned but not reported" Recall@k item into an honestly reported
negative result about domain coverage, rather than leaving it silently
unreported.

Canonical SMILES taken from PubChem, cross-verified by computing the
molecular formula from each SMILES with RDKit and matching it exactly
against the formula independently listed on each compound's own PubChem
page (not just a search-engine summary):
  - MC3 (DLin-MC3-DMA): PubChem CID 49785164, formula C43H79NO2 (matched)
  - SM-102: PubChem CID 126697616, formula C44H87NO5 (matched)
  - ALC-0315: PubChem CID 122666778, formula C48H95NO5 (matched)
"""
import json
from pathlib import Path

from rdkit import Chem
from leaf.fragments.vocabulary import decompose_lipid

LIPIDS = {
    "MC3 (DLin-MC3-DMA)": "CCCCC/C=C\\C/C=C\\CCCCCCCCC(CCCCCCCC/C=C\\C/C=C\\CCCCC)OC(=O)CCCN(C)C",
    "SM-102": "CCCCCCCCCCCOC(=O)CCCCCN(CCCCCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCO",
    "ALC-0315": "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCN(CCCCCCOC(=O)C(CCCCCC)CCCCCCCC)CCCCO",
}

out = {}
for name, smi in LIPIDS.items():
    mol = Chem.MolFromSmiles(smi)
    valid_smiles = mol is not None
    decomposed = decompose_lipid(smi) if valid_smiles else None
    out[name] = {
        "input_smiles": smi,
        "rdkit_parses": valid_smiles,
        "decomposes_as_ugi3cr": decomposed is not None,
        "recovered_fragment": None if decomposed is None else {
            "A": decomposed.A, "B": decomposed.B, "C": decomposed.C,
        },
    }
    print(f"{name}: parses={valid_smiles}, decomposes_as_Ugi3CR={decomposed is not None}")

Path("results/e_literature_lipid_check.json").write_text(json.dumps(out, indent=2))
