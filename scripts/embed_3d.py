"""Generate real 3D conformers (ETKDG + MMFF94 optimization) for the three
best-found lipids and write them as PDB files for PyMOL to render. Runs in
the MAIN project environment (has RDKit + project code already).
"""
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

data = json.loads(Path("results/e_best_lipid_structures.json").read_text())
out_dir = Path("results/pdb_structures")
out_dir.mkdir(parents=True, exist_ok=True)

for key, entry in data.items():
    mol = Chem.MolFromSmiles(entry["smiles"])
    mol = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=True)
    if cid < 0:
        print(f"{key}: embedding failed")
        continue
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    out_path = out_dir / f"{key}.pdb"
    Chem.MolToPDBFile(mol, str(out_path))
    print(f"{key}: wrote {out_path}")
