"""Fragment tuple -> vector embedding.

Real Morgan (ECFP-style) fingerprint of the assembled molecule, replacing
the earlier one-hot placeholder.

A direct check on the real, vendored AGILE set found that a bare radius-2
Morgan fingerprint collapses 1,200 structurally distinct real lipids down
to only 220 unique bit vectors (980 exact collisions) -- increasing bit
width (1024 -> 2048) does nothing, since the collisions aren't hash
folding, they're a fundamental property of circular fingerprints: this
chemical series varies mostly in aliphatic tail *length*, a global
property outside a radius-2 (or even radius-6, still 600 collisions)
local neighborhood.

Concatenating normalized global descriptors (molecular weight,
heavy-atom count, logP) recovers most of that lost resolution on paper
(1,200 -> 1,078 unique embeddings) -- but empirically made the real E1
oracle validation dramatically WORSE (chemotype-cold potency Spearman
0.49 -> 0.14; several previously-interpretable splits fell below the
go/no-go threshold). Root cause: TanimotoKernel's k(x,x') =
x.x'/(||x||^2+||x'||^2-x.x') is a Jaccard-style similarity defined for
sparse binary/count vectors; naively concatenating a few dense continuous
values (even bounded to [0,1]) distorts that similarity structure enough
to hurt GP generalization, despite "fixing" the collision count. This was
tested and reverted, not left half-wired -- `use_descriptors` remains
available (off by default) for anyone who wants to build a proper
multi-kernel (e.g., a product of a Tanimoto kernel over the fingerprint
and a separate RBF kernel over the descriptors) rather than flat
concatenation. See leaf/tests/test_embed.py for the collision-count
regression check, and LEAF_Research_Plan.md for the full writeup.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from leaf.fragments.vocabulary import Fragment, assemble_lipid

# Fixed min-max normalization bounds for the descriptor block, measured
# directly over the real, vendored AGILE vocabulary (all 1,200 real
# lipids). TanimotoKernel (oracle/gp_model.py) assumes roughly [0,1]-scale,
# non-negative features -- raw MolWt (~500-850) or LogP (~6-16) concatenated
# directly alongside 0/1 fingerprint bits would dominate the dot product
# and could even drive the kernel's denominator negative. Bounds are
# widened slightly beyond the observed range so a genuinely novel
# candidate (outside the current factorial vocabulary) doesn't clip hard
# at the edges.
_DESCRIPTOR_BOUNDS = {
    "MolWt": (450.0, 950.0),
    "HeavyAtomCount": (30.0, 70.0),
    "MolLogP": (4.0, 18.0),
}


class FragmentEmbedder:
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_descriptors: bool = False):
        self.radius = radius
        self.n_bits = n_bits
        self.use_descriptors = use_descriptors
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def _descriptors(self, mol) -> list[float]:
        raw = {
            "MolWt": Descriptors.MolWt(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "MolLogP": Descriptors.MolLogP(mol),
        }
        return [
            float(np.clip((raw[name] - lo) / (hi - lo), 0.0, 1.0))
            for name, (lo, hi) in _DESCRIPTOR_BOUNDS.items()
        ]

    def embed(self, frag: Fragment) -> np.ndarray:
        smiles = assemble_lipid(frag)
        if smiles is None:
            # Unassemblable fragment tuple (research plan §6.1 gate: this
            # should be 0% of the real AGILE vocabulary; a nonzero rate
            # signals an assembler bug, not something to silently paper
            # over with a zero vector in production use).
            return np.zeros(self.dim, dtype=np.float64)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.dim, dtype=np.float64)
        fp = self._generator.GetFingerprint(mol)
        arr = np.zeros(self.n_bits, dtype=np.float64)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        if self.use_descriptors:
            arr = np.concatenate([arr, np.array(self._descriptors(mol), dtype=np.float64)])
        return arr

    def embed_batch(self, frags: list[Fragment]) -> np.ndarray:
        return np.stack([self.embed(f) for f in frags])

    @property
    def dim(self) -> int:
        return self.n_bits + (3 if self.use_descriptors else 0)
