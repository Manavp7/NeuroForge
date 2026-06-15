"""Shared RDKit chemistry helpers used by the design and validation layers.

Kept dependency-light: no RDKit Contrib modules, so it works from a plain ``pip install rdkit``.
The synthetic-accessibility value is a transparent heuristic proxy (clearly *not* the
Ertl/Schuffenhauer SA score) — adequate for a simulation demo.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse and sanitize SMILES; return ``None`` if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol


def canonical_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def compute_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Return physicochemical descriptors used across design + validation."""
    return {
        "mol_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "qed": float(QED.qed(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def sa_proxy(mol: Chem.Mol) -> float:
    """Heuristic synthetic-accessibility proxy in ~[1, 10] (lower = easier)."""
    n_atoms = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    macro = sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) > 8)
    score = 1.0 + 0.03 * n_atoms + 0.4 * n_rings + 0.6 * n_stereo + 0.8 * spiro + 1.5 * macro
    return float(min(10.0, score))


# Normalization constants for the pharmacophore vector (rough drug-like scales).
_PHARM_SCALE = np.array([500.0, 5.0, 140.0, 5.0, 10.0, 4.0, 1.0, 10.0])


def pharmacophore_vector(mol: Chem.Mol) -> np.ndarray:
    """A fixed-length normalized descriptor vector for target matching."""
    d = compute_descriptors(mol)
    raw = np.array(
        [
            d["mol_weight"],
            d["logp"],
            d["tpsa"],
            d["hbd"],
            d["hba"],
            d["aromatic_rings"],
            d["fraction_csp3"],
            d["rotatable_bonds"],
        ]
    )
    return raw / _PHARM_SCALE


PHARM_DIM = len(_PHARM_SCALE)
