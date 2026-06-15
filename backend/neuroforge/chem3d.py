"""3D conformer generation and shape descriptors (RDKit ETKDG + MMFF).

Provides a 3D view of candidates (MolBlock for the viewer) and shape descriptors that can
optionally refine scoring. This is a lightweight stand-in for true 3D structure/docking.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


def embed_3d(smiles: str, seed: int = 7, optimize: bool = True) -> Chem.Mol | None:
    """Parse SMILES, add Hs, embed a 3D conformer (ETKDG), optionally MMFF-optimize."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Retry with random coordinates as a fallback.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    return mol


def mol_to_molblock(smiles: str, seed: int = 7) -> str | None:
    mol = embed_3d(smiles, seed=seed)
    return Chem.MolToMolBlock(mol) if mol is not None else None


def descriptors_3d(mol3d: Chem.Mol) -> dict[str, float]:
    """Shape descriptors from a 3D conformer (normalized principal moments, etc.)."""
    return {
        "npr1": float(rdMolDescriptors.CalcNPR1(mol3d)),
        "npr2": float(rdMolDescriptors.CalcNPR2(mol3d)),
        "radius_of_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol3d)),
        "asphericity": float(rdMolDescriptors.CalcAsphericity(mol3d)),
        "eccentricity": float(rdMolDescriptors.CalcEccentricity(mol3d)),
        "spherocity": float(rdMolDescriptors.CalcSpherocityIndex(mol3d)),
    }


def shape_profile(smiles: str, seed: int = 7) -> dict[str, float] | None:
    """Convenience: embed + compute 3D descriptors for a SMILES."""
    mol3d = embed_3d(smiles, seed=seed)
    return descriptors_3d(mol3d) if mol3d is not None else None
