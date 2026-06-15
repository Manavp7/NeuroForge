"""ADMET-style property computation + structural-alert flags (illustrative).

Uses RDKit descriptors and the built-in PAINS filter catalog plus a few SMARTS alerts.
These are screening heuristics for a simulation — they are not validated ADMET predictions.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from ..chem import compute_descriptors, sa_proxy
from ..models import ADMET

# Simple structural alerts (toxicophores / reactive groups), as (label, SMARTS).
_ALERTS: list[tuple[str, str]] = [
    ("nitro_group", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("aldehyde", "[CX3H1](=O)"),
    ("michael_acceptor", "[CX3]=[CX3][CX3]=O"),
    ("azide", "[N-]=[N+]=N"),
    ("acyl_halide", "[CX3](=O)[F,Cl,Br,I]"),
    ("thiol", "[#16X2H]"),
]
_ALERT_PATTERNS = [(name, Chem.MolFromSmarts(s)) for name, s in _ALERTS]

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog.FilterCatalog(_pains_params)


def _lipinski_violations(d: dict[str, float]) -> int:
    v = 0
    if d["mol_weight"] > 500:
        v += 1
    if d["logp"] > 5:
        v += 1
    if d["hbd"] > 5:
        v += 1
    if d["hba"] > 10:
        v += 1
    return v


def tox_flags(mol: Chem.Mol) -> list[str]:
    flags: list[str] = []
    if _PAINS.HasMatch(mol):
        flags.append("PAINS")
    for name, patt in _ALERT_PATTERNS:
        if patt is not None and mol.HasSubstructMatch(patt):
            flags.append(name)
    return flags


def compute_admet(mol: Chem.Mol) -> ADMET:
    d = compute_descriptors(mol)
    return ADMET(
        mol_weight=round(d["mol_weight"], 2),
        logp=round(d["logp"], 3),
        tpsa=round(d["tpsa"], 2),
        hbd=int(d["hbd"]),
        hba=int(d["hba"]),
        rotatable_bonds=int(d["rotatable_bonds"]),
        qed=round(d["qed"], 4),
        sa_score=round(sa_proxy(mol), 3),
        lipinski_violations=_lipinski_violations(d),
        tox_flags=tox_flags(mol),
    )
