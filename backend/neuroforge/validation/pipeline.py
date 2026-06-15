"""Assemble a fully-validated :class:`Candidate` from a SMILES + target profile."""

from __future__ import annotations

import hashlib

from rdkit import Chem

from ..loop.pkpd import recommend_regimen
from ..models import Candidate, TargetProfile, Uncertain
from .admet import compute_admet
from .binding import BindingPredictor
from .uncertainty import safety_gate


def _candidate_id(smiles: str) -> str:
    return "cand-" + hashlib.sha1(smiles.encode()).hexdigest()[:10]


def composite_score(binding: Uncertain, qed: float) -> float:
    norm_binding = max(0.0, (binding.value - 4.0) / 5.0)  # map pseudo-pKi [4,9] -> [0,1]
    return float(0.6 * norm_binding + 0.4 * qed)


def evaluate_molecule(
    smiles: str,
    target_profile: TargetProfile,
    seed: int | None = None,
    provenance: dict | None = None,
) -> Candidate | None:
    """Run ADMET + surrogate binding + safety gate. Returns ``None`` for invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    admet = compute_admet(mol)
    binding = BindingPredictor(target_profile.target_id, seed=seed).predict(mol)
    safe, notes = safety_gate(admet, binding)
    _, _, efficacy, pkpd_summary = recommend_regimen(binding.value)
    return Candidate(
        id=_candidate_id(smiles),
        smiles=smiles,
        admet=admet,
        binding=binding,
        score=round(composite_score(binding, admet.qed), 4),
        safe=safe,
        safety_notes=notes,
        predicted_effect=round(efficacy, 4),
        pkpd=pkpd_summary,
        provenance=provenance or {},
    )
