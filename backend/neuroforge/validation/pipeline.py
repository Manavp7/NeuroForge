"""Assemble a fully-validated :class:`Candidate` from a SMILES + target profile."""

from __future__ import annotations

import hashlib

from rdkit import Chem

from ..loop.pkpd import recommend_regimen
from ..models import Candidate, TargetProfile, Uncertain
from .admet import compute_admet
from .binding import make_predictor
from .uncertainty import safety_gate


def _candidate_id(smiles: str) -> str:
    return "cand-" + hashlib.sha1(smiles.encode()).hexdigest()[:10]


def composite_score(binding: Uncertain, qed: float) -> float:
    norm_binding = max(0.0, (binding.value - 4.0) / 5.0)  # map pseudo-pKi [4,9] -> [0,1]
    return float(0.6 * norm_binding + 0.4 * qed)


def risk_adjusted(score: float, binding: Uncertain, risk_aversion: float | None = None) -> float:
    """Penalize a candidate's score by its (normalized) binding uncertainty."""
    from ..config import SETTINGS

    lam = SETTINGS.risk_aversion if risk_aversion is None else risk_aversion
    norm_std = min(1.0, binding.std / max(SETTINGS.max_binding_std, 1e-6))
    return float(score * (1.0 - lam * norm_std))


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
    binding = make_predictor(target_profile.target_id, seed=seed).predict(mol)
    safe, notes = safety_gate(admet, binding)
    _, _, efficacy, pkpd_summary = recommend_regimen(binding.value)
    score = composite_score(binding, admet.qed)
    return Candidate(
        id=_candidate_id(smiles),
        smiles=smiles,
        admet=admet,
        binding=binding,
        score=round(score, 4),
        risk_adjusted_score=round(risk_adjusted(score, binding), 4),
        safe=safe,
        safety_notes=notes,
        predicted_effect=round(efficacy, 4),
        pkpd=pkpd_summary,
        provenance=provenance or {},
    )
