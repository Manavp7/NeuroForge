"""Translate inferred patient state into a molecular design objective, and score molecules.

This is intentionally rule-based and transparent: the mapping from the dominant abnormal
construct to a (mock) protein target is fully explainable.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ..chem import compute_descriptors, pharmacophore_vector
from ..models import PatientState, TargetProfile
from .library import TARGETS, Target

# Which (mock) target addresses each state construct.
_CONSTRUCT_TO_TARGET: dict[str, str] = {
    "neuroinflammation": "TNF_alpha",
    "dopaminergic_deficit": "D2",
    "serotonergic_deficit": "SERT",
    "pain_index": "Nav1_7",
    "mood_index": "SERT",
    "seizure_risk": "GABA_A",
}


def state_to_target(state: PatientState) -> TargetProfile:
    """Pick the target addressing the most abnormal construct."""
    if not state.constructs:
        raise ValueError("Empty patient state")
    dominant = max(state.constructs.items(), key=lambda kv: kv[1].value)
    construct, unc = dominant
    target_id = _CONSTRUCT_TO_TARGET[construct]
    target = TARGETS[target_id]
    rationale = (
        f"Dominant abnormality is {construct.replace('_', ' ')} "
        f"({unc.value:.2f}±{unc.std:.2f}); selecting {target.name} as the intervention target."
    )
    return TargetProfile(
        target_id=target_id,
        target_name=target.name,
        rationale=rationale,
        property_windows=target.property_windows,
        driving_constructs={construct: unc.value},
    )


def pharmacophore_similarity(mol: Chem.Mol, target: Target) -> float:
    """Similarity in [0, 1]: 1 = perfect match to the target's ideal pharmacophore."""
    vec = pharmacophore_vector(mol)
    dist = float(np.linalg.norm(vec - target.ideal_pharmacophore))
    return float(np.exp(-dist))


def _window_match(value: float, low: float, high: float) -> float:
    """1.0 inside the window, decaying smoothly outside it."""
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    dist = (low - value) if value < low else (value - high)
    return float(np.exp(-(dist / span) ** 2))


def property_match(descriptors: dict[str, float], target: Target) -> float:
    if not target.property_windows:
        return 1.0
    scores = [
        _window_match(descriptors.get(name, 0.0), lo, hi)
        for name, (lo, hi) in target.property_windows.items()
    ]
    return float(np.mean(scores))


def design_score(mol: Chem.Mol, target_profile: TargetProfile) -> float:
    """Composite design objective in ~[0, 1] used by the GA fitness function."""
    target = TARGETS[target_profile.target_id]
    desc = compute_descriptors(mol)
    sim = pharmacophore_similarity(mol, target)
    pm = property_match(desc, target)
    qed = desc["qed"]
    return float(0.5 * sim + 0.3 * pm + 0.2 * qed)
