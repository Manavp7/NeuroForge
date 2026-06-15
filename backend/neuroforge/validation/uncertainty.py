"""Uncertainty utilities and the safety gate."""

from __future__ import annotations

import numpy as np

from ..config import SETTINGS
from ..models import ADMET, Uncertain


def ensemble_stats(values: list[float]) -> Uncertain:
    arr = np.asarray(values, dtype=float)
    return Uncertain(value=float(arr.mean()), std=float(arr.std()))


def safety_gate(admet: ADMET, binding: Uncertain) -> tuple[bool, list[str]]:
    """Return ``(is_safe, notes)``. A candidate must clear every check to be approvable."""
    notes: list[str] = []
    if admet.tox_flags:
        notes.append(f"structural alerts: {', '.join(admet.tox_flags)}")
    if admet.qed < SETTINGS.min_qed:
        notes.append(f"low drug-likeness (QED {admet.qed:.2f} < {SETTINGS.min_qed})")
    if admet.lipinski_violations > SETTINGS.max_lipinski_violations:
        notes.append(
            f"Lipinski violations {admet.lipinski_violations} > {SETTINGS.max_lipinski_violations}"
        )
    if binding.std > SETTINGS.max_binding_std:
        notes.append(
            f"binding uncertainty too high (±{binding.std:.2f} > {SETTINGS.max_binding_std})"
        )
    return (len(notes) == 0, notes)
