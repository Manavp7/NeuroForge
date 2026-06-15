"""Simulate a patient's biological response to an approved therapy.

The approved candidate reduces the targeted latent construct in proportion to its (surrogate)
binding affinity, with noise and mild spontaneous drift in the other constructs. The patient is
then *re-observed* so the next loop iteration decodes a fresh, noisy signal.
"""

from __future__ import annotations

import numpy as np

from ..data.synthetic import SyntheticPatientGenerator
from ..models import Candidate, PatientProfile, TargetProfile


def simulate_response(
    profile: PatientProfile,
    candidate: Candidate,
    target_profile: TargetProfile,
    seed: int = 0,
    session: int = 1,
) -> PatientProfile:
    rng = np.random.default_rng(seed + 991 * session)
    latent = dict(profile.latent_state)

    # Normalized binding (pseudo-pKi 4..9 -> 0..1) drives efficacy, capped per dose.
    norm_binding = float(np.clip((candidate.binding.value - 4.0) / 5.0, 0.0, 1.0))
    efficacy = 0.55 * norm_binding

    for construct in target_profile.driving_constructs:
        before = latent.get(construct, 0.0)
        reduced = before * (1.0 - efficacy) + 0.03 * rng.standard_normal()
        latent[construct] = float(max(0.0, reduced))

    # Mild spontaneous drift elsewhere (regression toward/away from baseline).
    for c in latent:
        if c not in target_profile.driving_constructs:
            latent[c] = float(max(0.0, latent[c] + 0.02 * rng.standard_normal()))

    gen = SyntheticPatientGenerator(seed=seed)
    return gen.observe(
        latent,
        profile.condition,
        profile.id,
        rng=rng,
        eeg_seed=seed + session,
        session=session,
    )
