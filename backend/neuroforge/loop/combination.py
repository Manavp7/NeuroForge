"""Polypharmacology: design a combination regimen addressing multiple abnormal constructs.

For each sufficiently-elevated construct we design + validate a molecule against its target, then
return the best safe candidate per target. The combined response applies each molecule to its own
construct. This complements the main loop (which addresses the dominant construct each iteration).
"""

from __future__ import annotations

import numpy as np

from ..design.generator import make_generator
from ..design.objectives import state_to_targets
from ..models import Candidate, PatientProfile, PatientState, TargetProfile
from ..validation.pipeline import evaluate_molecule


def design_combination(
    state: PatientState,
    seed: int = 7,
    max_targets: int = 2,
    threshold: float = 0.4,
    population: int | None = None,
    generations: int | None = None,
) -> list[dict]:
    """Return a list of {target, candidate} items, one per addressed target."""
    targets = state_to_targets(state, threshold=threshold, max_targets=max_targets)
    items: list[dict] = []
    for i, target in enumerate(targets):
        results = make_generator(seed=seed + i).design(
            target, population=population, generations=generations
        )
        candidates = []
        for r in results:
            c = evaluate_molecule(
                r.smiles, target, seed=seed + i, provenance={"design_score": r.score}
            )
            if c is not None:
                candidates.append(c)
        safe = [c for c in candidates if c.safe]
        chosen = max(safe, key=lambda c: c.score) if safe else None
        items.append({"target": target, "candidate": chosen, "candidates": candidates})
    return items


def simulate_combination_response(
    profile: PatientProfile,
    items: list[dict],
    seed: int = 7,
    session: int = 1,
) -> PatientProfile:
    """Apply every approved candidate to its targeted construct simultaneously."""
    from ..data.synthetic import SyntheticPatientGenerator

    rng = np.random.default_rng(seed + 7 * session)
    latent = dict(profile.latent_state)
    addressed: set[str] = set()

    for item in items:
        candidate: Candidate | None = item.get("candidate")
        target: TargetProfile = item["target"]
        if candidate is None:
            continue
        efficacy = (
            float(candidate.predicted_effect)
            if candidate.predicted_effect > 0
            else 0.5 * float(np.clip((candidate.binding.value - 4.0) / 5.0, 0.0, 1.0))
        )
        for construct in target.driving_constructs:
            latent[construct] = float(max(0.0, latent[construct] * (1.0 - efficacy)))
            addressed.add(construct)

    for c in latent:
        if c not in addressed:
            latent[c] = float(max(0.0, latent[c] + 0.02 * rng.standard_normal()))

    gen = SyntheticPatientGenerator(seed=seed)
    return gen.observe(
        latent, profile.condition, profile.id, rng=rng, eeg_seed=seed + session, session=session
    )
