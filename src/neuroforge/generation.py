"""Controlled toy candidate generation for NeuroForge simulations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from neuroforge.schemas import MoleculeCandidate, PatientProfile, PatientState


@dataclass(frozen=True)
class CandidateTemplate:
    template_id: str
    target_pathway: str
    smiles: str
    rationale_seed: str


TEMPLATE_LIBRARY: dict[str, list[CandidateTemplate]] = {
    "neuroinflammation": [
        CandidateTemplate(
            "nf-glia-001",
            "glial_inflammation_modulation",
            "CC(=O)N1CCOCC1",
            "glial inflammatory tone is elevated",
        ),
        CandidateTemplate(
            "nf-cytokine-002",
            "cytokine_signal_dampening",
            "COc1cc(N)ccc1O",
            "cytokine-linked synthetic markers are elevated",
        ),
    ],
    "pain_risk": [
        CandidateTemplate(
            "nf-nocicept-001",
            "nociceptive_sensitization_buffering",
            "CCN(CC)C(=O)CO",
            "pain-risk proxy combines stress and inflammatory load",
        ),
        CandidateTemplate(
            "nf-prostag-002",
            "inflammatory_pain_proxy_modulation",
            "CC(C)Oc1ccc(O)cc1",
            "inflammatory pain pathway proxy is elevated",
        ),
    ],
    "seizure_risk": [
        CandidateTemplate(
            "nf-excite-001",
            "excitability_balance_modulation",
            "NCC(O)CN1CCOCC1",
            "glutamate/GABA and theta-gamma proxies suggest excitability pressure",
        ),
        CandidateTemplate(
            "nf-ion-002",
            "ion_channel_stability_proxy",
            "CC1CN(CCO)CCN1",
            "baseline ion-channel susceptibility proxy is active",
        ),
    ],
    "mood_instability": [
        CandidateTemplate(
            "nf-stress-001",
            "stress_axis_rebalancing",
            "CCOC(=O)NCCO",
            "stress, sleep, and HRV proxies suggest stress-axis load",
        ),
        CandidateTemplate(
            "nf-affect-002",
            "serotonergic_gabaergic_balance",
            "NCCOc1ccc(O)cc1",
            "serotonin/GABA balance proxies suggest affective instability",
        ),
    ],
}

CONTROLLED_FRAGMENTS = ["O", "N", "CO", "CN"]


class CandidateGenerator:
    """Generate bounded, template-derived toy candidates.

    The generator deliberately avoids unconstrained chemistry generation. All
    candidates originate from a small library and may receive only a controlled
    suffix fragment for descriptor variability.
    """

    def generate(
        self,
        state: PatientState,
        profile: PatientProfile,
        rng: np.random.Generator | None = None,
    ) -> MoleculeCandidate:
        rng = rng or np.random.default_rng()
        templates = TEMPLATE_LIBRARY[state.dominant_state]
        template = templates[int(rng.integers(0, len(templates)))]
        smiles = template.smiles

        mutation_roll = float(rng.random())
        fragment = ""
        if mutation_roll < 0.35:
            fragment = CONTROLLED_FRAGMENTS[int(rng.integers(0, len(CONTROLLED_FRAGMENTS)))]
            smiles = f"{smiles}.{fragment}"

        descriptors = compute_descriptors(smiles)
        candidate_id = self._candidate_id(profile.patient_id, state, template.template_id, smiles)
        rationale = (
            f"Research-only toy candidate selected for {state.dominant_state}: "
            f"{template.rationale_seed}. Dominant score evidence includes "
            f"{_top_evidence(state.feature_contributions)}. Confidence={state.confidence:.2f}; "
            "candidate remains gated by surrogate validation and clinician review."
        )

        if fragment:
            rationale += f" Controlled fragment '{fragment}' added for simulator variability."

        return MoleculeCandidate(
            candidate_id=candidate_id,
            smiles=smiles,
            target_pathway=template.target_pathway,
            template_id=template.template_id,
            rationale=rationale,
            descriptors=descriptors,
            controlled_template=True,
        )

    @staticmethod
    def _candidate_id(
        patient_id: str,
        state: PatientState,
        template_id: str,
        smiles: str,
    ) -> str:
        raw = f"{patient_id}:{state.dominant_state}:{template_id}:{smiles}".encode()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        return f"toy-{digest}"


def compute_descriptors(smiles: str) -> dict[str, float]:
    """Compute lightweight descriptors from a SMILES-like toy string."""

    hetero_atom_count = sum(1 for char in smiles if char in {"N", "O", "S", "P", "F"})
    carbon_count = smiles.count("C") + smiles.count("c")
    ring_like_token_count = sum(1 for char in smiles if char.isdigit())
    branch_count = smiles.count("(") + smiles.count(")")
    polar_token_proxy = hetero_atom_count + smiles.count("=")
    fragment_count = smiles.count(".") + 1
    synthetic_complexity_proxy = (
        0.03 * len(smiles)
        + 0.18 * ring_like_token_count
        + 0.08 * branch_count
        + 0.12 * max(0, fragment_count - 1)
    )

    return {
        "length": float(len(smiles)),
        "hetero_atom_count": float(hetero_atom_count),
        "carbon_count": float(carbon_count),
        "ring_like_token_count": float(ring_like_token_count),
        "branch_count": float(branch_count),
        "polar_token_proxy": float(polar_token_proxy),
        "fragment_count": float(fragment_count),
        "synthetic_complexity_proxy": max(0.0, synthetic_complexity_proxy),
    }


def _top_evidence(feature_contributions: dict[str, float]) -> str:
    positive = sorted(
        ((key, value) for key, value in feature_contributions.items() if value >= 0),
        key=lambda item: item[1],
        reverse=True,
    )
    if not positive:
        return "no positive synthetic evidence"
    return ", ".join(f"{key}={value:.2f}" for key, value in positive[:3])
