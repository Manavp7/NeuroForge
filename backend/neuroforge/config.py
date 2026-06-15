"""Global configuration and deterministic defaults for NeuroForge.

Everything in NeuroForge is seedable so runs are reproducible and tests are stable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: EEG canonical frequency bands (Hz) used by the simulator and feature extractor.
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

#: Patient-state constructs the system tracks (each as value + uncertainty).
STATE_CONSTRUCTS: tuple[str, ...] = (
    "neuroinflammation",
    "dopaminergic_deficit",
    "serotonergic_deficit",
    "pain_index",
    "mood_index",
    "seizure_risk",
)

#: Synthetic conditions the patient generator can produce.
CONDITIONS: tuple[str, ...] = (
    "neuroinflammatory",
    "parkinsonian",
    "mood_disorder",
    "epileptiform",
    "healthy_control",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings (mostly thresholds + seeds)."""

    default_seed: int = 7

    # EEG simulation
    eeg_fs: float = 128.0
    eeg_duration_s: float = 8.0

    # Inference uncertainty
    inference_ensemble: int = 16
    inference_noise: float = 0.04

    # Generative design
    ga_population: int = 40
    ga_generations: int = 12
    ga_top_k: int = 5
    ga_mutation_rate: float = 0.35
    generator_engine: str = field(
        default_factory=lambda: os.getenv("NEUROFORGE_GENERATOR", "ga")
    )  # ga | vae

    # Validation / safety gate
    binding_ensemble: int = 12
    max_binding_std: float = 1.5  # reject candidates whose binding UQ is too wide
    binding_model: str = field(
        default_factory=lambda: os.getenv("NEUROFORGE_BINDING_MODEL", "heuristic")
    )  # heuristic | mlp | torch
    min_qed: float = 0.30
    max_lipinski_violations: int = 1

    # Agentic redesign
    agentic_redesign: bool = True
    redesign_threshold: float = 0.55  # if best safe score below this, tighten + redesign

    # Uncertainty-aware decisions
    uncertainty_aware: bool = True
    risk_aversion: float = field(
        default_factory=lambda: float(os.getenv("NEUROFORGE_RISK_AVERSION", "0.4"))
    )

    # Closed loop
    max_iterations: int = 6
    state_target_threshold: float = 0.35  # max-construct abnormality below this => "stabilized"

    # Agent
    openai_model: str = field(
        default_factory=lambda: os.getenv("NEUROFORGE_OPENAI_MODEL", "gpt-4o-mini")
    )


SETTINGS = Settings()


def get_seed(seed: int | None = None) -> int:
    """Return an explicit seed or fall back to the deterministic default."""
    return SETTINGS.default_seed if seed is None else seed
