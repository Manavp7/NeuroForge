"""Human-readable formatting of model attributions.

The numeric attributions are produced inside :class:`~neuroforge.inference.state.StateEstimator`
(standardized feature value × averaged ensemble coefficient). Here we turn them into short,
plain-language factors for the agent rationale and the UI.
"""

from __future__ import annotations

from ..models import PatientState

_FEATURE_LABELS = {
    "proteo_IL6": "IL-6 (inflammatory cytokine)",
    "proteo_TNFa": "TNF-α (inflammatory cytokine)",
    "proteo_CRP": "CRP (acute-phase protein)",
    "proteo_BDNF": "BDNF (neurotrophin)",
    "proteo_dopamine": "dopamine level",
    "proteo_serotonin": "serotonin level",
    "proteo_alpha_synuclein": "α-synuclein",
    "proteo_neurofilament": "neurofilament (neuronal injury)",
    "geno_inflammatory": "inflammatory polygenic risk",
    "geno_dopaminergic": "dopaminergic polygenic risk",
    "geno_serotonergic": "serotonergic polygenic risk",
    "geno_excitability": "cortical-excitability polygenic risk",
    "wear_hrv": "heart-rate variability",
    "wear_rhr": "resting heart rate",
    "wear_sleep": "sleep efficiency",
    "wear_activity": "activity index",
    "eeg_delta": "EEG delta power",
    "eeg_theta": "EEG theta power",
    "eeg_alpha": "EEG alpha power",
    "eeg_beta": "EEG beta power",
    "eeg_gamma": "EEG gamma power",
    "eeg_faa": "frontal alpha asymmetry",
    "eeg_snr": "EEG signal-to-noise",
}


def label(feature: str) -> str:
    return _FEATURE_LABELS.get(feature, feature)


def top_factors(state: PatientState, construct: str, k: int = 3) -> list[str]:
    """Return up to ``k`` plain-language drivers for a construct."""
    factors = state.explanations.get(construct, [])[:k]
    out = []
    for feat, contrib in factors:
        direction = "↑" if contrib >= 0 else "↓"
        out.append(f"{direction} {label(feat)}")
    return out


def summarize_state(state: PatientState, threshold: float = 0.3) -> str:
    """One-line natural-language summary of the most abnormal constructs."""
    items = sorted(state.constructs.items(), key=lambda kv: kv[1].value, reverse=True)
    elevated = [(c, u) for c, u in items if u.value >= threshold]
    if not elevated:
        return f"State near baseline (confidence {state.confidence:.2f})."
    parts = [f"{c.replace('_', ' ')}={u.value:.2f}±{u.std:.2f}" for c, u in elevated]
    return "Elevated: " + ", ".join(parts) + f" (confidence {state.confidence:.2f})."
