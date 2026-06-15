"""Dataframe helpers for dashboard and reporting code."""

from __future__ import annotations

import pandas as pd

from neuroforge.schemas import ClosedLoopIteration


def biomarkers_to_frame(iterations: list[ClosedLoopIteration]) -> pd.DataFrame:
    """Convert iteration biomarker snapshots into a tidy dataframe."""

    rows = []
    for iteration in iterations:
        biomarker = iteration.biomarkers
        rows.extend(
            [
                {"step": iteration.step, "metric": "inflammation", "value": biomarker.inflammation},
                {"step": iteration.step, "metric": "stress", "value": biomarker.stress},
                {"step": iteration.step, "metric": "sleep_recovery", "value": biomarker.sleep_recovery},
                {"step": iteration.step, "metric": "hrv", "value": biomarker.hrv},
                {"step": iteration.step, "metric": "glutamate_proxy", "value": biomarker.glutamate_proxy},
                {"step": iteration.step, "metric": "gaba_proxy", "value": biomarker.gaba_proxy},
                {"step": iteration.step, "metric": "serotonin_proxy", "value": biomarker.serotonin_proxy},
            ]
        )
    return pd.DataFrame(rows)


def band_powers_to_frame(iterations: list[ClosedLoopIteration]) -> pd.DataFrame:
    """Convert neural band powers into a tidy dataframe."""

    rows = []
    for iteration in iterations:
        for band, value in iteration.signal_window.band_powers.items():
            rows.append({"step": iteration.step, "band": band, "power": value})
    return pd.DataFrame(rows)


def state_scores_to_frame(iteration: ClosedLoopIteration) -> pd.DataFrame:
    """Convert one inferred state into a dataframe suitable for bar/radar plots."""

    state = iteration.inferred_state
    rows = [
        {"state": "neuroinflammation", "score": state.neuroinflammation},
        {"state": "pain_risk", "score": state.pain_risk},
        {"state": "seizure_risk", "score": state.seizure_risk},
        {"state": "mood_instability", "score": state.mood_instability},
    ]
    return pd.DataFrame(rows)
