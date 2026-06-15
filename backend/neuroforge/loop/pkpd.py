"""One-compartment oral PK + Emax PD model (illustrative, research only).

Replaces the previous instantaneous "binding → efficacy" shortcut with a concentration-time
simulation: a dosing regimen is absorbed (first-order ``ka``), distributed into a central
compartment (volume ``vd``), and eliminated (first-order ``ke``). The effect at each time point
follows an Emax model ``E = Emax * C / (EC50 + C)``; the therapy's efficacy is the mean effect at
steady state. More potent molecules (higher predicted pKi) get a lower EC50.

This is a teaching/demonstration model, not validated pharmacology.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PKPDParams:
    ka: float = 1.0  # absorption rate constant (1/h)
    ke: float = 0.15  # elimination rate constant (1/h)
    vd: float = 30.0  # volume of distribution (L)
    f: float = 0.7  # oral bioavailability
    emax: float = 0.7  # maximal effect fraction
    ec50: float = 1.0  # concentration for half-max effect (mg/L)


@dataclass
class Regimen:
    dose_mg: float = 100.0
    n_doses: int = 5
    interval_h: float = 24.0


def simulate_regimen(
    regimen: Regimen, params: PKPDParams, dt: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_h, concentration_mg_per_L) across the whole regimen."""
    state = np.array([0.0, 0.0])  # [gut amount, central amount]
    times: list[np.ndarray] = []
    concs: list[np.ndarray] = []
    t_offset = 0.0

    def rhs(_t, y):
        return [-params.ka * y[0], params.ka * y[0] - params.ke * y[1]]

    for _ in range(regimen.n_doses):
        state[0] += params.f * regimen.dose_mg  # take a dose into the gut
        t_eval = np.arange(0.0, regimen.interval_h, dt)
        sol = solve_ivp(rhs, (0.0, regimen.interval_h), state, t_eval=t_eval, method="RK45")
        times.append(t_eval + t_offset)
        concs.append(sol.y[1] / params.vd)
        state = sol.y[:, -1]
        t_offset += regimen.interval_h

    return np.concatenate(times), np.concatenate(concs)


def effect_curve(conc: np.ndarray, params: PKPDParams) -> np.ndarray:
    return params.emax * conc / (params.ec50 + conc)


def steady_state_efficacy(regimen: Regimen, params: PKPDParams) -> float:
    """Mean effect fraction over the final dosing interval (approx. steady state)."""
    t, c = simulate_regimen(regimen, params)
    if len(t) == 0:
        return 0.0
    last = t >= (t[-1] - regimen.interval_h)
    return float(np.mean(effect_curve(c[last], params)))


def recommend_regimen(binding_pki: float) -> tuple[Regimen, PKPDParams, float, dict]:
    """Map a predicted pKi to a regimen + PK/PD params, returning (regimen, params, efficacy, summary)."""
    norm = float(np.clip((binding_pki - 4.0) / 5.0, 0.0, 1.0))
    # More potent -> lower EC50 (more effect at the same concentration).
    params = PKPDParams(ec50=float(max(0.1, 2.0 * (1.0 - norm) + 0.1)))
    regimen = Regimen(dose_mg=100.0, n_doses=5, interval_h=24.0)
    efficacy = steady_state_efficacy(regimen, params)
    t, c = simulate_regimen(regimen, params)
    summary = {
        "regimen": asdict(regimen),
        "params": asdict(params),
        "cmax_mg_per_L": float(np.max(c)) if len(c) else 0.0,
        "css_avg_mg_per_L": float(np.mean(c[t >= (t[-1] - regimen.interval_h)])) if len(c) else 0.0,
        "predicted_efficacy": efficacy,
    }
    return regimen, params, efficacy, summary
