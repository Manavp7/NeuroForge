"""Cohort / population mode: run several patients through the loop and summarize outcomes."""

from __future__ import annotations

import numpy as np

from .data.synthetic import SyntheticPatientGenerator
from .loop.orchestrator import ClosedLoopController


def run_cohort(
    condition: str,
    n: int = 6,
    seed: int = 3,
    controller: ClosedLoopController | None = None,
    max_iter: int = 6,
) -> dict:
    """Run ``n`` patients of ``condition`` through the auto loop; return per-patient + summary."""
    ctrl = controller or ClosedLoopController(
        seed=seed, ga_population=18, ga_generations=5, ga_top_k=4
    )
    gen = SyntheticPatientGenerator(seed=seed)
    patients = []
    for _ in range(n):
        profile = gen.generate(condition)
        run = ctrl.run(profile, max_iter=max_iter)
        first = run.iterations[0].abnormality_before if run.iterations else 0.0
        delivered = [
            it for it in run.iterations if it.approved and it.abnormality_after is not None
        ]
        last = delivered[-1].abnormality_after if delivered else first
        patients.append(
            {
                "patient_id": profile.id,
                "initial_abnormality": round(first, 4),
                "final_abnormality": round(last, 4),
                "reduction": round(max(0.0, first - last), 4),
                "stabilized": run.status == "stabilized",
                "iterations": len(run.iterations),
            }
        )

    reductions = [p["reduction"] for p in patients]
    summary = {
        "condition": condition,
        "n": n,
        "stabilized_rate": round(float(np.mean([p["stabilized"] for p in patients])), 3),
        "mean_reduction": round(float(np.mean(reductions)), 4),
        "std_reduction": round(float(np.std(reductions)), 4),
        "mean_iterations": round(float(np.mean([p["iterations"] for p in patients])), 2),
    }
    return {"summary": summary, "patients": patients}
