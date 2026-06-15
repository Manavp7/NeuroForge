"""Benchmark / regression harness for the closed loop.

Measures, per condition: stabilization rate, mean iterations to stabilize, mean abnormality
reduction, and mean chosen-candidate score/binding. Used both as a CLI (`neuroforge bench`) and
by a regression test that guards against quality regressions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import CONDITIONS
from .data.synthetic import SyntheticPatientGenerator
from .inference.state import StateEstimator
from .loop.orchestrator import ClosedLoopController


@dataclass
class ConditionMetrics:
    condition: str
    n: int
    stabilized_rate: float
    mean_iterations: float
    mean_abnormality_reduction: float
    mean_chosen_score: float
    mean_chosen_binding: float


def _run_one(controller: ClosedLoopController, gen: SyntheticPatientGenerator, condition: str):
    profile = gen.generate(condition)
    run = controller.run(profile, max_iter=6)
    delivered = [it for it in run.iterations if it.approved and it.abnormality_after is not None]
    first_abn = run.iterations[0].abnormality_before if run.iterations else 0.0
    last_abn = delivered[-1].abnormality_after if delivered else first_abn
    scores = [it.chosen.score for it in run.iterations if it.chosen]
    bindings = [it.chosen.binding.value for it in run.iterations if it.chosen]
    return {
        "stabilized": run.status == "stabilized",
        "iterations": len(run.iterations),
        "reduction": max(0.0, first_abn - last_abn),
        "score": float(np.mean(scores)) if scores else 0.0,
        "binding": float(np.mean(bindings)) if bindings else 0.0,
    }


def run_benchmark(
    seed: int = 3,
    n_per_condition: int = 4,
    estimator: StateEstimator | None = None,
    fast: bool = True,
) -> dict:
    est = estimator or StateEstimator(seed=seed, n_train=160, ensemble=8)
    kwargs = dict(ga_population=18, ga_generations=5, ga_top_k=4) if fast else {}
    controller = ClosedLoopController(seed=seed, estimator=est, **kwargs)

    per_condition: list[ConditionMetrics] = []
    for condition in CONDITIONS:
        gen = SyntheticPatientGenerator(seed=seed + hash(condition) % 1000)
        results = [_run_one(controller, gen, condition) for _ in range(n_per_condition)]
        per_condition.append(
            ConditionMetrics(
                condition=condition,
                n=n_per_condition,
                stabilized_rate=float(np.mean([r["stabilized"] for r in results])),
                mean_iterations=float(np.mean([r["iterations"] for r in results])),
                mean_abnormality_reduction=float(np.mean([r["reduction"] for r in results])),
                mean_chosen_score=float(np.mean([r["score"] for r in results])),
                mean_chosen_binding=float(np.mean([r["binding"] for r in results])),
            )
        )

    # "needs treatment" excludes healthy controls (which start near baseline).
    treat = [m for m in per_condition if m.condition != "healthy_control"]
    overall = {
        "stabilized_rate": float(np.mean([m.stabilized_rate for m in treat])),
        "mean_chosen_score": float(np.mean([m.mean_chosen_score for m in treat])),
        "mean_abnormality_reduction": float(np.mean([m.mean_abnormality_reduction for m in treat])),
    }
    return {"overall": overall, "per_condition": [asdict(m) for m in per_condition]}


def format_report(report: dict) -> str:
    lines = ["NeuroForge benchmark (research/simulation only)", ""]
    lines.append(f"{'condition':<18}{'stab%':>7}{'iters':>7}{'Δabn':>8}{'score':>7}{'pKi':>7}")
    for m in report["per_condition"]:
        lines.append(
            f"{m['condition']:<18}{m['stabilized_rate'] * 100:>6.0f}%"
            f"{m['mean_iterations']:>7.1f}{m['mean_abnormality_reduction']:>8.2f}"
            f"{m['mean_chosen_score']:>7.2f}{m['mean_chosen_binding']:>7.1f}"
        )
    o = report["overall"]
    lines += [
        "",
        f"OVERALL (excl. healthy): stabilized {o['stabilized_rate'] * 100:.0f}% · "
        f"mean score {o['mean_chosen_score']:.2f} · mean Δabnormality {o['mean_abnormality_reduction']:.2f}",
    ]
    return "\n".join(lines)
