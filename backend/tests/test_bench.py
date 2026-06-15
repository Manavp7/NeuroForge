"""Regression guard: the closed loop must keep meeting quality thresholds."""

from neuroforge.bench import format_report, run_benchmark


def test_benchmark_meets_thresholds():
    report = run_benchmark(seed=3, n_per_condition=2, fast=True)
    overall = report["overall"]
    # These are deliberately loose to avoid flakiness while still catching real regressions.
    assert overall["stabilized_rate"] >= 0.6, overall
    assert overall["mean_chosen_score"] >= 0.6, overall
    assert overall["mean_abnormality_reduction"] >= 0.2, overall
    # Healthy controls start near baseline, so any intervention is minimal.
    healthy = next(m for m in report["per_condition"] if m["condition"] == "healthy_control")
    assert healthy["mean_abnormality_reduction"] < 0.15

    assert isinstance(format_report(report), str)
