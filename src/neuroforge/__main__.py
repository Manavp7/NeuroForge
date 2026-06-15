"""Command-line demo for the NeuroForge synthetic loop."""

from __future__ import annotations

import argparse
import json

from neuroforge.orchestrator import ClosedLoopOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic NeuroForge closed-loop demo session."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--doctor-approved", action="store_true")
    parser.add_argument("--no-approval-required", action="store_true")
    args = parser.parse_args()

    iterations = ClosedLoopOrchestrator().run_session(
        seed=args.seed,
        steps=args.steps,
        doctor_approved=args.doctor_approved,
        require_approval=not args.no_approval_required,
    )
    print(
        json.dumps(
            [iteration.model_dump(mode="json") for iteration in iterations],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
