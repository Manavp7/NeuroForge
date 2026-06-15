"""Headless demo of the NeuroForge closed loop.

Examples
--------
    python -m neuroforge.cli demo --condition neuroinflammatory --iters 5
    python -m neuroforge.cli demo --json
"""

from __future__ import annotations

import argparse
import json
import sys

from . import DISCLAIMER, __version__
from .config import CONDITIONS, SETTINGS
from .data.synthetic import SyntheticPatientGenerator
from .loop.orchestrator import ClosedLoopController
from .models import LoopEvent


def _run_demo(args: argparse.Namespace) -> int:
    gen = SyntheticPatientGenerator(seed=args.seed)
    profile = gen.generate(args.condition)
    controller = ClosedLoopController(seed=args.seed)

    events: list[LoopEvent] = []

    def emit(ev: LoopEvent) -> None:
        events.append(ev)
        if not args.json:
            print(f"  [iter {ev.iteration}] {ev.phase:<9} {ev.message}")

    if not args.json:
        print(f"NeuroForge v{__version__} — closed-loop demo")
        print(f"!! {DISCLAIMER}")
        print(f"\nPatient {profile.id} (condition: {profile.condition})\n")

    run = controller.run(profile, max_iter=args.iters, emit=emit)

    if args.json:
        json.dump(
            {"run": json.loads(run.model_dump_json()), "disclaimer": DISCLAIMER},
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print(f"\nFinal status: {run.status}")
    delivered = [it for it in run.iterations if it.approved and it.abnormality_after is not None]
    if delivered:
        first = run.iterations[0].abnormality_before
        last = delivered[-1].abnormality_after
        print(
            f"Abnormality: {first:.3f} → {last:.3f} over {len(delivered)} delivered therapy step(s)."
        )
        chosen = delivered[-1].chosen
        if chosen:
            print(f"Last molecule: {chosen.smiles}")
            print(
                f"  predicted pKi {chosen.binding.value:.2f}±{chosen.binding.std:.2f}, "
                f"QED {chosen.admet.qed:.2f}, score {chosen.score:.2f}"
            )
    return 0


def _run_bench(args: argparse.Namespace) -> int:
    from .bench import format_report, run_benchmark

    report = run_benchmark(seed=args.seed, n_per_condition=args.n, fast=True)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print(format_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroforge", description="NeuroForge closed-loop simulator (research only)."
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="run the headless closed loop")
    d.add_argument("--condition", choices=CONDITIONS, default="neuroinflammatory")
    d.add_argument("--iters", type=int, default=SETTINGS.max_iterations)
    d.add_argument("--seed", type=int, default=SETTINGS.default_seed)
    d.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    d.set_defaults(func=_run_demo)

    b = sub.add_parser("bench", help="run the benchmark / regression harness")
    b.add_argument("--seed", type=int, default=SETTINGS.default_seed)
    b.add_argument("--n", type=int, default=4, help="patients per condition")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=_run_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
