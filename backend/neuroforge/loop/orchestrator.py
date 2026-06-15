"""Closed-loop controller: sense → infer → plan → design → validate → critique → approve →
deliver → monitor, iterating until the patient state stabilizes or the budget is exhausted.

Exposes both a headless :meth:`run` (used by the CLI/tests, with an approval callback) and the
fine-grained building blocks (:meth:`build_iteration`, :meth:`apply_decision`) that the API uses
to pause at the doctor-in-the-loop approval gate.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from ..agent.llm import LLMClient, get_llm
from ..config import SETTINGS
from ..design.generator import make_generator
from ..design.objectives import state_to_target
from ..inference.state import StateEstimator, get_default_estimator
from ..models import Iteration, LoopEvent, LoopRun, PatientProfile, PatientState
from ..validation.pipeline import evaluate_molecule
from .response import simulate_response

EmitFn = Callable[[LoopEvent], None]
ApprovalFn = Callable[[Iteration], bool]


class ClosedLoopController:
    def __init__(
        self,
        seed: int | None = None,
        estimator: StateEstimator | None = None,
        llm: LLMClient | None = None,
        ga_population: int | None = None,
        ga_generations: int | None = None,
        ga_top_k: int | None = None,
        generator_engine: str | None = None,
        agentic_redesign: bool | None = None,
        redesign_threshold: float | None = None,
    ):
        self.seed = SETTINGS.default_seed if seed is None else seed
        self.estimator = estimator or get_default_estimator()
        self.llm = llm or get_llm()
        self.ga_population = ga_population
        self.ga_generations = ga_generations
        self.ga_top_k = ga_top_k
        self.generator_engine = generator_engine or SETTINGS.generator_engine
        self.agentic_redesign = (
            SETTINGS.agentic_redesign if agentic_redesign is None else agentic_redesign
        )
        self.redesign_threshold = (
            SETTINGS.redesign_threshold if redesign_threshold is None else redesign_threshold
        )

    # ------------------------------------------------------------------ #
    def infer(self, profile: PatientProfile) -> PatientState:
        return self.estimator.estimate(profile)

    def build_iteration(
        self, profile: PatientProfile, state: PatientState, index: int
    ) -> tuple[Iteration, str, str]:
        """Run plan→design→validate→critique. Returns (iteration, plan_text, critique_text)."""
        target = state_to_target(state)
        plan_text = self.llm.plan(state, target)

        generator = make_generator(seed=self.seed + index, engine=self.generator_engine)

        def _design(constraints=None):
            results = generator.design(
                target,
                population=self.ga_population,
                generations=self.ga_generations,
                top_k=self.ga_top_k,
                constraints=constraints,
            )
            out = []
            for r in results:
                cand = evaluate_molecule(
                    r.smiles, target, seed=self.seed + index, provenance={"design_score": r.score}
                )
                if cand is not None:
                    cand.rationale = self.llm.rationale(cand, target)
                    out.append(cand)
            return out

        candidates = _design()

        # Agentic critique -> redesign: if the round is weak, tighten constraints and retry.
        redesigned = False
        if self.agentic_redesign:
            safe0 = [c for c in candidates if c.safe]
            best0 = max((c.score for c in safe0), default=0.0)
            if not safe0 or best0 < self.redesign_threshold:
                from ..design.constraints import derive_constraints, passes_constraints

                constraints = derive_constraints(candidates, target)
                extra = _design(constraints=constraints)
                candidates = candidates + [c for c in extra if passes_constraints(c, constraints)]
                candidates = list({c.id: c for c in candidates}.values())
                redesigned = True

        # Selection key: uncertainty-aware (risk-adjusted, scaled up when state confidence is low).
        if SETTINGS.uncertainty_aware:
            from ..validation.pipeline import risk_adjusted

            eff_lambda = SETTINGS.risk_aversion * (1.5 - 0.5 * float(state.confidence))

            def _key(c):
                return risk_adjusted(c.score, c.binding, risk_aversion=eff_lambda)

        else:

            def _key(c):
                return c.score

        candidates.sort(key=lambda c: (c.safe, _key(c)), reverse=True)
        safe = [c for c in candidates if c.safe]
        chosen = max(safe, key=_key) if safe else None
        critique_text = self.llm.critique(candidates, target)
        if redesigned:
            critique_text = "[redesign triggered] " + critique_text

        iteration = Iteration(
            index=index,
            state=state,
            target=target,
            candidates=candidates,
            chosen=chosen,
            abnormality_before=state.abnormality(),
        )
        return iteration, plan_text, critique_text

    def apply_decision(
        self, profile: PatientProfile, iteration: Iteration, approved: bool
    ) -> tuple[PatientProfile, float | None]:
        """Apply (or skip) the therapy; returns (next_profile, abnormality_after)."""
        iteration.approved = approved
        if approved and iteration.chosen is not None:
            next_profile = simulate_response(
                profile,
                iteration.chosen,
                iteration.target,
                seed=self.seed,
                session=iteration.index + 1,
            )
            iteration.abnormality_after = self.infer(next_profile).abnormality()
            return next_profile, iteration.abnormality_after
        return profile, None

    # ------------------------------------------------------------------ #
    def run(
        self,
        profile: PatientProfile,
        max_iter: int | None = None,
        approval_fn: ApprovalFn | None = None,
        emit: EmitFn | None = None,
    ) -> LoopRun:
        max_iter = max_iter or SETTINGS.max_iterations
        approval_fn = approval_fn or (lambda it: it.chosen is not None and it.chosen.safe)
        run = LoopRun(id=str(uuid.uuid4()), patient_id=profile.id, status="running")

        def _emit(iteration: int, phase: str, message: str, **payload) -> None:
            ev = LoopEvent(iteration=iteration, phase=phase, message=message, payload=payload)
            run.events.append(ev)
            if emit:
                emit(ev)

        current = profile
        for i in range(max_iter):
            state = self.infer(current)
            abn = state.abnormality()
            _emit(i, "sense", f"Sensed multimodal data; signal confidence {state.confidence:.2f}.")
            _emit(i, "infer", f"Inferred state; abnormality index {abn:.3f}.", abnormality=abn)

            if abn < SETTINGS.state_target_threshold:
                run.status = "stabilized"
                _emit(i, "done", f"Patient state stabilized (abnormality {abn:.3f}).")
                break

            iteration, plan_text, critique_text = self.build_iteration(current, state, i)
            _emit(i, "plan", plan_text, target=iteration.target.target_id)
            _emit(
                i,
                "design",
                f"Generated {len(iteration.candidates)} candidate(s) for {iteration.target.target_name}.",
            )
            _emit(
                i,
                "validate",
                f"{sum(c.safe for c in iteration.candidates)} candidate(s) cleared the safety gate.",
            )
            _emit(i, "critique", critique_text)

            if iteration.chosen is None:
                iteration.approved = False
                run.iterations.append(iteration)
                _emit(i, "gate", "No safe candidate available; halting for review.")
                run.status = "rejected"
                break

            approved = approval_fn(iteration)
            _emit(
                i,
                "gate",
                f"Doctor-in-the-loop {'APPROVED' if approved else 'REJECTED'} candidate "
                f"{iteration.chosen.id}.",
                candidate_id=iteration.chosen.id,
                approved=approved,
            )

            current, abn_after = self.apply_decision(current, iteration, approved)
            run.iterations.append(iteration)
            if approved and abn_after is not None:
                _emit(
                    i,
                    "deliver",
                    f"Therapy delivered (sim); abnormality {iteration.abnormality_before:.3f} "
                    f"→ {abn_after:.3f}.",
                    abnormality_after=abn_after,
                )
            else:
                _emit(i, "monitor", "Therapy not delivered; continuing observation.")

        else:
            final_abn = self.infer(current).abnormality()
            run.status = (
                "stabilized" if final_abn < SETTINGS.state_target_threshold else "exhausted"
            )

        return run
