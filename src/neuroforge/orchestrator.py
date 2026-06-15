"""Closed-loop orchestration across NeuroForge simulator components."""

from __future__ import annotations

from neuroforge.generation import CandidateGenerator
from neuroforge.inference import StateInferenceEngine
from neuroforge.schemas import (
    ApprovalStatus,
    ClosedLoopIteration,
    PatientProfile,
    SAFETY_DISCLAIMER,
)
from neuroforge.synthetic import SyntheticPatientGenerator
from neuroforge.validation import SurrogateValidator


class ClosedLoopOrchestrator:
    """Run the synthetic neural-biomarker-to-candidate loop."""

    def __init__(
        self,
        synthetic_generator: SyntheticPatientGenerator | None = None,
        inference_engine: StateInferenceEngine | None = None,
        candidate_generator: CandidateGenerator | None = None,
        validator: SurrogateValidator | None = None,
    ) -> None:
        self.synthetic_generator = synthetic_generator or SyntheticPatientGenerator()
        self.inference_engine = inference_engine or StateInferenceEngine()
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.validator = validator or SurrogateValidator()

    def run_iteration(
        self,
        seed: int = 7,
        step: int = 0,
        doctor_approved: bool = False,
        require_approval: bool = True,
        patient: PatientProfile | None = None,
    ) -> ClosedLoopIteration:
        """Run one complete synthetic loop iteration."""

        profile = patient or self.synthetic_generator.generate_profile(seed)
        rng = self.synthetic_generator.rng(seed, step)
        biomarkers = self.synthetic_generator.generate_biomarkers(profile, step=step, rng=rng)
        signal_window = self.synthetic_generator.generate_neural_window(
            profile, biomarkers, step=step, rng=rng
        )
        inferred_state = self.inference_engine.infer(profile, biomarkers, signal_window)
        candidate = self.candidate_generator.generate(inferred_state, profile, rng=rng)
        validation = self.validator.validate(candidate, inferred_state, profile)

        effective_approval = doctor_approved or not require_approval
        if not validation.passed:
            approval_status = ApprovalStatus.BLOCKED_VALIDATION_FAILED
            deliverable = False
        elif effective_approval:
            approval_status = ApprovalStatus.APPROVED_FOR_SIMULATED_DELIVERY
            deliverable = True
        else:
            approval_status = ApprovalStatus.BLOCKED_PENDING_REVIEW
            deliverable = False

        audit_notes = self._audit_notes(
            inferred_state=inferred_state.dominant_state,
            validation_warnings=validation.warnings,
            require_approval=require_approval,
            doctor_approved=doctor_approved,
            deliverable=deliverable,
        )

        return ClosedLoopIteration(
            step=step,
            patient=profile,
            biomarkers=biomarkers,
            signal_window=signal_window,
            inferred_state=inferred_state,
            candidate=candidate,
            validation=validation,
            approval_required=require_approval,
            doctor_approved=doctor_approved,
            approval_status=approval_status,
            deliverable=deliverable,
            audit_notes=audit_notes,
            safety_disclaimer=SAFETY_DISCLAIMER,
        )

    def run_session(
        self,
        seed: int = 7,
        steps: int = 3,
        doctor_approved: bool = False,
        require_approval: bool = True,
    ) -> list[ClosedLoopIteration]:
        """Run a deterministic multi-step synthetic session for one profile."""

        profile = self.synthetic_generator.generate_profile(seed)
        return [
            self.run_iteration(
                seed=seed,
                step=step,
                doctor_approved=doctor_approved,
                require_approval=require_approval,
                patient=profile,
            )
            for step in range(steps)
        ]

    @staticmethod
    def _audit_notes(
        inferred_state: str,
        validation_warnings: list[str],
        require_approval: bool,
        doctor_approved: bool,
        deliverable: bool,
    ) -> list[str]:
        notes = [
            "Synthetic/demo-only run: no real patient data or validated therapy was used.",
            f"Dominant inferred state driver: {inferred_state}.",
            "Validation is a transparent toy surrogate, not a physics or clinical assay.",
        ]
        notes.extend(f"Validation note: {warning}" for warning in validation_warnings)

        if require_approval and not doctor_approved:
            notes.append("Doctor-in-the-loop gate blocked simulated delivery pending review.")
        elif require_approval and doctor_approved and deliverable:
            notes.append("Doctor approval flag enabled; candidate passed simulated delivery gate.")
        elif not require_approval and deliverable:
            notes.append("Approval requirement disabled for sandbox-only semi-autonomous mode.")
        else:
            notes.append("Candidate remains blocked by validation or approval gate.")

        return notes
