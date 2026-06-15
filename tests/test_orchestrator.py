from neuroforge.orchestrator import ClosedLoopOrchestrator
from neuroforge.schemas import ApprovalStatus
from neuroforge.validation import SurrogateValidator, ValidationThresholds


def permissive_orchestrator() -> ClosedLoopOrchestrator:
    return ClosedLoopOrchestrator(
        validator=SurrogateValidator(
            ValidationThresholds(
                min_efficacy=0.0,
                max_toxicity=1.0,
                max_off_target=1.0,
                max_uncertainty=1.0,
            )
        )
    )


def test_run_session_uses_one_synthetic_patient_and_audit_trail() -> None:
    orchestrator = ClosedLoopOrchestrator()

    iterations = orchestrator.run_session(seed=21, steps=3)

    assert len(iterations) == 3
    assert len({iteration.patient.patient_id for iteration in iterations}) == 1
    assert [iteration.step for iteration in iterations] == [0, 1, 2]
    assert all(
        "Synthetic/demo-only run" in iteration.audit_notes[0] for iteration in iterations
    )


def test_validation_pass_alone_is_insufficient_when_approval_required() -> None:
    orchestrator = permissive_orchestrator()

    iteration = orchestrator.run_iteration(
        seed=4, step=0, doctor_approved=False, require_approval=True
    )

    assert iteration.validation.passed is True
    assert iteration.deliverable is False
    assert iteration.approval_status == ApprovalStatus.BLOCKED_PENDING_REVIEW
    assert any("pending review" in note for note in iteration.audit_notes)


def test_doctor_approval_allows_simulated_delivery_when_validation_passes() -> None:
    orchestrator = permissive_orchestrator()

    iteration = orchestrator.run_iteration(
        seed=4, step=0, doctor_approved=True, require_approval=True
    )

    assert iteration.validation.passed is True
    assert iteration.deliverable is True
    assert iteration.approval_status == ApprovalStatus.APPROVED_FOR_SIMULATED_DELIVERY
