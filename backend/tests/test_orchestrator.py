import pytest

from neuroforge.agent.llm import MockLLM
from neuroforge.data.synthetic import SyntheticPatientGenerator
from neuroforge.design.objectives import state_to_target
from neuroforge.inference.state import StateEstimator
from neuroforge.loop.orchestrator import ClosedLoopController
from neuroforge.loop.response import simulate_response


@pytest.fixture(scope="module")
def controller():
    est = StateEstimator(seed=3, n_train=150, ensemble=8)
    return ClosedLoopController(
        seed=3, estimator=est, llm=MockLLM(), ga_population=16, ga_generations=4, ga_top_k=4
    )


def test_full_loop_improves_state(controller):
    profile = SyntheticPatientGenerator(seed=3).generate("parkinsonian")
    run = controller.run(profile, max_iter=6)
    assert run.events
    delivered = [it for it in run.iterations if it.approved and it.abnormality_after is not None]
    assert delivered, "expected at least one delivered therapy"
    # Net improvement across the run.
    assert delivered[-1].abnormality_after < run.iterations[0].abnormality_before
    assert run.status in {"stabilized", "exhausted"}


def test_rejection_path_no_delivery(controller):
    profile = SyntheticPatientGenerator(seed=4).generate("mood_disorder")
    run = controller.run(profile, max_iter=2, approval_fn=lambda it: False)
    for it in run.iterations:
        assert it.approved is False
        assert it.abnormality_after is None
    assert any(ev.phase == "gate" and ev.payload.get("approved") is False for ev in run.events)


def test_events_have_expected_phases(controller):
    profile = SyntheticPatientGenerator(seed=5).generate("neuroinflammatory")
    run = controller.run(profile, max_iter=3)
    phases = {ev.phase for ev in run.events}
    assert {"sense", "infer", "plan", "design", "validate"} <= phases


def test_response_reduces_targeted_construct(controller):
    profile = SyntheticPatientGenerator(seed=6).generate("neuroinflammatory")
    state = controller.infer(profile)
    iteration, _, _ = controller.build_iteration(profile, state, 0)
    assert iteration.chosen is not None
    target = iteration.target
    construct = next(iter(target.driving_constructs))
    before = profile.latent_state[construct]
    new_profile = simulate_response(profile, iteration.chosen, target, seed=3, session=1)
    assert new_profile.latent_state[construct] < before


def test_mock_llm_outputs_text(controller):
    profile = SyntheticPatientGenerator(seed=7).generate("epileptiform")
    state = controller.infer(profile)
    target = state_to_target(state)
    assert isinstance(MockLLM().plan(state, target), str)
