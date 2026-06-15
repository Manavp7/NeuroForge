from neuroforge.data.synthetic import SyntheticPatientGenerator
from neuroforge.design.library import TARGETS
from neuroforge.design.objectives import state_to_targets
from neuroforge.loop.combination import design_combination, simulate_combination_response
from neuroforge.models import PatientState, Uncertain


def test_new_targets_present():
    for tid in ["COX2", "AChE", "NMDA"]:
        assert tid in TARGETS
        assert TARGETS[tid].ideal_pharmacophore.shape[0] == 8


def test_state_to_targets_multiple():
    state = PatientState(
        constructs={
            "neuroinflammation": Uncertain(value=0.9, std=0.1),
            "pain_index": Uncertain(value=0.7, std=0.1),
            "mood_index": Uncertain(value=0.1, std=0.1),
        }
    )
    targets = state_to_targets(state, threshold=0.4, max_targets=3)
    assert len(targets) >= 2
    ids = {t.target_id for t in targets}
    assert "TNF_alpha" in ids  # neuroinflammation primary


def test_design_and_apply_combination_reduces_constructs():
    profile = SyntheticPatientGenerator(seed=3).generate("neuroinflammatory")
    # Build a state with two elevated constructs that map to distinct targets.
    state = PatientState(
        constructs={
            "neuroinflammation": Uncertain(value=0.9, std=0.1),
            "pain_index": Uncertain(value=0.7, std=0.1),
        }
    )
    items = design_combination(state, seed=3, max_targets=2, population=14, generations=3)
    assert len(items) == 2
    assert any(it["candidate"] is not None for it in items)

    before = dict(profile.latent_state)
    new_profile = simulate_combination_response(profile, items, seed=3, session=1)
    # At least one addressed construct should drop.
    drops = [
        new_profile.latent_state[c] < before[c]
        for it in items
        if it["candidate"]
        for c in it["target"].driving_constructs
    ]
    assert any(drops)
