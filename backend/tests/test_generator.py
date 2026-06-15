from rdkit import Chem

from neuroforge.design.generator import MoleculeGenerator
from neuroforge.design.library import TARGETS
from neuroforge.design.objectives import design_score, state_to_target
from neuroforge.models import PatientState, Uncertain


def _target():
    state = PatientState(
        constructs={
            "neuroinflammation": Uncertain(value=1.0, std=0.1),
            "pain_index": Uncertain(value=0.1, std=0.1),
        }
    )
    return state_to_target(state)


def test_state_to_target_maps_dominant():
    tp = _target()
    assert tp.target_id == "TNF_alpha"
    assert "neuroinflammation" in tp.driving_constructs


def test_design_outputs_valid_smiles():
    gen = MoleculeGenerator(seed=1)
    results = gen.design(_target(), population=20, generations=4, top_k=5)
    assert results
    for r in results:
        assert Chem.MolFromSmiles(r.smiles) is not None
        assert 0.0 <= r.score <= 1.5


def test_ga_improves_score():
    gen = MoleculeGenerator(seed=2)
    target = _target()
    # Baseline: best of the raw seed library.
    from neuroforge.design.library import SEED_SMILES

    seed_best = max(design_score(Chem.MolFromSmiles(s), target) for s in SEED_SMILES)
    results = gen.design(target, population=30, generations=10, top_k=3)
    assert results[0].score >= seed_best  # GA should not do worse than the best seed


def test_determinism():
    a = MoleculeGenerator(seed=5).design(_target(), population=16, generations=4)
    b = MoleculeGenerator(seed=5).design(_target(), population=16, generations=4)
    assert [r.smiles for r in a] == [r.smiles for r in b]


def test_target_for_each_construct_exists():
    for tid in TARGETS:
        assert TARGETS[tid].ideal_pharmacophore.shape[0] == 8
