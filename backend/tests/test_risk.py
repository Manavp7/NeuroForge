from neuroforge.models import Uncertain
from neuroforge.validation.pipeline import composite_score, risk_adjusted


def test_risk_adjustment_penalizes_uncertainty():
    score = 0.8
    confident = Uncertain(value=8.0, std=0.1)
    uncertain = Uncertain(value=8.0, std=1.4)
    assert risk_adjusted(score, confident) > risk_adjusted(score, uncertain)
    # No risk aversion -> no penalty.
    assert risk_adjusted(score, uncertain, risk_aversion=0.0) == score


def test_risk_adjusted_in_pipeline():
    from rdkit import Chem

    from neuroforge.design.objectives import state_to_target
    from neuroforge.models import PatientState
    from neuroforge.validation.pipeline import evaluate_molecule

    st = PatientState(constructs={"neuroinflammation": Uncertain(value=1.0, std=0.1)})
    target = state_to_target(st)
    cand = evaluate_molecule("CC(=O)Oc1ccccc1C(=O)O", target, seed=1)
    assert cand is not None
    assert cand.risk_adjusted_score <= cand.score
    assert Chem.MolFromSmiles(cand.smiles) is not None


def test_composite_unchanged_baseline():
    s = composite_score(Uncertain(value=9.0, std=0.0), qed=1.0)
    assert 0.9 <= s <= 1.0
