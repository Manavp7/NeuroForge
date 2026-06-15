from rdkit import Chem

from neuroforge.design.objectives import state_to_target
from neuroforge.models import PatientState, Uncertain
from neuroforge.validation.admet import compute_admet
from neuroforge.validation.binding import BindingPredictor
from neuroforge.validation.pipeline import evaluate_molecule
from neuroforge.validation.uncertainty import safety_gate


def _target():
    st = PatientState(constructs={"neuroinflammation": Uncertain(value=1.0, std=0.1)})
    return state_to_target(st)


def test_admet_ranges_sane():
    a = compute_admet(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    assert 150 < a.mol_weight < 220
    assert 0.0 <= a.qed <= 1.0
    assert a.lipinski_violations == 0
    assert 1.0 <= a.sa_score <= 10.0


def test_tox_flag_detected():
    a = compute_admet(Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1"))  # nitrobenzene
    assert "nitro_group" in a.tox_flags


def test_binding_has_uncertainty():
    bp = BindingPredictor("TNF_alpha", seed=1)
    u = bp.predict(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert 4.0 <= u.value <= 9.0
    assert u.std >= 0.0


def test_safety_gate_rejects_toxic():
    cand = evaluate_molecule("O=[N+]([O-])c1ccccc1", _target(), seed=1)
    assert cand is not None
    assert cand.safe is False
    assert cand.safety_notes


def test_invalid_smiles_returns_none():
    assert evaluate_molecule("not_a_smiles", _target(), seed=1) is None


def test_evaluate_good_molecule():
    cand = evaluate_molecule("CC(=O)Oc1ccccc1C(=O)O", _target(), seed=1)
    assert cand is not None
    assert cand.id.startswith("cand-")
    assert 0.0 <= cand.score <= 1.0
    safe, notes = safety_gate(cand.admet, cand.binding)
    assert isinstance(safe, bool)
