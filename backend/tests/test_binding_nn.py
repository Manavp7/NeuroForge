import numpy as np
from rdkit import Chem

from neuroforge.validation.binding import BindingPredictor, make_predictor
from neuroforge.validation.binding_nn import MLPBindingPredictor


def test_mlp_approximates_teacher():
    target = "TNF_alpha"
    teacher = BindingPredictor(target, seed=3)
    mlp = MLPBindingPredictor(target, seed=3, ensemble=4, n_train=300)
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccncc1"]
    errs = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        errs.append(abs(teacher.predict(mol).value - mlp.predict(mol).value))
    assert np.mean(errs) < 1.0  # learned surrogate tracks the teacher


def test_mlp_has_uncertainty_and_bounds():
    mlp = MLPBindingPredictor("D2", seed=1, ensemble=4, n_train=250)
    u = mlp.predict(Chem.MolFromSmiles("CC(N)Cc1ccccc1"))
    assert 4.0 <= u.value <= 9.0
    assert u.std >= 0.0


def test_factory_selects_models():
    heur = make_predictor("SERT", seed=2, kind="heuristic")
    assert type(heur).__name__ == "BindingPredictor"
    mlp = make_predictor("SERT", seed=2, kind="mlp")
    assert type(mlp).__name__ in {"MLPBindingPredictor", "TorchBindingPredictor"}
    # Cached: same instance returned.
    assert make_predictor("SERT", seed=2, kind="mlp") is mlp


def test_torch_kind_falls_back_when_unavailable():
    # Whether or not torch is installed, this must return a working predictor.
    pred = make_predictor("GABA_A", seed=5, kind="torch")
    u = pred.predict(Chem.MolFromSmiles("CCO"))
    assert 4.0 <= u.value <= 9.0
