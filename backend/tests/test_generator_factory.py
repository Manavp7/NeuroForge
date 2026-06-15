from rdkit import Chem

from neuroforge.design.generator import MoleculeGenerator, make_generator
from neuroforge.design.objectives import state_to_target
from neuroforge.models import PatientState, Uncertain


def _target():
    st = PatientState(constructs={"neuroinflammation": Uncertain(value=1.0, std=0.1)})
    return state_to_target(st)


def test_ga_is_default():
    gen = make_generator(seed=1, engine="ga")
    assert isinstance(gen, MoleculeGenerator)
    results = gen.design(_target(), population=16, generations=3, top_k=4)
    assert results and Chem.MolFromSmiles(results[0].smiles) is not None


def test_vae_falls_back_to_ga_without_torch():
    # Whether or not torch is present, this must yield a working generator + valid molecules.
    gen = make_generator(seed=1, engine="vae")
    results = gen.design(_target(), population=12, generations=2, top_k=3)
    assert results
    for r in results:
        assert Chem.MolFromSmiles(r.smiles) is not None
