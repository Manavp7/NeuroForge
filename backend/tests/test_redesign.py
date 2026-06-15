from rdkit import Chem

from neuroforge.design.constraints import (
    DesignConstraints,
    derive_constraints,
    passes_constraints,
)
from neuroforge.design.objectives import design_score, state_to_target
from neuroforge.models import ADMET, Candidate, PatientState, Uncertain


def _target():
    st = PatientState(constructs={"neuroinflammation": Uncertain(value=1.0, std=0.1)})
    return state_to_target(st)


def _cand(cid, qed, binding, tox=None, sa=3.0):
    return Candidate(
        id=cid,
        smiles="CCO",
        admet=ADMET(qed=qed, sa_score=sa, tox_flags=tox or []),
        binding=Uncertain(value=binding, std=0.1),
        score=0.5,
        safe=not tox,
    )


def test_derive_constraints_reacts_to_failures():
    cands = [_cand("a", 0.4, 5.5, tox=["nitro_group"]), _cand("b", 0.45, 6.0)]
    c = derive_constraints(cands, _target())
    assert c.exclude_tox is True
    assert c.min_binding >= 6.0
    assert "qed" in c.property_windows  # low QED -> tightened


def test_passes_constraints_filters():
    c = DesignConstraints(exclude_tox=True, min_binding=6.5, max_sa=5.0)
    good = _cand("g", 0.7, 7.0)
    toxic = _cand("t", 0.7, 7.0, tox=["PAINS"])
    weak = _cand("w", 0.7, 6.0)
    hard = _cand("h", 0.7, 7.0, sa=8.0)
    assert passes_constraints(good, c)
    assert not passes_constraints(toxic, c)
    assert not passes_constraints(weak, c)
    assert not passes_constraints(hard, c)


def test_constraints_affect_design_score():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    target = _target()
    base = design_score(mol, target)
    constrained = design_score(mol, target, DesignConstraints(max_sa=0.0))  # force SA penalty
    assert constrained < base
