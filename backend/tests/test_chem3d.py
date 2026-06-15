from neuroforge.chem3d import descriptors_3d, embed_3d, mol_to_molblock, shape_profile


def test_embed_and_molblock():
    mol = embed_3d("CC(=O)Oc1ccccc1C(=O)O", seed=3)  # aspirin
    assert mol is not None
    assert mol.GetNumConformers() == 1
    block = mol_to_molblock("CC(=O)Oc1ccccc1C(=O)O", seed=3)
    assert block is not None
    assert "V2000" in block or "V3000" in block


def test_3d_descriptors_ranges():
    d = shape_profile("c1ccccc1", seed=3)  # benzene (planar/disc-like)
    assert d is not None
    # Normalized principal moment ratios live in [0, 1].
    assert 0.0 <= d["npr1"] <= 1.0
    assert 0.0 <= d["npr2"] <= 1.0
    assert d["radius_of_gyration"] > 0


def test_invalid_smiles():
    assert embed_3d("not_a_smiles") is None
    assert mol_to_molblock("not_a_smiles") is None


def test_descriptors_from_mol():
    mol = embed_3d("CCO", seed=1)
    d = descriptors_3d(mol)
    assert set(d) >= {"npr1", "npr2", "radius_of_gyration", "asphericity"}
