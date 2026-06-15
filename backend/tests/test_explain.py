from neuroforge.data.synthetic import SyntheticPatientGenerator
from neuroforge.explain.shap_explain import explain_state, shap_available
from neuroforge.inference.state import StateEstimator


def test_occlusion_explanation_structure():
    est = StateEstimator(seed=3, n_train=150, ensemble=6)
    profile = SyntheticPatientGenerator(seed=9).generate("neuroinflammatory")
    result = explain_state(est, profile, method="occlusion", top_k=4)
    assert result["method"] == "occlusion"
    assert "neuroinflammation" in result["factors"]
    factors = result["factors"]["neuroinflammation"]
    assert len(factors) == 4
    assert {"feature", "label", "attribution"} <= set(factors[0])


def test_inflammatory_drivers_are_inflammatory_features():
    est = StateEstimator(seed=3, n_train=180, ensemble=8)
    profile = SyntheticPatientGenerator(seed=21).generate("neuroinflammatory")
    result = explain_state(est, profile, method="occlusion", top_k=6)
    feats = {f["feature"] for f in result["factors"]["neuroinflammation"]}
    # Inflammatory markers / CRP should be among the top drivers.
    assert feats & {"proteo_IL6", "proteo_TNFa", "proteo_CRP", "lab_crp_mgL", "geno_inflammatory"}


def test_shap_available_is_bool():
    assert isinstance(shap_available(), bool)
