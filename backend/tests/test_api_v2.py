import importlib

import pytest
from fastapi.testclient import TestClient

import neuroforge.api.app as appmod
from neuroforge.inference.state import StateEstimator
from neuroforge.loop.orchestrator import ClosedLoopController


@pytest.fixture(scope="module")
def client():
    est = StateEstimator(seed=3, n_train=150, ensemble=8)
    appmod._CONTROLLER = ClosedLoopController(
        seed=3, estimator=est, ga_population=16, ga_generations=4, ga_top_k=4
    )
    return TestClient(appmod.app)


def test_targets_and_evaluate(client):
    tids = {t["id"] for t in client.get("/targets").json()["targets"]}
    assert "TNF_alpha" in tids and "COX2" in tids
    r = client.post("/molecule/evaluate", json={"smiles": "CCO", "target_id": "TNF_alpha"})
    assert r.status_code == 200
    assert r.json()["candidate"]["smiles"] == "CCO"
    bad = client.post("/molecule/evaluate", json={"smiles": "xx!!", "target_id": "TNF_alpha"})
    assert bad.status_code == 422


def test_cohort(client):
    r = client.post("/cohort", json={"condition": "neuroinflammatory", "n": 3, "seed": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["n"] == 3
    assert len(body["patients"]) == 3


def test_explain_endpoint(client):
    pid = client.post("/patients", json={"condition": "mood_disorder", "seed": 5}).json()[
        "patient"
    ]["id"]
    r = client.get(f"/patients/{pid}/explain")
    assert r.status_code == 200
    assert "factors" in r.json()


def test_modelcards(client):
    cards = client.get("/modelcards").json()["modelcards"]
    assert cards
    cid = cards[0]["id"]
    assert "content" in client.get(f"/modelcards/{cid}").json()


def test_rag_endpoint(client):
    r = client.get("/rag", params={"query": "serotonin mood SERT"})
    assert r.status_code == 200
    assert r.json()["citations"]


def test_molblock_endpoint(client):
    r = client.get("/molecule/molblock", params={"smiles": "CCO"})
    assert r.status_code == 200
    assert "molblock" in r.json()


def test_auth_optional_then_enforced(monkeypatch):
    # With no token configured, approval role is advisory (handled in other tests).
    # Configure a token and confirm role enforcement on a fresh app import.
    monkeypatch.setenv("NEUROFORGE_API_TOKEN", "secret")
    import neuroforge.api.auth as auth

    importlib.reload(auth)
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        auth.require_clinician(x_role="researcher")
    assert auth.require_clinician(x_role="clinician") == "clinician"
    monkeypatch.delenv("NEUROFORGE_API_TOKEN", raising=False)
    importlib.reload(auth)
