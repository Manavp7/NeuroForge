import pytest
from fastapi.testclient import TestClient

import neuroforge.api.app as appmod
from neuroforge.inference.state import StateEstimator
from neuroforge.loop.orchestrator import ClosedLoopController


@pytest.fixture(scope="module")
def client():
    # Use a small, fast controller for the API in tests.
    est = StateEstimator(seed=3, n_train=150, ensemble=8)
    appmod._CONTROLLER = ClosedLoopController(
        seed=3, estimator=est, ga_population=16, ga_generations=4, ga_top_k=4
    )
    return TestClient(appmod.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert "disclaimer" in r.json()


def test_conditions(client):
    r = client.get("/conditions")
    assert r.status_code == 200
    assert "neuroinflammatory" in r.json()["conditions"]


def test_create_patient_hides_latent(client):
    r = client.post("/patients", json={"condition": "parkinsonian", "seed": 3})
    assert r.status_code == 200
    patient = r.json()["patient"]
    assert "latent_state" not in patient
    assert patient["condition"] == "parkinsonian"


def test_patient_state_endpoint(client):
    pid = client.post("/patients", json={"condition": "neuroinflammatory", "seed": 5}).json()[
        "patient"
    ]["id"]
    r = client.get(f"/patients/{pid}/state")
    assert r.status_code == 200
    assert "constructs" in r.json()["state"]


def test_manual_run_step_approve(client):
    pid = client.post("/patients", json={"condition": "parkinsonian", "seed": 7}).json()["patient"][
        "id"
    ]
    rid = client.post("/runs", json={"patient_id": pid, "max_iter": 4}).json()["run_id"]
    step = client.post(f"/runs/{rid}/step").json()
    assert step["status"] in {"awaiting_approval", "stabilized"}
    if step["status"] == "awaiting_approval":
        assert step["pending"]["candidates"]
        appr = client.post(f"/runs/{rid}/approve", json={}).json()
        assert appr["status"] == "running"
    run = client.get(f"/runs/{rid}").json()["run"]
    assert run["events"]


def test_reject_without_pending_conflicts(client):
    pid = client.post("/patients", json={"condition": "mood_disorder", "seed": 9}).json()[
        "patient"
    ]["id"]
    rid = client.post("/runs", json={"patient_id": pid, "max_iter": 2}).json()["run_id"]
    r = client.post(f"/runs/{rid}/reject")
    assert r.status_code == 409  # nothing pending yet


def test_stream_run(client):
    pid = client.post("/patients", json={"condition": "neuroinflammatory", "seed": 11}).json()[
        "patient"
    ]["id"]
    rid = client.post("/runs", json={"patient_id": pid, "max_iter": 4}).json()["run_id"]
    with client.stream("GET", f"/runs/{rid}/stream") as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    assert "data:" in body
    assert "eof" in body


def test_molecule_svg(client):
    r = client.get("/molecule/svg", params={"smiles": "CCO"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in r.text


def test_patient_not_found(client):
    assert client.get("/patients/nope").status_code == 404
