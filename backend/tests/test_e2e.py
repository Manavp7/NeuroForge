"""End-to-end smoke test: drive the whole API closed loop to a terminal state."""

import pytest
from fastapi.testclient import TestClient

import neuroforge.api.app as appmod
from neuroforge.inference.state import StateEstimator
from neuroforge.loop.orchestrator import ClosedLoopController


@pytest.fixture(scope="module")
def client():
    est = StateEstimator(seed=3, n_train=150, ensemble=8)
    appmod._CONTROLLER = ClosedLoopController(
        seed=3, estimator=est, ga_population=20, ga_generations=5, ga_top_k=5
    )
    return TestClient(appmod.app)


def test_full_loop_via_stream_stabilizes(client):
    pid = client.post("/patients", json={"condition": "parkinsonian", "seed": 3}).json()["patient"][
        "id"
    ]
    rid = client.post("/runs", json={"patient_id": pid, "max_iter": 6}).json()["run_id"]

    with client.stream("GET", f"/runs/{rid}/stream") as resp:
        body = "".join(chunk for chunk in resp.iter_text())
    assert "eof" in body
    assert "deliver" in body  # at least one therapy was delivered

    run = client.get(f"/runs/{rid}").json()["run"]
    assert run["status"] in {"stabilized", "exhausted", "rejected"}
    assert run["iterations"]
    # Net improvement: final delivered abnormality below the first iteration's baseline.
    delivered = [
        it for it in run["iterations"] if it["approved"] and it["abnormality_after"] is not None
    ]
    if delivered:
        assert delivered[-1]["abnormality_after"] < run["iterations"][0]["abnormality_before"]
