from fastapi.testclient import TestClient

from neuroforge.api import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_health_route() -> None:
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "synthetic-research-simulator"


def test_safety_route_exposes_disclaimer_and_thresholds() -> None:
    response = client().get("/safety")
    payload = response.json()

    assert response.status_code == 200
    assert payload["clinical_use"] == "not_allowed"
    assert payload["data_scope"] == "synthetic_only"
    assert "max_toxicity" in payload["thresholds"]
    assert "synthetic research simulator" in payload["disclaimer"]


def test_simulate_iteration_route_returns_closed_loop_payload() -> None:
    response = client().post(
        "/simulate/iteration",
        json={"seed": 12, "step": 1, "doctor_approved": False},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["step"] == 1
    assert payload["patient"]["synthetic"] is True
    assert payload["deliverable"] is False
    assert "candidate" in payload
    assert "validation" in payload


def test_simulate_session_route_returns_requested_iterations() -> None:
    response = client().post(
        "/simulate/session",
        json={"seed": 12, "steps": 2, "doctor_approved": True},
    )
    payload = response.json()

    assert response.status_code == 200
    assert len(payload["iterations"]) == 2
    assert payload["metadata"]["steps"] == 2
    assert len({item["patient"]["patient_id"] for item in payload["iterations"]}) == 1
