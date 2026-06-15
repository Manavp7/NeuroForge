# NeuroForge

NeuroForge is a **synthetic, research-only MVP** for exploring a closed-loop
neural-biomarker feedback workflow. The current implementation demonstrates a
safe software vertical slice:

1. Generate a synthetic patient profile.
2. Simulate noisy BCI-like neural windows plus wearable/lab-style biomarkers.
3. Infer transparent patient-state proxies.
4. Generate a controlled toy molecule/candidate from a template library.
5. Run surrogate validation for binding, efficacy, toxicity, off-target risk,
   ADMET, and uncertainty.
6. Apply validation and doctor-in-the-loop gates before any simulated delivery.
7. Inspect the loop through a FastAPI backend or Streamlit dashboard.

## Safety and clinical-use boundary

> NeuroForge MVP is a synthetic research simulator. Outputs are toy surrogate
> artifacts, not medical advice, clinical decision support, dosing guidance, or
> validated molecular designs.

This repository does **not** ingest real patient data, connect to real BCI
hardware, perform validated chemistry, design real drugs, recommend dosing, or
provide autonomous medical decisions. The molecule strings and validation scores
are constrained demonstration artifacts for product and architecture iteration.

## Architecture

```text
synthetic patient + mock neural/wearable/lab streams
        |
        v
state inference engine
        |
        v
controlled toy candidate generator
        |
        v
surrogate validation + uncertainty
        |
        v
safety/approval gate + audit log
        |
        +--> FastAPI JSON API
        +--> Streamlit dashboard
```

Core modules live under `src/neuroforge`:

- `synthetic.py` — seeded synthetic profile, biomarker, and neural-window generation.
- `inference.py` — transparent heuristic patient-state inference.
- `generation.py` — template-bounded toy candidate generation and descriptors.
- `validation.py` — surrogate validation scores and safety thresholds.
- `orchestrator.py` — complete closed-loop session runner and audit trail.
- `api.py` — FastAPI routes for health, safety, and simulation.
- `data.py` — dataframe helpers for dashboard charts.

See `docs/architecture.md` for more detail.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The MVP intentionally avoids heavy optional scientific dependencies such as
RDKit, MNE, OpenMM, or LLM provider SDKs. Future adapters can plug into the same
module boundaries once the product workflow is validated.

## Run the CLI demo

```bash
python -m neuroforge --seed 7 --steps 2
```

Optional flags:

```bash
python -m neuroforge --doctor-approved
python -m neuroforge --no-approval-required
```

## Run the API

```bash
uvicorn neuroforge.api:create_app --factory --reload
```

Useful endpoints:

- `GET /health`
- `GET /safety`
- `POST /simulate/iteration`
- `POST /simulate/session`

Example:

```bash
curl -X POST http://127.0.0.1:8000/simulate/session \
  -H 'content-type: application/json' \
  -d '{"seed": 7, "steps": 3, "doctor_approved": false}'
```

## Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard shows the synthetic patient profile, biomarker timeline, neural
band-power proxies, inferred patient state, generated toy candidate, surrogate
validation scores, and audit trail.

## Run tests

```bash
python -m pytest
```

The test suite is fully local and requires no network access, hardware, GPU,
real patient data, RDKit, MNE, LLM APIs, or external services.

## Roadmap

Near-term extension seams:

- Optional RDKit adapter for real chemical descriptors.
- Optional MNE/OpenBCI/Emotiv adapters for mock-to-real neural-signal ingestion.
- Surrogate model interfaces for PyTorch/JAX/GNN predictors.
- Explainability layer for state and candidate decisions.
- Federated-learning prototype for privacy-preserving synthetic cohorts.
- Stronger audit, governance, and regulatory workflow prototypes.

Each future integration should preserve the current safety boundary: no clinical
recommendations, no autonomous therapy delivery, and no real-patient operation
without rigorous validation and governance.