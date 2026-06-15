# NeuroForge

> ⚠️ **RESEARCH / SIMULATION ONLY.** NeuroForge is a software-only demonstration that uses
> **synthetic data and surrogate models**. It is **NOT a medical device**, is **NOT validated**,
> and must **NOT** be used for any clinical, diagnostic, or treatment decision. Nothing here
> designs real drugs or guides real therapy.

NeuroForge is a fully-simulated **adaptive closed-loop** for personalized molecular therapy
research. It demonstrates the end-to-end loop:

```
Synthetic EEG + multi-omics + wearables
  → State inference (+ uncertainty)
  → Generative molecule design (genetic algorithm over RDKit)
  → In-silico validation (ADMET + surrogate binding + ensemble uncertainty + safety gate)
  → Doctor-in-the-loop approval
  → Simulated delivery + response → re-sense → iterate until the patient state stabilizes
```

The "frontier" components (real BCI, AlphaFold3, molecular dynamics, deep generative models) are
deliberately replaced by **clearly-labeled surrogates** so the whole system runs **without GPUs,
API keys, or hardware** and is fully reproducible/testable. Interfaces are left in place so real
models can be swapped in later.

## Repository layout

```
backend/    Python package `neuroforge` (engine) + FastAPI service + CLI demo
frontend/   React + Vite + TypeScript dashboard (Plotly visualizations)
```

## Quickstart

The repo ships a `Makefile` for convenience (`make help` lists everything):

```bash
make install     # backend (pip -e) + frontend (npm) deps
make demo        # headless closed-loop CLI demo
make api         # FastAPI server  -> http://localhost:8000/docs
make web         # Vite dev server -> http://localhost:5173
make test        # backend (pytest) + frontend (vitest)
```

### Backend (manual)

```bash
cd backend
pip install -e ".[dev]"          # installs RDKit, FastAPI, scikit-learn, etc.
python -m neuroforge.cli demo --condition parkinsonian --iters 6
pytest
uvicorn neuroforge.api.app:app --reload
```

### Frontend (manual)

```bash
cd frontend
npm install
npm run dev                      # expects the API on :8000 (override with VITE_API_BASE)
```

### Sample CLI run

```
Patient mood-95040 (condition: mood_disorder)
  [iter 0] plan      ... design a molecule modulating Serotonin transporter to reduce mood index ...
  [iter 0] deliver   Therapy delivered (sim); abnormality 0.785 → 0.646.
  [iter 1] deliver   Therapy delivered (sim); abnormality 0.646 → 0.439.
  [iter 2] deliver   Therapy delivered (sim); abnormality 0.439 → 0.269.
  [iter 3] done      Patient state stabilized (abnormality 0.269).

Final status: stabilized
Abnormality: 0.785 → 0.269 over 3 delivered therapy step(s).
```

## Conditions & configuration

Supported synthetic conditions: `neuroinflammatory`, `parkinsonian`, `mood_disorder`,
`epileptiform`, `healthy_control`.

Environment variables:

| Variable | Effect |
| --- | --- |
| `OPENAI_API_KEY` | If set (and `pip install -e ".[openai]"`), the agent uses OpenAI for plan/critique text; otherwise a deterministic `MockLLM` is used. No key required. |
| `NEUROFORGE_OPENAI_MODEL` | Override the OpenAI model name (default `gpt-4o-mini`). |
| `VITE_API_BASE` | Frontend API base URL (default `http://localhost:8000`). |

## Architecture

See [`backend/neuroforge`](backend/neuroforge) for module-by-module code:

| Module | Responsibility |
| --- | --- |
| `data/` | Synthetic patient generator + EEG simulator (sensing layer) |
| `inference/` | Multimodal state estimation with uncertainty |
| `design/` | Property objectives + genetic-algorithm molecule generator |
| `validation/` | ADMET descriptors, surrogate binding, ensemble uncertainty, safety gate |
| `explain/` | Feature attributions for inference & design decisions |
| `agent/` | Pluggable LLM client (deterministic mock by default; optional OpenAI) |
| `loop/` | Closed-loop orchestrator (sense→infer→design→validate→approve→monitor) |
| `api/` | FastAPI service (REST + SSE streaming + molecule SVG) |
| `cli.py` | Headless full-loop demo |

## v2 capabilities

NeuroForge has grown well beyond the MVP. Highlights:

**Science / modeling**
- **PK/PD dosing model** — one-compartment oral PK + Emax PD (SciPy ODE) drives efficacy.
- **3D conformers** — RDKit ETKDG+MMFF, shape descriptors, MolBlock for a 3D viewer.
- **Neural binding surrogate** — scikit-learn MLP ensemble (always on) + optional torch MC-dropout (`[ml]`).
- **Generative SMILES-VAE** — optional torch engine behind the generator interface (GA fallback).
- **Polypharmacology** — extra targets (COX-2, AChE, NMDA) + combination-therapy designer.
- **Real EEG ingestion** — MNE EDF/BDF adapter (`[bio]`) sharing the simulator's feature extractor.

**Agent / decisions**
- **Critique → redesign** — tightens constraints and re-designs when a round is weak.
- **Explainability** — SHAP (`[explain]`) with an occlusion fallback; `/patients/{id}/explain`.
- **Uncertainty-aware** — risk-adjusted scoring + confidence-scaled candidate selection.
- **RAG** — TF-IDF over a mechanism corpus; citations attached to plan events; `/rag`.

**Privacy / governance**
- **Federated learning** — FedAvg across simulated sites (`neuroforge bench`/`federated` CLIs).
- **Differential privacy** — Gaussian mechanism for federated updates.
- **Audit log** — tamper-evident hash chain; `/runs/{id}/audit`.
- **Model cards** — per-component datasheets; `/modelcards`.

**Product / ops**
- **Dashboard tabs** — Closed Loop, Cohort (population outcomes), Molecule Lab (edit + validate +
  3D viewer), Compare runs; plus a clinician/researcher role selector and live trajectory chart.
- **Persistence** — SQLite (`NEUROFORGE_DB`) for patients/runs/audit; run & patient history endpoints.
- **Docker** — `docker compose up --build`; **CI** — lint, tests, benchmark, and Playwright e2e.
- **Typed client** — `npm run gen:types` generates TS types from the live OpenAPI schema.

Optional extras: `pip install -e ".[ml]"` (torch), `".[bio]"` (MNE), `".[explain]"` (SHAP),
`".[openai]"` (OpenAI agent). All are optional with graceful fallbacks.

CLIs: `neuroforge demo`, `neuroforge bench`, `neuroforge federated`.

## License

MIT (see `pyproject.toml`). Provided with no warranty; research use only.
