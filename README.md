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

## Quickstart (backend)

```bash
cd backend
pip install -e ".[dev]"          # installs RDKit, FastAPI, scikit-learn, etc.
python -m neuroforge.cli demo    # run the full closed loop headless
pytest                           # run the test suite
uvicorn neuroforge.api.app:app --reload   # start the API (http://localhost:8000/docs)
```

## Quickstart (frontend)

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173 (expects API on :8000)
```

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

## License

MIT (see `pyproject.toml`). Provided with no warranty; research use only.
