# NeuroForge — backend

Python engine + FastAPI service for the NeuroForge adaptive closed-loop simulator.

> ⚠️ **RESEARCH / SIMULATION ONLY.** Synthetic data and surrogate models. NOT a medical device,
> NOT validated, NOT for any clinical, diagnostic, or treatment use.

See the [repository README](../README.md) for the full overview, architecture, and quickstart.

```bash
pip install -e ".[dev]"
python -m neuroforge.cli demo
uvicorn neuroforge.api.app:app --reload
pytest
```

### Optional extras

| Extra | Adds |
| --- | --- |
| `.[ml]` | Torch-based neural binding surrogate + SMILES-VAE generator |
| `.[bio]` | MNE-based real EEG (EDF/BDF) ingestion |
| `.[explain]` | SHAP explanations |
| `.[openai]` | OpenAI-backed agent (otherwise a deterministic MockLLM is used) |

All extras are optional; the core runs without them.
