# NeuroForge MVP Architecture

NeuroForge currently implements a synthetic, software-only closed-loop simulator.
It is designed to make the product workflow tangible while explicitly avoiding
clinical claims or real therapeutic outputs.

## Data flow

```text
SyntheticPatientGenerator
  ├─ PatientProfile
  ├─ BiomarkerSnapshot
  └─ NeuralSignalWindow
        |
        v
StateInferenceEngine
  └─ PatientState with feature contributions and explanation
        |
        v
CandidateGenerator
  └─ MoleculeCandidate from a controlled toy template library
        |
        v
SurrogateValidator
  └─ ValidationResult with threshold flags and warnings
        |
        v
ClosedLoopOrchestrator
  └─ ClosedLoopIteration with approval status and audit notes
```

## Module responsibilities

### `schemas.py`

Defines Pydantic models shared by the API, dashboard, orchestrator, and tests.
The schemas enforce normalized synthetic values, positive signal sample rates,
signal shape consistency, and delivery-gate consistency.

### `synthetic.py`

Generates seeded synthetic inputs:

- synthetic patient baselines and mock omics proxies;
- wearable/lab-style biomarkers such as inflammation, stress, HRV, sleep
  recovery, and neurotransmitter-balance proxies;
- EEG-like multi-channel neural windows with theta/alpha/beta/gamma band-power
  summaries.

The generator is deterministic for a given seed and step.

### `inference.py`

Infers transparent state proxies with heuristic weighted scoring:

- neuroinflammation;
- pain risk;
- seizure risk;
- mood instability.

The engine returns feature contributions and plain-English explanations so the
demo remains inspectable before replacing any component with learned models.

### `generation.py`

Generates a toy candidate from a small controlled template library. The selected
template is tied to the dominant inferred state, and only small controlled
fragments may be appended for simulator variability. Descriptor extraction is a
lightweight string-based fallback so the MVP does not depend on RDKit.

### `validation.py`

Computes surrogate scores:

- binding score;
- efficacy score;
- toxicity risk;
- off-target risk;
- ADMET score;
- uncertainty.

Thresholds gate candidates by minimum efficacy and maximum toxicity,
off-target, and uncertainty. Unknown or uncontrolled templates are penalized and
blocked.

### `orchestrator.py`

Wires all modules into a complete loop. It produces an audit trail for every
iteration and enforces the approval gate:

- validation failure always blocks simulated delivery;
- when approval is required, validation pass alone is insufficient;
- simulated delivery is possible only when the approval gate is satisfied.

### `api.py`

FastAPI app factory with:

- `GET /health`;
- `GET /safety`;
- `POST /simulate/iteration`;
- `POST /simulate/session`.

### `app/streamlit_app.py`

Local dashboard for exploring seeded sessions with charts for biomarkers, band
powers, inferred states, validation scores, candidate descriptors, and audit
notes.

## Safety gates

The MVP has safety controls at multiple layers:

1. **Synthetic-only schema gate** — `PatientProfile.synthetic` must be true.
2. **Controlled generation gate** — candidates originate from template IDs known
   to the toy library.
3. **Surrogate threshold gate** — candidates fail when efficacy is too low or
   toxicity, off-target risk, or uncertainty is too high.
4. **Approval gate** — simulated delivery remains blocked unless validation
   passes and the configured approval requirement is satisfied.
5. **Audit gate** — every iteration records synthetic-data notices, validation
   warnings, and approval status.

## Extension points

Future modules can replace the current transparent surrogates without changing
the surrounding application flow:

- **BCI adapters:** MNE, OpenBCI, Emotiv, or file-based signal loaders can
  produce `NeuralSignalWindow` objects.
- **Omics/wearables adapters:** data connectors can produce
  `BiomarkerSnapshot` objects after privacy and governance controls exist.
- **Chemistry descriptors:** RDKit can replace string descriptors while keeping
  `MoleculeCandidate` stable.
- **Generative models:** diffusion, VAE, GNN, or LLM-orchestrated generators can
  implement the same candidate interface but must preserve safety gates.
- **Physics validation:** AlphaFold/OpenMM/GROMACS-style workflows can feed
  richer validation scores into `ValidationResult`.
- **Explainability:** SHAP/LIME or model cards can enrich patient-state and
  candidate rationales.
- **Federated learning:** privacy-preserving training can be layered around the
  synthetic cohort/session APIs.

## Non-goals for the current MVP

- No clinical decision support.
- No real patient data.
- No real BCI hardware integration.
- No validated molecule generation.
- No dosing or treatment recommendations.
- No autonomous therapy delivery.
