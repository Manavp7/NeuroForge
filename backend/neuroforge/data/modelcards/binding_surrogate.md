# Model Card — Binding Surrogate

**Type:** Analytic pharmacophore-distance model (default), optional sklearn MLP / torch ensemble.
**Purpose:** Estimate a *pseudo* binding affinity (pKi-like) of a molecule for a mock target, with
ensemble uncertainty and out-of-distribution inflation.

- **Inputs:** RDKit pharmacophore descriptor vector + a mock target's "ideal" vector.
- **Outputs:** pseudo-pKi value + uncertainty.
- **Training data (neural variants):** distilled from the analytic teacher over the seed library.
- **Limitations:** NOT a real docking/MD/structure-based affinity; targets are illustrative vectors,
  not real binding sites; values are not physically calibrated.
- **NOT intended for:** real affinity prediction or any decision-making. Simulation/demo only.
