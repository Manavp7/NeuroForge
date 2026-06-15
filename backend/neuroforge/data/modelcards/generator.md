# Model Card — Molecule Generator

**Type:** Genetic algorithm over RDKit molecules (default); optional SMILES-VAE (torch).
**Purpose:** Propose de novo small molecules optimized toward a target property/pharmacophore profile.

- **Inputs:** a TargetProfile (property windows + ideal pharmacophore) and optional constraints.
- **Outputs:** ranked valid SMILES with design scores + provenance.
- **Operators (GA):** atom/bond mutation, fragment crossover, with RDKit sanitization.
- **Limitations:** explores a small chemical space seeded from a curated library; the VAE is trained
  on a tiny corpus and is illustrative; no synthesizability guarantees beyond a heuristic SA proxy.
- **NOT intended for:** proposing real drug candidates. Simulation/demo only.
