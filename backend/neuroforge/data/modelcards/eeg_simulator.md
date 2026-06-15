# Model Card / Datasheet — EEG & Multi-omics Simulator

**Type:** Procedural synthetic data generator (NumPy).
**Purpose:** Produce noisy, non-stationary EEG band powers and correlated multi-omics/wearable/lab
features driven by a hidden latent state, for the whole pipeline to consume.

- **EEG:** band oscillations modulated by latent state + 1/f noise + eye-blink/EMG artifacts +
  session drift. Real EDF/BDF can be ingested via the optional MNE adapter.
- **Omics/wearables/labs:** correlated with the latent state plus Gaussian noise.
- **Provenance:** fully synthetic; deterministic given a seed. No real patient data is used.
- **Limitations:** hand-designed correlations; not representative of any real population.
- **NOT intended for:** training models for real-world deployment. Simulation/demo only.
