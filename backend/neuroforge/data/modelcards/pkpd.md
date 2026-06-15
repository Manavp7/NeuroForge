# Model Card — PK/PD Model

**Type:** One-compartment oral pharmacokinetics + Emax pharmacodynamics (scipy ODE).
**Purpose:** Translate a candidate's potency + a dosing regimen into a steady-state efficacy used by
the response simulator.

- **Inputs:** dose/regimen, absorption/elimination/volume parameters, predicted potency (EC50).
- **Outputs:** concentration-time curve, Cmax/Css, mean steady-state effect fraction.
- **Limitations:** generic textbook parameters; not patient-specific; not validated pharmacology.
- **NOT intended for:** dosing decisions of any kind. Simulation/demo only.
