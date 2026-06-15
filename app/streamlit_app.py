"""Streamlit dashboard for the NeuroForge synthetic simulator."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from neuroforge.data import band_powers_to_frame, biomarkers_to_frame, state_scores_to_frame
from neuroforge.orchestrator import ClosedLoopOrchestrator
from neuroforge.schemas import SAFETY_DISCLAIMER
from neuroforge.validation import ValidationThresholds


st.set_page_config(page_title="NeuroForge Synthetic Simulator", layout="wide")


@st.cache_resource
def orchestrator() -> ClosedLoopOrchestrator:
    return ClosedLoopOrchestrator()


st.title("NeuroForge")
st.caption("Synthetic closed-loop neural-biomarker-to-candidate simulator")
st.warning(SAFETY_DISCLAIMER)

thresholds = ValidationThresholds()
with st.sidebar:
    st.header("Simulation controls")
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1)
    steps = st.slider("Loop iterations", min_value=1, max_value=12, value=4)
    doctor_approved = st.toggle("Doctor approval flag", value=False)
    require_approval = st.toggle("Require approval gate", value=True)
    st.subheader("Safety thresholds")
    st.write(
        {
            "min_efficacy": thresholds.min_efficacy,
            "max_toxicity": thresholds.max_toxicity,
            "max_off_target": thresholds.max_off_target,
            "max_uncertainty": thresholds.max_uncertainty,
        }
    )

iterations = orchestrator().run_session(
    seed=int(seed),
    steps=int(steps),
    doctor_approved=doctor_approved,
    require_approval=require_approval,
)
latest = iterations[-1]

patient_col, status_col = st.columns([2, 1])
with patient_col:
    st.subheader("Synthetic patient profile")
    st.json(
        {
            "patient_id": latest.patient.patient_id,
            "age": latest.patient.age,
            "sex": latest.patient.sex,
            "baseline_neuroinflammation_risk": latest.patient.baseline_neuroinflammation_risk,
            "baseline_seizure_susceptibility": latest.patient.baseline_seizure_susceptibility,
            "baseline_mood_instability": latest.patient.baseline_mood_instability,
            "genomic_markers": latest.patient.genomic_markers,
            "proteomic_markers": latest.patient.proteomic_markers,
        }
    )

with status_col:
    st.subheader("Latest gate status")
    st.metric("Dominant state", latest.inferred_state.dominant_state)
    st.metric("State confidence", f"{latest.inferred_state.confidence:.2f}")
    st.metric("Validation passed", "yes" if latest.validation.passed else "no")
    st.metric("Deliverable", "yes" if latest.deliverable else "no")
    st.code(latest.approval_status)

biomarker_df = biomarkers_to_frame(iterations)
band_df = band_powers_to_frame(iterations)
state_df = state_scores_to_frame(latest)

chart_col_1, chart_col_2 = st.columns(2)
with chart_col_1:
    st.subheader("Multimodal biomarker timeline")
    st.plotly_chart(
        px.line(
            biomarker_df,
            x="step",
            y="value",
            color="metric",
            markers=True,
            range_y=[0, 1],
        ),
        use_container_width=True,
    )

with chart_col_2:
    st.subheader("Neural band-power proxies")
    st.plotly_chart(
        px.bar(
            band_df,
            x="step",
            y="power",
            color="band",
            barmode="group",
            range_y=[0, 1],
        ),
        use_container_width=True,
    )

state_col, validation_col = st.columns(2)
with state_col:
    st.subheader("Latest inferred state")
    radar = go.Figure(
        data=[
            go.Scatterpolar(
                r=state_df["score"],
                theta=state_df["state"],
                fill="toself",
                name="state score",
            )
        ]
    )
    radar.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(radar, use_container_width=True)
    with st.expander("Inference explanation"):
        for line in latest.inferred_state.explanation:
            st.write(f"- {line}")

with validation_col:
    st.subheader("Surrogate validation")
    validation_scores = {
        "binding_score": latest.validation.binding_score,
        "efficacy_score": latest.validation.efficacy_score,
        "toxicity_risk": latest.validation.toxicity_risk,
        "off_target_risk": latest.validation.off_target_risk,
        "admet_score": latest.validation.admet_score,
        "uncertainty": latest.validation.uncertainty,
    }
    st.plotly_chart(
        px.bar(
            x=list(validation_scores.keys()),
            y=list(validation_scores.values()),
            range_y=[0, 1],
            labels={"x": "metric", "y": "score"},
        ),
        use_container_width=True,
    )
    st.write("Threshold flags")
    st.json(latest.validation.threshold_flags)

st.subheader("Generated toy candidate")
candidate_col, rationale_col = st.columns([1, 2])
with candidate_col:
    st.code(latest.candidate.smiles, language="text")
    st.write(f"**Target pathway:** {latest.candidate.target_pathway}")
    st.write(f"**Template:** {latest.candidate.template_id}")
    st.json(latest.candidate.descriptors)
with rationale_col:
    st.write(latest.candidate.rationale)
    st.write("Validation warnings")
    for warning in latest.validation.warnings:
        st.write(f"- {warning}")

st.subheader("Audit trail")
for iteration in iterations:
    with st.expander(f"Iteration {iteration.step}: {iteration.approval_status}"):
        for note in iteration.audit_notes:
            st.write(f"- {note}")
