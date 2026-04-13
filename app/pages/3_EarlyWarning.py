"""Early Warning — alert list and score trajectory."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.early_warning.behavioral_signals import (
    aggregate_signals,
    flag_volume_drop,
    simulate_behavioral_data,
)
from src.early_warning.score_trajectory import (
    compute_score_trajectory,
    flag_score_drop,
)

st.set_page_config(page_title="Early Warning", layout="wide")
st.title("Early Warning")
st.caption("Credit deterioration detection before default.")


@st.cache_data
def load_alerts():
    df = simulate_behavioral_data(n_entities=500, n_months=12, seed=42)
    trajectory = compute_score_trajectory(df)
    alerts = flag_score_drop(trajectory)
    vol_alerts = flag_volume_drop(df)
    composite = aggregate_signals(alerts, volume_alerts=vol_alerts)
    return df, trajectory, composite


df_behavior, trajectory, composite = load_alerts()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
st.subheader("Alert Summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Monitored entities", f"{len(trajectory):,}")
with c2:
    critico = (composite["composite_risk"] == 3).sum() if not composite.empty else 0
    st.metric(
        "Critical", critico, delta=f"-{critico} vs prior month" if critico else None
    )
with c3:
    alto = (composite["composite_risk"] == 2).sum() if not composite.empty else 0
    st.metric("High risk", alto)
with c4:
    moderado = (composite["composite_risk"] == 1).sum() if not composite.empty else 0
    st.metric("Moderate", moderado)

st.divider()

# ---------------------------------------------------------------------------
# Alert table
# ---------------------------------------------------------------------------
st.subheader("Entities in Alert")
if composite.empty:
    st.info("No active alerts at this time.")
else:
    display_cols = [
        c
        for c in [
            "SK_ID_CURR",
            "score_drop",
            "alert_level",
            "composite_risk",
            "risk_label",
            "signal_volume",
        ]
        if c in composite.columns
    ]
    st.dataframe(
        composite[display_cols].head(50),
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Interactive score trajectory
# ---------------------------------------------------------------------------
st.subheader("Score Trajectory — Individual Analysis")

if not composite.empty:
    entity_ids = composite["SK_ID_CURR"].head(20).tolist()
    selected = st.selectbox("Select entity", options=entity_ids)

    entity_data = df_behavior[df_behavior["SK_ID_CURR"] == selected].sort_values(
        "reference_date"
    )

    if not entity_data.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=entity_data["reference_date"],
                y=entity_data["pd_score"],
                mode="lines+markers",
                name="Score",
                line=dict(color="steelblue", width=2),
            )
        )
        fig.add_hline(
            y=entity_data["pd_score"].iloc[-1],
            line_dash="dash",
            line_color="red",
            annotation_text="Current score",
        )
        fig.update_layout(
            title=f"Score Trajectory — Entity {selected}",
            xaxis_title="Date",
            yaxis_title="Score (0–1000)",
            yaxis=dict(range=[0, 1000]),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        row = trajectory[trajectory["SK_ID_CURR"] == selected]
        if not row.empty:
            drop = float(row["score_drop"].values[0])
            st.metric(
                "Score drop in the last 30 days",
                f"{drop:.0f} pts",
                delta=f"-{drop:.0f} pts",
                delta_color="inverse",
            )
