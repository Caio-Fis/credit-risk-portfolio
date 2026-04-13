"""Contextual Score — same client, different products live."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contextual.data_generator import generate_dataset
from src.contextual.interaction_model import score_by_context, train_contextual

st.set_page_config(page_title="Contextual Score", layout="wide")
st.title("Contextual Score")
st.caption(
    "Live demonstration: the same client has radically different risk "
    "depending on the product and tenor requested."
)


@st.cache_resource(show_spinner="Training contextual model...")
def load_contextual_model():
    df = generate_dataset(n=3000, seed=42)
    model, feature_names = train_contextual(df, seed=42)
    return model, feature_names


model_ctx, feature_names = load_contextual_model()

# ---------------------------------------------------------------------------
# Client profile
# ---------------------------------------------------------------------------
st.subheader("Configure Client Profile")
col_a, col_b, col_c = st.columns(3)

with col_a:
    score_fin = st.slider("Financial score (0=weak, 1=strong)", 0.0, 1.0, 0.5, 0.05)
with col_b:
    faturamento = st.number_input(
        "Annual revenue", 50_000, 10_000_000, 500_000, 50_000
    )
with col_c:
    has_collateral = int(st.checkbox("Has real collateral"))

ead = st.number_input("Operation amount (EAD)", 5_000, 2_000_000, 100_000, 5_000)

client_profile = {
    "score_financeiro": score_fin,
    "idade_empresa_anos": 5,
    "faturamento_anual": faturamento,
    "has_collateral": has_collateral,
    "ead": ead,
}

# ---------------------------------------------------------------------------
# Live calculation
# ---------------------------------------------------------------------------
scores = score_by_context(client_profile, model_ctx, feature_names)
scores["el_value"] = scores["el_contextual"]
scores["pd_pct"] = scores["pd_contextual"] * 100

# PD Heatmap
st.subheader("Contextual PD — Product × Tenor (same client)")
pivot_pd = scores.pivot(
    index="tenor_months", columns="product_type", values="pd_contextual"
)

fig_heat = px.imshow(
    pivot_pd,
    text_auto=".3f",
    color_continuous_scale="RdYlGn_r",
    zmin=0,
    zmax=min(0.5, pivot_pd.values.max() * 1.5),
    labels={"x": "Product", "y": "Tenor (months)", "color": "PD"},
    title="PD by Product × Tenor (a single score would ignore these differences)",
)
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# EL by product chart
# ---------------------------------------------------------------------------
st.subheader("Expected Loss by Product and Tenor")
fig_el = px.line(
    scores.sort_values("tenor_months"),
    x="tenor_months",
    y="el_value",
    color="product_type",
    markers=True,
    labels={
        "tenor_months": "Tenor (months)",
        "el_value": "Expected Loss",
        "product_type": "Product",
    },
    title="How EL evolves with tenor — different by product",
)
st.plotly_chart(fig_el, use_container_width=True)

# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
min_el = scores["el_value"].min()
max_el = scores["el_value"].max()
ratio = max_el / (min_el + 0.01)

st.divider()
st.subheader("The central argument")
st.info(f"""
For this client with financial score **{score_fin:.2f}** and EAD of **{ead:,.0f}**:

- **Minimum** Expected Loss: {min_el:,.0f} (lowest-risk operation)
- **Maximum** Expected Loss: {max_el:,.0f} (highest-risk operation)
- **Max/min ratio**: {ratio:.1f}x

Using a single score means treating all these cases as equivalent.
The contextual model **differentiates {ratio:.1f}x** the risk across products/tenors for the same client.
""")
