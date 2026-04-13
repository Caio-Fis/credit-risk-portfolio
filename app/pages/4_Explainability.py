"""SHAP Explainability — interactive waterfall per contract."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contextual.data_generator import generate_dataset
from src.contextual.interaction_model import train_contextual
from src.explain.shap_explain import compute_shap, top_features, waterfall_plot

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("Explainability via SHAP")
st.caption(
    "Interactive waterfall: which features most influenced each credit decision."
)


@st.cache_resource(show_spinner="Training model and computing SHAP values...")
def load_model_and_shap():
    df = generate_dataset(n=2000, seed=42)
    model, feature_names = train_contextual(df, seed=42)

    from src.contextual.context_features import (
        add_context_features,
        create_interaction_features,
        encode_product,
    )

    df_feat = add_context_features(df.copy())
    df_feat = encode_product(df_feat, drop_first=True)
    df_feat = create_interaction_features(df_feat)
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    X = df_feat[feature_names].fillna(0)

    explanation = compute_shap(model, X)
    return df, X, explanation, feature_names


df, X, explanation, feature_names = load_model_and_shap()

# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
st.subheader("Global Feature Importance")
top_feat = top_features(explanation, n=15)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_feat["feature"][::-1], top_feat["mean_abs_shap"][::-1], color="steelblue")
ax.set_xlabel("Mean SHAP importance (|SHAP|)")
ax.set_title("Top 15 Features — Global Importance")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ---------------------------------------------------------------------------
# Waterfall per contract
# ---------------------------------------------------------------------------
st.subheader("Waterfall per Contract")
st.caption(
    "Select a contract to see how each feature contributed to the credit decision."
)

idx = st.slider(
    "Contract index", min_value=0, max_value=min(len(X) - 1, 200), value=0
)

contract_info = df.iloc[idx]
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Product", contract_info["product_type"])
with c2:
    st.metric("Tenor (months)", int(contract_info["tenor_months"]))
with c3:
    st.metric("True PD (DGP)", f"{contract_info['pd_true']:.3f}")

fig_wf = waterfall_plot(explanation, idx=idx, max_display=12)
st.pyplot(fig_wf)
