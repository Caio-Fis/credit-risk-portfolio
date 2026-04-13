"""Explicabilidade SHAP — waterfall interativo por contrato."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contextual.data_generator import generate_dataset
from src.contextual.interaction_model import train_contextual
from src.explain.shap_explain import compute_shap, top_features, waterfall_plot

st.set_page_config(page_title="Explicabilidade SHAP", layout="wide")
st.title("Explicabilidade por SHAP")
st.caption(
    "Waterfall interativo: quais features mais influenciaram cada decisão de crédito."
)


@st.cache_resource(show_spinner="Treinando modelo e calculando SHAP values...")
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
# Importância global
# ---------------------------------------------------------------------------
st.subheader("Importância Global das Features")
top_feat = top_features(explanation, n=15)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_feat["feature"][::-1], top_feat["mean_abs_shap"][::-1], color="steelblue")
ax.set_xlabel("Importância SHAP média (|SHAP|)")
ax.set_title("Top 15 Features — Importância Global")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ---------------------------------------------------------------------------
# Waterfall por contrato
# ---------------------------------------------------------------------------
st.subheader("Waterfall por Contrato")
st.caption(
    "Selecione um contrato para ver como cada feature contribuiu para a decisão."
)

idx = st.slider(
    "Índice do contrato", min_value=0, max_value=min(len(X) - 1, 200), value=0
)

contract_info = df.iloc[idx]
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Produto", contract_info["product_type"])
with c2:
    st.metric("Prazo (meses)", int(contract_info["tenor_months"]))
with c3:
    st.metric("PD real (DGP)", f"{contract_info['pd_true']:.3f}")

fig_wf = waterfall_plot(explanation, idx=idx, max_display=12)
st.pyplot(fig_wf)
