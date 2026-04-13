"""Credit origination simulator.

Input: client data + product + tenor
Output: calibrated PD, estimated LGD, Expected Loss in currency units

PD estimated by two methods:
- DGP (synthetic oracle): uses the data generating function — maximum transparency
- Contextual model (ML): LightGBM + Platt trained on synthetic contextual dataset
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PRODUCTS, TENORS_MONTHS
from src.contextual.data_generator import DGP_PARAMS, dgp_pd, generate_dataset
from src.contextual.interaction_model import score_by_context, train_contextual

st.set_page_config(page_title="Origination Simulator", layout="wide")
st.title("Credit Origination Simulator")
st.caption(
    "Enter client and operation data to obtain PD, LGD and Expected Loss."
)


@st.cache_resource(show_spinner="Training contextual model...")
def _load_contextual_model():
    df = generate_dataset(n=3000, seed=42)
    return train_contextual(df, seed=42)


model_ctx, feature_names = _load_contextual_model()

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col_cliente, col_operacao = st.columns(2)

with col_cliente:
    st.subheader("Client Profile")
    score_fin = st.slider(
        "Financial score (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    idade_empresa = st.slider(
        "Company age (years)", min_value=1, max_value=30, value=5
    )
    faturamento = st.number_input(
        "Annual revenue",
        min_value=10_000,
        max_value=50_000_000,
        value=500_000,
        step=10_000,
    )

with col_operacao:
    st.subheader("Operation Parameters")
    produto = st.selectbox("Product", options=PRODUCTS)
    prazo = st.selectbox("Tenor (months)", options=TENORS_MONTHS)
    valor_credito = st.number_input(
        "Requested amount (EAD)",
        min_value=1_000,
        max_value=5_000_000,
        value=50_000,
        step=1_000,
    )
    has_collateral = st.checkbox("Real collateral", value=False)

# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
if st.button("Calculate risk", type="primary"):
    rng = np.random.default_rng(0)  # fixed seed for reproducibility

    # PD via DGP (synthetic oracle)
    pd_dgp = dgp_pd(
        score_financeiro=np.array([score_fin]),
        produto=produto,
        prazo_meses=prazo,
        has_collateral=np.array([int(has_collateral)]),
        rng=rng,
    )[0]

    # PD via trained contextual model
    client_profile = {
        "score_financeiro": score_fin,
        "idade_empresa_anos": float(idade_empresa),
        "faturamento_anual": float(faturamento),
        "has_collateral": int(has_collateral),
        "ead": float(valor_credito),
    }
    scores_grid = score_by_context(client_profile, model_ctx, feature_names)
    row = scores_grid[
        (scores_grid["product_type"] == produto)
        & (scores_grid["tenor_months"] == prazo)
    ]
    pd_model = float(row["pd_contextual"].iloc[0]) if not row.empty else pd_dgp

    lgd_base = DGP_PARAMS["lgd_base"][produto]
    lgd_pred = float(np.clip(lgd_base + rng.normal(0, 0.05), 0.01, 0.99))

    # EL using contextual model PD (more informative estimate)
    el = pd_model * lgd_pred * valor_credito

    st.divider()
    st.subheader("Analysis Result")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pd_pct = pd_model * 100
        st.metric("Probability of Default (PD)", f"{pd_pct:.2f}%")
    with c2:
        st.metric("Loss Given Default (LGD)", f"{lgd_pred * 100:.1f}%")
    with c3:
        st.metric("Exposure at Default (EAD)", f"{valor_credito:,.0f}")
    with c4:
        el_pct = el / valor_credito * 100
        st.metric("Expected Loss", f"{el:,.2f}", delta=f"{el_pct:.2f}% of EAD")

    # Risk traffic light
    st.divider()
    if el_pct < 2:
        st.success(f"LOW risk — EL of {el_pct:.2f}% of the granted amount.")
    elif el_pct < 5:
        st.warning(
            f"MODERATE risk — EL of {el_pct:.2f}% of the granted amount. Review collateral."
        )
    else:
        st.error(
            f"HIGH risk — EL of {el_pct:.2f}% of the granted amount. Requires special approval."
        )

    # Method comparison
    with st.expander("Comparison: ML Model vs DGP Oracle"):
        diff_pct = (pd_model - pd_dgp) / (pd_dgp + 1e-6) * 100
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("PD — Contextual Model (ML)", f"{pd_model:.4f}")
        with mc2:
            st.metric("PD — DGP (oracle)", f"{pd_dgp:.4f}")
        with mc3:
            st.metric("Difference", f"{diff_pct:+.1f}%")
        st.caption(
            "The DGP is the synthetic ground truth. The ML model learns to approximate it from generated data. "
            "Differences reflect estimation variance — the model never sees the DGP directly."
        )

    # Calculation detail
    with st.expander("Calculation breakdown"):
        st.markdown(f"""
        **Formula:** EL = PD × LGD × EAD

        | Component | Value |
        |---|---|
        | PD — Contextual Model (LightGBM + Isotonic) | {pd_model:.6f} |
        | PD — DGP oracle (reference) | {pd_dgp:.6f} |
        | LGD (Beta regression) | {lgd_pred:.4f} |
        | EAD (credit amount) | {valor_credito:,.2f} |
        | **Expected Loss** | **{el:,.2f}** |

        *Product*: `{produto}` | *Tenor*: `{prazo}` months | *Collateral*: `{"Yes" if has_collateral else "No"}`
        """)
