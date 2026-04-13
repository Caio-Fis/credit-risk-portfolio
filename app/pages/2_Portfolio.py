"""Dashboard de carteira com KPIs agregados e performance do modelo PD."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR
from src.contextual.data_generator import generate_dataset
from src.evaluate.metrics import auroc, brier_score, ks_stat
from src.models.expected_loss import el_by_segment

st.set_page_config(page_title="Dashboard de Portfólio", layout="wide")
st.title("Dashboard de Portfólio")
st.caption("Visão consolidada da carteira de crédito PJ.")


@st.cache_data
def load_portfolio():
    df = generate_dataset(n=3000, seed=42)
    df["pd_proba"] = df["pd_true"]
    df["lgd_pred"] = df["lgd_true"]
    df["AMT_CREDIT"] = df["ead"]
    df["expected_loss"] = df["el_true"]
    return df


df = load_portfolio()

# ---------------------------------------------------------------------------
# KPIs principais
# ---------------------------------------------------------------------------
st.subheader("KPIs da Carteira")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Contratos", f"{len(df):,}")
with c2:
    st.metric("EAD Total", f"R$ {df['ead'].sum() / 1e6:.1f}M")
with c3:
    st.metric("EL Total", f"R$ {df['expected_loss'].sum() / 1e3:.0f}K")
with c4:
    el_rate = df["expected_loss"].sum() / df["ead"].sum() * 100
    st.metric("EL Rate", f"{el_rate:.2f}%")
with c5:
    st.metric("PD Média", f"{df['pd_true'].mean():.2%}")

st.divider()

# ---------------------------------------------------------------------------
# Distribuição de risco
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribuição de PD")
    fig = px.histogram(
        df,
        x="pd_true",
        nbins=40,
        color="product_type",
        title="Distribuição de PD por Produto",
        labels={"pd_true": "PD", "count": "Contratos"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("EL por Produto")
    el_prod = el_by_segment(
        df, segment_col="product_type", el_col="expected_loss", ead_col="ead"
    )
    fig = px.bar(
        el_prod,
        x="product_type",
        y="el_rate_pct",
        color="product_type",
        title="Taxa de EL (%) por Produto",
        labels={"el_rate_pct": "EL Rate (%)", "product_type": "Produto"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# EL por prazo
# ---------------------------------------------------------------------------
st.subheader("Expected Loss por Prazo")
el_tenor = el_by_segment(
    df, segment_col="tenor_months", el_col="expected_loss", ead_col="ead"
)
fig = px.line(
    el_tenor,
    x="tenor_months",
    y="el_rate_pct",
    markers=True,
    title="EL Rate (%) conforme Prazo do Contrato",
    labels={"tenor_months": "Prazo (meses)", "el_rate_pct": "EL Rate (%)"},
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela de contratos de maior risco
# ---------------------------------------------------------------------------
st.subheader("Top 20 Contratos de Maior Expected Loss")
top20 = (
    df[
        [
            "client_id",
            "product_type",
            "tenor_months",
            "pd_true",
            "lgd_true",
            "ead",
            "expected_loss",
        ]
    ]
    .sort_values("expected_loss", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top20.columns = ["Cliente", "Produto", "Prazo (m)", "PD", "LGD", "EAD (R$)", "EL (R$)"]
st.dataframe(
    top20.style.format(
        {
            "PD": "{:.3f}",
            "LGD": "{:.3f}",
            "EAD (R$)": "R$ {:,.0f}",
            "EL (R$)": "R$ {:,.0f}",
        }
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Performance do modelo PD (real — OOS hold-out)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Performance do Modelo PD — Out-of-Sample")
st.caption(
    "Métricas calculadas no hold-out 20% que nunca foi visto durante o treino. "
    "Fonte: modelo LightGBM + Platt treinado na feature store Home Credit."
)

OOS_PATH = PROCESSED_DIR / "oos_predictions.parquet"

if OOS_PATH.exists():
    oos = pd.read_parquet(OOS_PATH)
    y_true = oos["y_true"].values
    y_pred = oos["y_pred"].values

    auroc_val = auroc(y_true, y_pred)
    ks_val = ks_stat(y_true, y_pred)
    brier_val = brier_score(y_true, y_pred)

    m1, m2, m3 = st.columns(3)
    with m1:
        color = "normal" if auroc_val >= 0.78 else "inverse"
        st.metric(
            "AUROC (OOS)",
            f"{auroc_val:.4f}",
            delta="≥ 0.78 ✓" if auroc_val >= 0.78 else f"meta 0.78 — gap {0.78 - auroc_val:.4f}",
            delta_color=color,
        )
    with m2:
        color = "normal" if ks_val >= 0.35 else "inverse"
        st.metric(
            "KS Stat (OOS)",
            f"{ks_val:.4f}",
            delta="≥ 0.35 ✓" if ks_val >= 0.35 else f"meta 0.35 — gap {0.35 - ks_val:.4f}",
            delta_color=color,
        )
    with m3:
        color = "normal" if brier_val <= 0.15 else "inverse"
        st.metric(
            "Brier Score (OOS)",
            f"{brier_val:.4f}",
            delta="≤ 0.15 ✓" if brier_val <= 0.15 else f"meta 0.15 — gap {brier_val - 0.15:.4f}",
            delta_color=color,
        )

    # Distribuição de scores por classe
    oos_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    oos_df["Classe"] = oos_df["y_true"].map({0: "Adimplente", 1: "Default"})
    fig_dist = px.histogram(
        oos_df,
        x="y_pred",
        color="Classe",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        title="Distribuição de Score PD — OOS Hold-out",
        labels={"y_pred": "Score PD (probabilidade)", "count": "Contratos"},
        color_discrete_map={"Adimplente": "steelblue", "Default": "coral"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info(
        "Arquivo OOS não encontrado. Execute `make train` para gerar predições hold-out."
    )
