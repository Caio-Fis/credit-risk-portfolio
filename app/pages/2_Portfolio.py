"""Portfolio dashboard with aggregated KPIs and PD model performance."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR
from src.contextual.data_generator import generate_dataset
from src.evaluate.metrics import (
    auroc,
    binomial_test_by_bucket,
    brier_score,
    gini,
    hosmer_lemeshow_test,
    ks_stat,
)
from src.models.expected_loss import el_by_segment

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("Portfolio Dashboard")
st.caption("Consolidated view of the corporate credit portfolio.")


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
# Main KPIs
# ---------------------------------------------------------------------------
st.subheader("Portfolio KPIs")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Contracts", f"{len(df):,}")
with c2:
    st.metric("Total EAD", f"{df['ead'].sum() / 1e6:.1f}M")
with c3:
    st.metric("Total EL", f"{df['expected_loss'].sum() / 1e3:.0f}K")
with c4:
    el_rate = df["expected_loss"].sum() / df["ead"].sum() * 100
    st.metric("EL Rate", f"{el_rate:.2f}%")
with c5:
    st.metric("Mean PD", f"{df['pd_true'].mean():.2%}")

st.divider()

# ---------------------------------------------------------------------------
# Risk distribution
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("PD Distribution")
    fig = px.histogram(
        df,
        x="pd_true",
        nbins=40,
        color="product_type",
        title="PD Distribution by Product",
        labels={"pd_true": "PD", "count": "Contracts"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("EL by Product")
    el_prod = el_by_segment(
        df, segment_col="product_type", el_col="expected_loss", ead_col="ead"
    )
    fig = px.bar(
        el_prod,
        x="product_type",
        y="el_rate_pct",
        color="product_type",
        title="EL Rate (%) by Product",
        labels={"el_rate_pct": "EL Rate (%)", "product_type": "Product"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# EL by tenor
# ---------------------------------------------------------------------------
st.subheader("Expected Loss by Tenor")
el_tenor = el_by_segment(
    df, segment_col="tenor_months", el_col="expected_loss", ead_col="ead"
).sort_values("tenor_months")
fig = px.line(
    el_tenor,
    x="tenor_months",
    y="el_rate_pct",
    markers=True,
    title="EL Rate (%) by Contract Tenor",
    labels={"tenor_months": "Tenor (months)", "el_rate_pct": "EL Rate (%)"},
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Top-risk contracts table
# ---------------------------------------------------------------------------
st.subheader("Top 20 Contracts by Expected Loss")
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
top20.columns = ["Client", "Product", "Tenor (m)", "PD", "LGD", "EAD", "EL"]
st.dataframe(
    top20.style.format(
        {
            "PD": "{:.3f}",
            "LGD": "{:.3f}",
            "EAD": "{:,.0f}",
            "EL": "{:,.0f}",
        }
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# PD model performance (real — OOS hold-out)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("PD Model Performance — Out-of-Sample")
st.caption(
    "Metrics calculated on the 20% hold-out never seen during training. "
    "Source: LightGBM + Isotonic model trained on the Home Credit feature store."
)

OOS_PATH = PROCESSED_DIR / "oos_predictions.parquet"

if OOS_PATH.exists():
    oos = pd.read_parquet(OOS_PATH)
    y_true = oos["y_true"].values
    y_pred = oos["y_pred"].values

    auroc_val = auroc(y_true, y_pred)
    gini_val = gini(y_true, y_pred)
    ks_val = ks_stat(y_true, y_pred)
    brier_val = brier_score(y_true, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        color = "normal" if auroc_val >= 0.78 else "inverse"
        st.metric(
            "AUROC (OOS)",
            f"{auroc_val:.4f}",
            delta="≥ 0.78 ✓" if auroc_val >= 0.78 else f"target 0.78 — gap {0.78 - auroc_val:.4f}",
            delta_color=color,
        )
    with m2:
        st.metric(
            "Gini (OOS)",
            f"{gini_val:.4f}",
            delta="credit industry standard",
            delta_color="off",
        )
    with m3:
        color = "normal" if ks_val >= 0.35 else "inverse"
        st.metric(
            "KS Stat (OOS)",
            f"{ks_val:.4f}",
            delta="≥ 0.35 ✓" if ks_val >= 0.35 else f"target 0.35 — gap {0.35 - ks_val:.4f}",
            delta_color=color,
        )
    with m4:
        color = "normal" if brier_val <= 0.15 else "inverse"
        st.metric(
            "Brier Score (OOS)",
            f"{brier_val:.4f}",
            delta="≤ 0.15 ✓" if brier_val <= 0.15 else f"target 0.15 — gap {brier_val - 0.15:.4f}",
            delta_color=color,
        )

    # Score distribution by class
    oos_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    oos_df["Class"] = oos_df["y_true"].map({0: "Non-default", 1: "Default"})
    fig_dist = px.histogram(
        oos_df,
        x="y_pred",
        color="Class",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        title="PD Score Distribution — OOS Hold-out",
        labels={"y_pred": "PD Score (probability)", "count": "Contracts"},
        color_discrete_map={"Non-default": "steelblue", "Default": "coral"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Statistical calibration tests
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Statistical Calibration Validation")
    st.caption(
        "Validates whether predicted PDs correspond to observed default rates — "
        "the foundation for trusting EL = PD × LGD × EAD in currency units."
    )

    tab_hl, tab_bucket = st.tabs(["Hosmer-Lemeshow", "Binomial Test by Bucket (Basel)"])

    with tab_hl:
        hl_stat, hl_pvalue, hl_table = hosmer_lemeshow_test(y_true, y_pred)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("H-L Statistic", f"{hl_stat:.2f}")
        with c2:
            color = "normal" if hl_pvalue > 0.05 else "inverse"
            st.metric(
                "p-value",
                f"{hl_pvalue:.4f}",
                delta="calibration OK (p > 0.05)" if hl_pvalue > 0.05 else "miscalibration detected",
                delta_color=color,
            )
        st.caption(
            "H0: model is well calibrated. Failing to reject (p > 0.05) is the desired result. "
            "A high p-value means observed defaults are consistent with predicted PDs."
        )
        display_cols = ["faixa_pd", "contratos", "observed", "expected", "default_rate_obs", "pd_mean", "ratio_obs_exp"]
        available_cols = [c for c in display_cols if c in hl_table.columns]
        st.dataframe(
            hl_table[available_cols].style.format({
                "default_rate_obs": "{:.3f}",
                "pd_mean": "{:.3f}",
                "ratio_obs_exp": "{:.2f}",
                "expected": "{:.1f}",
            }),
            use_container_width=True,
        )

    with tab_bucket:
        bucket_result = binomial_test_by_bucket(y_true, y_pred)

        n_verde = (bucket_result["semaforo"] == "Verde").sum()
        n_amarelo = (bucket_result["semaforo"] == "Amarelo").sum()
        n_vermelho = (bucket_result["semaforo"] == "Vermelho").sum()

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Green (p > 0.05)", n_verde)
        with b2:
            st.metric("Yellow (0.01–0.05)", n_amarelo)
        with b3:
            st.metric("Red (p ≤ 0.01)", n_vermelho)

        st.caption(
            "One-sided binomial test: for each PD bucket, checks whether the number "
            "of observed defaults is consistent with the mean predicted PD. "
            "Basel III standard (BCB Resolution 4.557)."
        )
        st.dataframe(
            bucket_result.style.format({
                "pd_media_predita": "{:.3f}",
                "taxa_obs": "{:.3f}",
                "p_value": "{:.4f}",
            }).applymap(
                lambda v: "background-color: #d4edda" if v == "Verde"
                else ("background-color: #fff3cd" if v == "Amarelo"
                else "background-color: #f8d7da"),
                subset=["semaforo"],
            ),
            use_container_width=True,
        )
else:
    st.info(
        "OOS file not found. Run `make train` to generate hold-out predictions."
    )
