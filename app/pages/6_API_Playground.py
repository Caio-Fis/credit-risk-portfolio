"""API Playground — interactive demo that consumes the FastAPI service.

The Streamlit lab and the production API live independently. This page
proves they integrate: every action here is an HTTP request to the
running FastAPI service (default ``http://localhost:7860``), so the
panels react to the same model + drift state the API serves to any
other client.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="API Playground", page_icon="🔌", layout="wide")
st.title("FastAPI Playground")
st.caption(
    "Every panel below is a live HTTP call to the production API. "
    "The Streamlit app is just a client."
)

# ---------------------------------------------------------------------------
# Sidebar — connection settings
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:7860"

with st.sidebar:
    st.header("API connection")
    base_url = st.text_input("Base URL", DEFAULT_URL, help="Point at your local container or HF Space.")
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            r.raise_for_status()
            st.success(f"OK · {r.json()}")
        except Exception as e:
            st.error(f"Unreachable · {e}")
    st.markdown(f"[Open Swagger /docs ↗]({base_url}/docs)")
    st.markdown(f"[Open ReDoc /redoc ↗]({base_url}/redoc)")
    st.markdown(f"[Open Prometheus /metrics ↗]({base_url}/metrics)")


def _post(path: str, payload: dict, timeout: int = 15) -> requests.Response:
    return requests.post(f"{base_url}{path}", json=payload, timeout=timeout)


def _get(path: str, timeout: int = 10) -> requests.Response:
    return requests.get(f"{base_url}{path}", timeout=timeout)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_explain, tab_monitor, tab_recal = st.tabs(
    ["Predict", "Explain", "Drift & calibration", "Recalibrate"]
)

# ---------- Predict ----------
with tab_predict:
    st.subheader("POST /v1/predict")
    col_l, col_r = st.columns(2)
    with col_l:
        revenue = st.number_input("Annual revenue ($)", value=65000.0, min_value=0.0, step=1000.0)
        dti = st.number_input("DTI ratio", value=18.5, min_value=0.0, max_value=999.0, step=0.5)
        loan_amnt = st.number_input("Loan amount ($)", value=15000.0, min_value=500.0, max_value=40_000.0, step=500.0)
        fico = st.number_input("FICO score", value=720, min_value=300, max_value=850)
        experience_c = st.selectbox("Experience class", [1, 0], index=0)
        emp_length = st.slider("Employment length (years)", 0.0, 10.0, 5.0, 0.5)
    with col_r:
        purpose = st.selectbox(
            "Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "other",
             "major_purchase", "medical", "small_business", "car"],
        )
        ownership = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        addr_state = st.text_input("State (2-letter)", "CA").upper()[:2]
        zip_code = st.text_input("ZIP (first 3)", "900xx")
        issue_d = st.date_input("Issue date (for macro lookup)", value=pd.Timestamp("2017-06-01").date())

    if st.button("Predict", type="primary"):
        payload = {
            "revenue": revenue, "dti_n": dti, "loan_amnt": loan_amnt, "fico_n": fico,
            "experience_c": experience_c, "emp_length": emp_length, "purpose": purpose,
            "home_ownership_n": ownership, "addr_state": addr_state, "zip_code": zip_code,
            "issue_d": str(issue_d),
        }
        st.session_state["last_payload"] = payload
        t0 = time.perf_counter()
        try:
            r = _post("/v1/predict", payload)
            r.raise_for_status()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            elapsed = (time.perf_counter() - t0) * 1000
            data = r.json()
            mc, ms, mb = st.columns(3)
            mc.metric("Calibrated PD", f"{data['pd_calibrated']:.2%}")
            ms.metric("Score / 1000", str(data["score_0_1000"]))
            mb.metric("Risk band", data["risk_band"])
            st.caption(f"Latency end-to-end: {elapsed:.1f} ms · request_id echo in headers · model `{data['model_version']}`")
            with st.expander("Macro snapshot merged server-side", expanded=False):
                st.json(data["macro_snapshot"])
            with st.expander("Raw response payload", expanded=False):
                st.code(json.dumps(data, indent=2), language="json")

# Persist last predict payload across tabs
if "last_payload" not in st.session_state:
    st.session_state["last_payload"] = None

# ---------- Explain ----------
with tab_explain:
    st.subheader("POST /v1/explain")
    st.caption("Reuses the inputs from the Predict tab — run a Predict first.")
    payload_explain = st.session_state.get("last_payload")
    if payload_explain is None:
        st.info("No payload yet — fill the form in the Predict tab and click Predict.")
    if st.button("Explain the last loan", disabled=payload_explain is None):
        try:
            r = _post("/v1/explain", payload_explain)
            r.raise_for_status()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            data = r.json()
            st.metric("Base value (logit)", f"{data['base_value']:.3f}")
            df_top = pd.DataFrame(data["top_drivers"])
            st.markdown("**Top 5 drivers**")
            st.dataframe(df_top, hide_index=True)

            df_all = pd.DataFrame(data["contributions"]).sort_values("shap_value")
            colors = ["#c0392b" if v > 0 else "#27ae60" for v in df_all["shap_value"]]
            fig = go.Figure(
                go.Bar(
                    x=df_all["shap_value"], y=df_all["feature"], orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in df_all["shap_value"]],
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Per-feature SHAP contribution (logit)",
                xaxis_title="Contribution to log-odds", yaxis_title="",
                height=520, margin=dict(l=120, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- Drift & calibration ----------
with tab_monitor:
    st.subheader("Live drift state · /v1/monitor/drift/live")
    if st.button("Refresh drift state"):
        try:
            live = _get("/v1/monitor/drift/live").json()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Predictions seen", live["samples_seen"])
            c2.metric("Drift events (total)", live["events_total"])
            c3.metric("Started at", str(live["started_at"])[:19])
            st.markdown("**Events by detector**")
            st.write(live["events_by_detector"])
            if live["last_psi"]:
                psi_df = pd.DataFrame(
                    [{"feature": k, "psi": v} for k, v in live["last_psi"].items()]
                ).sort_values("psi", ascending=False)
                psi_df["status"] = psi_df["psi"].apply(
                    lambda p: "drift" if p > 0.20 else ("attention" if p > 0.10 else "stable")
                )
                st.dataframe(psi_df, hide_index=True)
            if live["events_recent"]:
                st.markdown("**Recent events (last 20)**")
                st.dataframe(pd.DataFrame(live["events_recent"]), hide_index=True)

    st.subheader("Historical drift timeline · /v1/monitor/drift")
    if st.button("Load offline timeline"):
        try:
            hist = _get("/v1/monitor/drift").json()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            yr = pd.Series(hist["by_year"]).sort_index()
            yr.index = yr.index.astype(int)
            fig = go.Figure(go.Bar(x=yr.index, y=yr.values, marker_color="firebrick"))
            fig.update_layout(
                title="Drift events per year (ARF stream replay)",
                xaxis_title="Year", yaxis_title="Drift events",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Source: {hist['source']}  ·  total events: {hist['total_events']}")

    st.subheader("Calibration summary · /v1/monitor/calibration")
    if st.button("Load calibration metrics"):
        try:
            cal = _get("/v1/monitor/calibration").json()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            st.write(cal["summary"])
            yearly = pd.DataFrame(cal["yearly"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["brier_static"],
                                     name="Static", mode="lines+markers", line=dict(color="firebrick")))
            fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["brier_sliding"],
                                     name="Sliding (12mo, 90d delay)", mode="lines+markers", line=dict(color="seagreen")))
            fig.update_layout(title="Brier score over time", xaxis_title="Year", yaxis_title="Brier", height=380)
            st.plotly_chart(fig, use_container_width=True)

# ---------- Recalibrate ----------
with tab_recal:
    st.subheader("POST /v1/monitor/recalibrate")
    st.caption(
        "Manually trigger a sliding-window isotonic refit. The endpoint returns 202 immediately; "
        "the work runs as a FastAPI BackgroundTask."
    )
    trig = st.selectbox("Trigger", ["manual", "scheduled", "drift"])
    if st.button("Trigger recalibration"):
        try:
            r = _post("/v1/monitor/recalibrate", {"trigger": trig})
            r.raise_for_status()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            st.success(f"Accepted · job {r.json()['job_id']}")
            st.json(r.json())
