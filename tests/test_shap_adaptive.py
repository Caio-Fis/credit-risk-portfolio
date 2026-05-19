"""Tests for `src.explain.shap_adaptive`.

Trains a tiny LightGBM on synthetic monthly data and exercises the three public
entry points (rebaselined monthly SHAP, per-risk-decile SHAP, incremental Ridge
surrogate) plus the heatmap helper. Kept lightweight so the suite still finishes
in seconds in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.explain.shap_adaptive import (
    IncrementalRidgeSurrogate,
    adaptive_shap_heatmap,
    rebaselined_shap_monthly,
    shap_by_risk_decile,
)


# ---------------------------------------------------------------------------
# Tiny synthetic dataset + LightGBM
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_dataset() -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    months = pd.date_range("2015-01-01", "2016-12-01", freq="MS")
    rows = []
    for m_idx, month in enumerate(months):
        n = 250
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)
        x4 = rng.normal(0, 1, n)
        shift = 0.0 if m_idx < len(months) // 2 else 0.8
        logit = -0.4 + 0.7 * x1 - 0.5 * x2 + 0.3 * x3 + shift
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
        for i in range(n):
            rows.append({
                "issue_d": month + pd.Timedelta(days=int(rng.integers(0, 27))),
                "x1": float(x1[i]),
                "x2": float(x2[i]),
                "x3": float(x3[i]),
                "x4": float(x4[i]),
                "target": int(y[i]),
            })
    df = pd.DataFrame(rows).sort_values("issue_d").reset_index(drop=True)
    return df, ["x1", "x2", "x3", "x4"]


@pytest.fixture(scope="module")
def fitted_model(synthetic_dataset):
    df, feature_cols = synthetic_dataset
    model = LGBMClassifier(
        n_estimators=30, num_leaves=7, learning_rate=0.1,
        min_child_samples=20, random_state=0, verbose=-1,
    )
    model.fit(df[feature_cols], df["target"])
    return model


# ---------------------------------------------------------------------------
# rebaselined_shap_monthly
# ---------------------------------------------------------------------------
def test_rebaselined_shap_monthly_returns_long_frame(synthetic_dataset, fitted_model):
    df, feature_cols = synthetic_dataset
    out = rebaselined_shap_monthly(
        fitted_model, df, feature_cols=feature_cols,
        background_window_months=3, explain_per_month=200, seed=0,
    )
    assert set(out.columns) == {"month", "feature", "mean_abs_shap"}
    assert (out["mean_abs_shap"] >= 0).all()
    # Every emitted (month, feature) pair must be present exactly once.
    grouped = out.groupby(["month", "feature"]).size()
    assert (grouped == 1).all()


def test_rebaselined_shap_monthly_warm_up_months_skipped(synthetic_dataset, fitted_model):
    """The first months don't have enough history to fill the background window,
    so they must be omitted instead of raising or emitting NaNs."""
    df, feature_cols = synthetic_dataset
    out = rebaselined_shap_monthly(
        fitted_model, df, feature_cols=feature_cols,
        background_window_months=4, explain_per_month=200, seed=0,
    )
    first_month_in_data = pd.to_datetime(df["issue_d"].min()).to_period("M")
    first_month_in_output = pd.Period(out["month"].min(), freq="M")
    assert first_month_in_output > first_month_in_data


def test_rebaselined_shap_monthly_empty_when_data_too_small(fitted_model):
    df_tiny = pd.DataFrame({
        "issue_d": pd.date_range("2015-01-01", periods=5, freq="D"),
        "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "x2": [0.0, 0.0, 0.0, 0.0, 0.0],
        "x3": [0.0, 0.0, 0.0, 0.0, 0.0],
        "x4": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    out = rebaselined_shap_monthly(
        fitted_model, df_tiny, feature_cols=["x1", "x2", "x3", "x4"],
        background_window_months=3, explain_per_month=100, seed=0,
    )
    # Not enough data for any month → empty frame with the right columns.
    assert out.empty
    assert set(out.columns) >= {"month", "feature", "mean_abs_shap"}


# ---------------------------------------------------------------------------
# shap_by_risk_decile
# ---------------------------------------------------------------------------
def test_shap_by_risk_decile_covers_all_deciles(synthetic_dataset, fitted_model):
    df, feature_cols = synthetic_dataset
    sample = df.sample(2000, random_state=0).copy()
    sample["p"] = fitted_model.predict_proba(sample[feature_cols])[:, 1]
    out = shap_by_risk_decile(
        fitted_model, sample, feature_cols=feature_cols,
        score_col="p", n_deciles=10, explain_per_decile=100, seed=0,
    )
    assert set(out.columns) == {"decile", "feature", "mean_abs_shap"}
    # Some deciles may be skipped if too small, but at least 5 must survive.
    assert out["decile"].nunique() >= 5
    assert (out["mean_abs_shap"] >= 0).all()


# ---------------------------------------------------------------------------
# IncrementalRidgeSurrogate
# ---------------------------------------------------------------------------
def test_incremental_ridge_surrogate_fit_monthly(synthetic_dataset, fitted_model):
    df, feature_cols = synthetic_dataset
    scored = df.copy()
    scored["p"] = fitted_model.predict_proba(df[feature_cols])[:, 1]

    surrogate = IncrementalRidgeSurrogate(window_months=3, alpha=1.0)
    coefs = surrogate.fit_monthly(scored, feature_cols=feature_cols, score_col="p", date_col="issue_d")
    assert "month" in coefs.columns
    assert set(feature_cols).issubset(coefs.columns)
    # Coefficients are finite numbers, not NaN.
    assert coefs[feature_cols].notna().all().all()


def test_incremental_ridge_surrogate_handles_categorical_via_freq_encoding(synthetic_dataset, fitted_model):
    """The Ridge surrogate frequency-encodes object/category columns so the
    surrogate can stay purely numeric. This regression test pins that behaviour."""
    df, feature_cols = synthetic_dataset
    scored = df.copy()
    scored["p"] = fitted_model.predict_proba(df[feature_cols])[:, 1]
    # Inject a categorical column the surrogate has to encode.
    rng = np.random.default_rng(0)
    scored["seg"] = rng.choice(["A", "B", "C"], len(scored))

    surrogate = IncrementalRidgeSurrogate(window_months=3, alpha=1.0)
    coefs = surrogate.fit_monthly(
        scored, feature_cols=feature_cols + ["seg"], score_col="p", date_col="issue_d",
    )
    assert "seg" in coefs.columns
    assert coefs["seg"].notna().all()


# ---------------------------------------------------------------------------
# adaptive_shap_heatmap
# ---------------------------------------------------------------------------
def test_adaptive_shap_heatmap_returns_figure(synthetic_dataset, fitted_model, tmp_path):
    df, feature_cols = synthetic_dataset
    monthly = rebaselined_shap_monthly(
        fitted_model, df, feature_cols=feature_cols,
        background_window_months=3, explain_per_month=200, seed=0,
    )
    fig = adaptive_shap_heatmap(monthly, top_k=3, save_path=tmp_path / "heatmap.png")
    assert fig is not None
    assert (tmp_path / "heatmap.png").exists()
