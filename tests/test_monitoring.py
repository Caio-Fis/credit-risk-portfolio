"""Testes para os módulos de monitoramento e early warning."""

import numpy as np
import pandas as pd

from src.early_warning.behavioral_signals import (
    flag_volume_drop,
    simulate_behavioral_data,
)
from src.early_warning.score_trajectory import compute_score_trajectory, flag_score_drop
from src.monitoring.drift_detector import detect_drift
from src.monitoring.psi import compute_psi, psi_all_features
from src.monitoring.vintage_analysis import build_vintage_matrix, simulate_vintage_data

# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def test_compute_psi_identical_distributions():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    psi = compute_psi(x, x)
    assert psi < 0.01, (
        f"PSI de distribuições idênticas deveria ser ~0, mas foi {psi:.4f}"
    )


def test_compute_psi_different_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 2000)
    curr = rng.normal(2, 1, 1000)  # shift claro
    psi = compute_psi(ref, curr)
    assert psi > 0.20, f"PSI com shift deveria ser > 0.20, mas foi {psi:.4f}"


def test_compute_psi_non_negative():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, 500)
    curr = rng.uniform(0.3, 1, 300)
    psi = compute_psi(ref, curr)
    assert psi >= 0.0


def test_psi_all_features_returns_dataframe(psi_dataframes):
    ref, curr = psi_dataframes
    result = psi_all_features(ref, curr)
    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert "psi" in result.columns
    assert "status" in result.columns


def test_psi_all_features_status_values(psi_dataframes):
    ref, curr = psi_dataframes
    result = psi_all_features(ref, curr)
    valid_statuses = {"stable", "attention", "drift"}
    assert set(result["status"].unique()).issubset(valid_statuses)


def test_psi_detects_drift_in_feature_a(psi_dataframes):
    ref, curr = psi_dataframes
    result = psi_all_features(ref, curr)
    feature_a_row = result[result["feature"] == "feature_a"]
    assert not feature_a_row.empty
    assert feature_a_row.iloc[0]["status"] == "drift", (
        "feature_a deveria estar em drift"
    )


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


def test_detect_drift_classifies_correctly():
    psi_series = pd.Series(
        {"feat_stable": 0.05, "feat_attention": 0.15, "feat_drift": 0.25}
    )
    statuses = detect_drift(psi_series)
    assert statuses["feat_stable"] == "stable"
    assert statuses["feat_attention"] == "attention"
    assert statuses["feat_drift"] == "drift"


# ---------------------------------------------------------------------------
# Vintage analysis
# ---------------------------------------------------------------------------


def test_simulate_vintage_data_shape():
    df = simulate_vintage_data(n_contracts=1000, n_vintages=6, max_maturity=12, seed=0)
    assert "vintage" in df.columns
    assert "maturity_month" in df.columns
    assert "is_default" in df.columns
    assert len(df) > 0


def test_build_vintage_matrix_shape():
    df = simulate_vintage_data(n_contracts=2000, n_vintages=6, max_maturity=12, seed=0)
    matrix = build_vintage_matrix(df)
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0


def test_build_vintage_matrix_values_in_range():
    df = simulate_vintage_data(n_contracts=2000, n_vintages=6, max_maturity=12, seed=0)
    matrix = build_vintage_matrix(df)
    assert (matrix.fillna(0) >= 0).all().all()
    assert (matrix.fillna(0) <= 1).all().all()


# ---------------------------------------------------------------------------
# Score trajectory
# ---------------------------------------------------------------------------


def test_flag_score_drop_returns_alerts():
    df = simulate_behavioral_data(n_entities=100, n_months=12, seed=42)
    trajectory = compute_score_trajectory(df)
    alerts = flag_score_drop(trajectory, threshold=50)
    assert isinstance(alerts, pd.DataFrame)
    if not alerts.empty:
        assert (alerts["score_drop"] >= 50).all()


def test_flag_score_drop_alert_level():
    df = simulate_behavioral_data(n_entities=200, n_months=12, seed=0)
    trajectory = compute_score_trajectory(df)
    alerts = flag_score_drop(trajectory, threshold=30)
    if not alerts.empty:
        assert "alert_level" in alerts.columns
        assert set(alerts["alert_level"].unique()).issubset({"attention", "critical"})


# ---------------------------------------------------------------------------
# Behavioral signals
# ---------------------------------------------------------------------------


def test_flag_volume_drop_returns_dataframe():
    df = simulate_behavioral_data(n_entities=100, n_months=12, seed=0)
    result = flag_volume_drop(df, threshold_pct=0.20)
    assert isinstance(result, pd.DataFrame)


def test_simulate_behavioral_data_shape():
    df = simulate_behavioral_data(n_entities=50, n_months=6, seed=1)
    assert len(df) == 50 * 6
    assert "pd_score" in df.columns
    assert "monthly_payment_count" in df.columns
