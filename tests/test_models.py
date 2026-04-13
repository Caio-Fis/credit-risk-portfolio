"""Testes para os módulos de modelagem e Expected Loss."""

import numpy as np
import pandas as pd
import pytest

from src.contextual.interaction_model import train_contextual
from src.evaluate.metrics import auroc, brier_score, ks_stat
from src.models.expected_loss import compute_el, el_by_segment, el_summary

# ---------------------------------------------------------------------------
# Expected Loss
# ---------------------------------------------------------------------------


def test_compute_el_basic():
    pd_ = np.array([0.05, 0.10, 0.20])
    lgd = np.array([0.50, 0.60, 0.70])
    ead = np.array([100_000, 200_000, 50_000])
    el = compute_el(pd_, lgd, ead)
    expected = pd_ * lgd * ead
    np.testing.assert_allclose(el.values, expected, rtol=1e-6)


def test_compute_el_all_zero_pd():
    pd_ = np.zeros(5)
    lgd = np.ones(5) * 0.5
    ead = np.ones(5) * 100_000
    el = compute_el(pd_, lgd, ead)
    assert (el == 0).all()


def test_compute_el_shape_mismatch():
    with pytest.raises(ValueError, match="mesmo comprimento"):
        compute_el(np.array([0.1, 0.2]), np.array([0.5]), np.array([100_000, 200_000]))


def test_el_summary_returns_dataframe(synthetic_df):
    df = synthetic_df.copy()
    df["expected_loss"] = df["el_true"]
    df["AMT_CREDIT"] = df["ead"]
    summary = el_summary(df)
    assert isinstance(summary, pd.DataFrame)
    assert "total_el_R$" in summary.columns
    assert "el_rate_%" in summary.columns


def test_el_by_segment(synthetic_df):
    df = synthetic_df.copy()
    df["expected_loss"] = df["el_true"]
    df["AMT_CREDIT"] = df["ead"]
    result = el_by_segment(df, segment_col="product_type")
    assert len(result) == df["product_type"].nunique()
    assert "el_rate_pct" in result.columns


# ---------------------------------------------------------------------------
# Métricas de avaliação
# ---------------------------------------------------------------------------


def test_auroc_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.uniform(0, 1, 500)
    score = auroc(y_true, y_proba)
    assert 0.0 <= score <= 1.0


def test_ks_stat_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.uniform(0, 1, 500)
    ks = ks_stat(y_true, y_proba)
    assert 0.0 <= ks <= 1.0


def test_brier_score_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.uniform(0, 1, 500)
    bs = brier_score(y_true, y_proba)
    assert 0.0 <= bs <= 1.0


def test_auroc_perfect_model():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y_true, y_proba) == 1.0


# ---------------------------------------------------------------------------
# Modelo contextual (smoke test)
# ---------------------------------------------------------------------------


def test_contextual_model_trains(synthetic_df):
    model, feature_names = train_contextual(synthetic_df, seed=0)
    assert model is not None
    assert len(feature_names) > 0


def test_contextual_model_predicts(synthetic_df):
    from src.contextual.context_features import (
        add_context_features,
        create_interaction_features,
        encode_product,
    )

    model, feature_names = train_contextual(synthetic_df, seed=0)
    df_feat = add_context_features(synthetic_df.copy())
    df_feat = encode_product(df_feat, drop_first=True)
    df_feat = create_interaction_features(df_feat)
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    X = df_feat[feature_names].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == len(synthetic_df)
    assert ((proba >= 0) & (proba <= 1)).all()
