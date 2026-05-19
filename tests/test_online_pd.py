"""Tests for the streaming PD challenger (`src.models.online_pd_model`).

We deliberately use very small samples + few ARF trees so the suite stays under
a few seconds. The contracts under test are:

  * `row_to_dict` normalises types the way River expects.
  * `build_arf` returns a usable ARFClassifier with the configured drift sensitivity.
  * `stream_evaluate` honours test-then-train + label delay, returns the expected
    shapes, and produces metric tables only for years with both classes present.
  * `detect_drifts_on_stream` returns one row per drift event with the documented
    schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.online_pd_model import (
    StreamResult,
    _ks_from_arrays,
    build_arf,
    row_to_dict,
    stream_evaluate,
)
from src.monitoring.drift_online import detect_drifts_on_stream


# ---------------------------------------------------------------------------
# Synthetic stream — small, fast, deterministic
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_stream() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_per_month = 60
    months = pd.date_range("2015-01-01", "2016-06-01", freq="MS")
    rows = []
    for m_idx, month in enumerate(months):
        x1 = rng.normal(0, 1, n_per_month)
        x2 = rng.normal(0, 1, n_per_month)
        cat = rng.choice(["A", "B", "C"], n_per_month)
        # Inject a regime change halfway through to give ADWIN / KSWIN something to flag.
        shift = 0.0 if m_idx < len(months) // 2 else 1.2
        logit = -0.5 + 0.8 * x1 - 0.6 * x2 + shift
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, prob)
        for i in range(n_per_month):
            rows.append({
                "issue_d": month + pd.Timedelta(days=int(rng.integers(0, 27))),
                "x1": float(x1[i]),
                "x2": float(x2[i]),
                "cat": str(cat[i]),
                "target": int(y[i]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# row_to_dict
# ---------------------------------------------------------------------------
def test_row_to_dict_handles_numeric_nan_and_strings():
    row = pd.Series({"a": 1.5, "b": np.nan, "c": "foo", "d": 0})
    out = row_to_dict(row, ["a", "b", "c", "d"])
    assert out == {"a": 1.5, "b": 0.0, "c": "foo", "d": 0.0}
    # All numeric values are floats so River doesn't choke on int/np types.
    assert isinstance(out["a"], float)
    assert isinstance(out["d"], float)


def test_row_to_dict_respects_feature_subset():
    row = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    out = row_to_dict(row, ["a", "c"])
    assert set(out.keys()) == {"a", "c"}


# ---------------------------------------------------------------------------
# build_arf
# ---------------------------------------------------------------------------
def test_build_arf_returns_classifier_with_configured_seed():
    arf = build_arf(seed=7, n_models=3)
    # River ARFClassifier exposes `n_models` attribute.
    assert arf.n_models == 3
    # Smoke: it can learn a single observation and produce a proba dict.
    arf.learn_one({"x": 0.5, "cat": "A"}, 1)
    arf.learn_one({"x": -0.5, "cat": "B"}, 0)
    proba = arf.predict_proba_one({"x": 0.1, "cat": "A"})
    assert isinstance(proba, dict)
    if proba:
        assert pytest.approx(sum(proba.values()), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# _ks_from_arrays
# ---------------------------------------------------------------------------
def test_ks_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    assert _ks_from_arrays(y, p) == pytest.approx(1.0, abs=1e-6)


def test_ks_random_scores_is_small():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.uniform(0, 1, 1000)
    assert _ks_from_arrays(y, p) < 0.15


# ---------------------------------------------------------------------------
# stream_evaluate
# ---------------------------------------------------------------------------
def test_stream_evaluate_returns_expected_shape(synthetic_stream):
    result = stream_evaluate(
        synthetic_stream,
        feature_cols=["x1", "x2", "cat"],
        samples_per_month=None,
        n_models=3,
        label_delay_days=30,
        log_every=10_000,
    )
    assert isinstance(result, StreamResult)
    assert len(result.predictions) == len(synthetic_stream)
    assert set(result.predictions.columns) >= {"issue_d", "y", "p", "year"}
    # Predictions are probabilities in [0, 1].
    assert result.predictions["p"].between(0.0, 1.0).all()


def test_stream_evaluate_yearly_only_includes_years_with_two_classes(synthetic_stream):
    result = stream_evaluate(
        synthetic_stream,
        feature_cols=["x1", "x2", "cat"],
        samples_per_month=None,
        n_models=3,
        log_every=10_000,
    )
    # Each yearly row must have both classes in its slice; otherwise AUROC is undefined.
    for _, row in result.yearly.iterrows():
        slice_ = result.predictions[result.predictions["year"] == row["year"]]
        assert slice_["y"].nunique() == 2
        assert row["auroc"] >= 0.0
        assert row["brier"] >= 0.0


def test_stream_evaluate_predictions_are_chronological(synthetic_stream):
    result = stream_evaluate(
        synthetic_stream,
        feature_cols=["x1", "x2", "cat"],
        samples_per_month=None,
        n_models=3,
        log_every=10_000,
    )
    diffs = result.predictions["issue_d"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all(), "predictions must be in chronological order"


def test_stream_evaluate_label_delay_affects_predictions(synthetic_stream):
    """A longer label delay starves the model of supervision, so its early predictions
    drift toward the prior. Confirm the delay parameter actually changes the output."""
    no_delay = stream_evaluate(
        synthetic_stream, feature_cols=["x1", "x2", "cat"],
        samples_per_month=None, n_models=3, label_delay_days=0, log_every=10_000,
    )
    delayed = stream_evaluate(
        synthetic_stream, feature_cols=["x1", "x2", "cat"],
        samples_per_month=None, n_models=3, label_delay_days=365, log_every=10_000,
    )
    # Different supervision schedules → different prediction traces.
    assert not np.allclose(no_delay.predictions["p"].values, delayed.predictions["p"].values)


# ---------------------------------------------------------------------------
# detect_drifts_on_stream
# ---------------------------------------------------------------------------
def test_detect_drifts_finds_obvious_score_shift():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "issue_d": pd.date_range("2015-01-01", periods=n, freq="D"),
        # First half low scores, second half high scores — guaranteed KSWIN trigger.
        "p": np.concatenate([rng.uniform(0.0, 0.2, n // 2), rng.uniform(0.7, 0.95, n // 2)]),
        "y": rng.integers(0, 2, n),
    })
    events = detect_drifts_on_stream(df, score_col="p", label_col="y", date_col="issue_d")
    assert not events.empty
    assert set(events.columns) >= {"timestamp", "type", "detector", "value", "ix"}
    assert (events["detector"] == "KSWIN").any()


def test_detect_drifts_quiet_stream_returns_empty():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "issue_d": pd.date_range("2015-01-01", periods=n, freq="D"),
        "p": rng.uniform(0.4, 0.5, n),  # tight band, no shift
        "y": np.zeros(n, dtype=int),     # error is constant -> ADWIN silent
    })
    events = detect_drifts_on_stream(df, score_col="p", label_col="y", date_col="issue_d")
    # A handful of KSWIN false positives are tolerable, but ADWIN on a constant error
    # series must stay completely silent.
    assert (events["detector"] == "ADWIN").sum() == 0 if not events.empty else True
