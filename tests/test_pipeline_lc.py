"""Smoke tests for the LendingClub v2 pipeline (`make pipeline-lc`).

The full pipeline downloads a 167 MB CSV from Zenodo and is not suitable for
CI. Instead, we exercise `src.features.lendingclub_features.transform` on a
synthetic DataFrame that mirrors the documented Zenodo schema, using the
committed `data/processed/macro_features.parquet` so no network call is made.

A schema break (e.g. a column rename in `transform`) or a regression in the
macro merge will surface here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MACRO_CACHE_PATH
from src.features.lendingclub_features import transform


@pytest.fixture(scope="module")
def synthetic_lc_frame() -> pd.DataFrame:
    """A 200-row LendingClub-shaped frame spanning 2007-2018.

    Mirrors the 15 raw columns from `data/schemas/lendingclub.json`.
    """
    rng = np.random.default_rng(0)
    months = pd.date_range("2007-06-01", "2018-12-01", freq="MS")
    n = 200
    issue_dates = rng.choice(months, n)
    return pd.DataFrame({
        "id": np.arange(n),
        "issue_d": pd.to_datetime(issue_dates).strftime("%b-%Y"),
        "revenue": rng.uniform(20_000, 250_000, n),
        "dti_n": rng.uniform(0, 40, n),
        "loan_amnt": rng.integers(1_000, 40_000, n),
        "fico_n": rng.uniform(600, 800, n),
        "experience_c": np.ones(n, dtype=int),  # constant — should be dropped
        "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years", "n/a"], n),
        "purpose": rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "car"], n,
        ),
        "home_ownership_n": rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], n),
        "addr_state": rng.choice(["CA", "NY", "TX", "FL"], n),
        "zip_code": rng.choice(["900xx", "100xx", "770xx", "330xx"], n),
        "Default": rng.integers(0, 2, n),
        "title": ["loan"] * n,           # free-text, dropped
        "desc": [None] * n,              # dropped
    })


# ---------------------------------------------------------------------------
# Schema contract — transform()
# ---------------------------------------------------------------------------
def test_transform_renames_default_to_target(synthetic_lc_frame):
    out = transform(synthetic_lc_frame, add_macro=False)
    assert "target" in out.columns
    assert "Default" not in out.columns
    assert out["target"].dtype.name == "int8"
    assert set(out["target"].unique()).issubset({0, 1})


def test_transform_parses_issue_d_to_datetime(synthetic_lc_frame):
    out = transform(synthetic_lc_frame, add_macro=False)
    assert pd.api.types.is_datetime64_any_dtype(out["issue_d"])
    assert out["issue_d"].isna().sum() == 0


def test_transform_drops_noise_columns(synthetic_lc_frame):
    out = transform(synthetic_lc_frame, add_macro=False)
    for col in ("id", "title", "desc", "experience_c"):
        assert col not in out.columns, f"{col} should have been dropped"


def test_transform_emp_length_becomes_numeric(synthetic_lc_frame):
    out = transform(synthetic_lc_frame, add_macro=False)
    assert pd.api.types.is_float_dtype(out["emp_length"])
    # '< 1 year' → 0, '10+ years' → 10
    assert out["emp_length"].max() <= 10
    assert out["emp_length"].min() >= 0


def test_transform_preserves_row_count(synthetic_lc_frame):
    out = transform(synthetic_lc_frame, add_macro=False)
    assert len(out) == len(synthetic_lc_frame)


# ---------------------------------------------------------------------------
# Macro merge — uses the committed parquet, no network
# ---------------------------------------------------------------------------
def test_transform_with_macro_merge(synthetic_lc_frame):
    """Confirms the FRED macro merge works against the committed cache.

    Skipped if the macro cache is somehow missing — otherwise this is the
    hermetic equivalent of `make pipeline-lc`.
    """
    if not MACRO_CACHE_PATH.exists():
        pytest.skip(f"macro cache missing at {MACRO_CACHE_PATH}")

    out = transform(synthetic_lc_frame, add_macro=True)
    expected_macro = {
        "fed_funds_rate",
        "us_unemployment",
        "vix_close",
        "us_10y_treasury",
        "us_real_gdp_yoy",
    }
    assert expected_macro.issubset(out.columns), (
        f"macro columns missing: {expected_macro - set(out.columns)}"
    )
    # At least one macro column must have non-null values for the 2007-2018 window.
    assert out["fed_funds_rate"].notna().any()
    assert out["us_unemployment"].notna().any()
