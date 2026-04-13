"""Fixtures compartilhadas entre todos os testes."""

import numpy as np
import pandas as pd
import pytest

from src.contextual.data_generator import generate_dataset


@pytest.fixture(scope="session")
def synthetic_df():
    """Dataset sintético pequeno para testes rápidos."""
    return generate_dataset(n=300, seed=0)


@pytest.fixture(scope="session")
def small_credit_df():
    """DataFrame simulando feature store do Home Credit (sem dados reais)."""
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": rng.integers(0, 2, n),
            "AMT_CREDIT": rng.uniform(50_000, 500_000, n),
            "AMT_INCOME_TOTAL": rng.uniform(30_000, 300_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 30_000, n),
            "DAYS_BIRTH": rng.integers(-25000, -6000, n),
            "DAYS_EMPLOYED": rng.integers(-5000, 0, n),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "credit_income_ratio": rng.uniform(0.5, 5.0, n),
            "annuity_income_ratio": rng.uniform(0.05, 0.5, n),
            "ext_source_mean": rng.uniform(0, 1, n),
            "ext_source_min": rng.uniform(0, 0.5, n),
            "ext_source_max": rng.uniform(0.5, 1, n),
            "age_years": rng.uniform(20, 70, n),
            "employed_years": rng.uniform(0, 20, n),
        }
    )


@pytest.fixture(scope="session")
def psi_dataframes():
    """Par de DataFrames para testar PSI (referência vs atual com drift)."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 1000),
            "feature_b": rng.uniform(0, 1, 1000),
            "feature_c": rng.exponential(1.0, 1000),
        }
    )
    curr_drift = pd.DataFrame(
        {
            "feature_a": rng.normal(0.8, 1.2, 500),  # drift claro
            "feature_b": rng.uniform(0, 1, 500),  # estável
            "feature_c": rng.exponential(2.0, 500),  # drift moderado
        }
    )
    return ref, curr_drift
