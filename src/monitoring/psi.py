"""Population Stability Index (PSI) per feature.

PSI measures how different the current distribution of a feature is
relative to the reference distribution (training).

Interpretation:
- PSI < 0.10  → stable (no action required)
- PSI 0.10–0.20 → attention (investigate cause)
- PSI > 0.20  → drift confirmed (retraining required)

Main functions:
- compute_psi: PSI for a single feature
- psi_all_features: PSI for all numeric features of two DataFrames
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PSI_ATTENTION, PSI_STABLE

EPS = 1e-6  # avoids log(0)


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Calculates PSI between expected and actual distributions.

    Args:
        expected: Reference distribution (e.g.: training data).
        actual: Current distribution (e.g.: production data).
        bins: Number of bins for discretisation.

    Returns:
        PSI score (float >= 0).
    """
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)

    # Remove NaNs
    expected_clean = expected_arr[~np.isnan(expected_arr)]
    actual_clean = actual_arr[~np.isnan(actual_arr)]

    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0

    # Use quantiles of the reference distribution to define bins
    breakpoints = np.nanpercentile(expected_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates (constant features)

    if len(breakpoints) < 2:
        return 0.0

    # Count frequencies per bin
    expected_counts = np.histogram(expected_clean, bins=breakpoints)[0]
    actual_counts = np.histogram(actual_clean, bins=breakpoints)[0]

    # Convert to proportions (sum = 1)
    expected_pct = (expected_counts / len(expected_clean)).clip(EPS)
    actual_pct = (actual_counts / len(actual_clean)).clip(EPS)

    # PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def psi_all_features(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    bins: int = 10,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Calculates PSI for all features and classifies status.

    Args:
        ref_df: Reference DataFrame (training).
        curr_df: Current DataFrame (production/validation).
        bins: Number of bins per feature.
        numeric_only: If True, processes only numeric columns.

    Returns:
        DataFrame with columns: feature, psi, status (stable/attention/drift).
    """
    if numeric_only:
        cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in ref_df.columns if c in curr_df.columns]

    records = []
    for col in cols:
        if col not in curr_df.columns:
            continue
        psi_val = compute_psi(ref_df[col], curr_df[col], bins=bins)
        records.append({"feature": col, "psi": psi_val})

    result = (
        pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)
    )
    result["status"] = result["psi"].apply(_classify_psi)

    n_drift = (result["status"] == "drift").sum()
    n_attention = (result["status"] == "attention").sum()
    logger.info(
        f"PSI computed — {n_drift} features in drift, {n_attention} under attention"
    )

    return result


def _classify_psi(psi_val: float) -> str:
    """Classifies PSI according to operational thresholds."""
    if psi_val < PSI_STABLE:
        return "stable"
    elif psi_val < PSI_ATTENTION:
        return "attention"
    else:
        return "drift"
