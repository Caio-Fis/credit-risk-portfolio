"""Expected Loss calculation per contract.

EL = PD × LGD × EAD

- PD: Probability of Default (calibrated model)
- LGD: Loss Given Default (Beta regression)
- EAD: Exposure at Default — amount at risk (proxy: AMT_CREDIT)

Main functions:
- compute_el: calculates EL per contract in currency units
- el_summary: summarises portfolio EL (total, mean, by risk decile)
- el_by_segment: breakdown by segment (product, tenor, etc.)
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_el(
    pd_proba: np.ndarray | pd.Series,
    lgd: np.ndarray | pd.Series,
    ead: np.ndarray | pd.Series,
) -> pd.Series:
    """Calculates Expected Loss per contract: EL = PD × LGD × EAD.

    Args:
        pd_proba: Calibrated default probability (0–1).
        lgd: Estimated Loss Given Default (0–1).
        ead: Exposure at Default in currency units (credit amount).

    Returns:
        pd.Series with EL in currency units per contract.
    """
    pd_arr = np.asarray(pd_proba, dtype=float)
    lgd_arr = np.asarray(lgd, dtype=float)
    ead_arr = np.asarray(ead, dtype=float)

    if not (len(pd_arr) == len(lgd_arr) == len(ead_arr)):
        raise ValueError("pd_proba, lgd, and ead must have the same length.")

    el = pd_arr * lgd_arr * ead_arr

    logger.info(
        f"EL computed — total: {el.sum():,.2f} | "
        f"mean: {el.mean():,.2f} | "
        f"max: {el.max():,.2f}"
    )
    return pd.Series(el, name="expected_loss")


def el_summary(
    df: pd.DataFrame,
    el_col: str = "expected_loss",
    ead_col: str = "AMT_CREDIT",
) -> pd.DataFrame:
    """Summarises portfolio Expected Loss.

    Args:
        df: DataFrame with EL and EAD columns.
        el_col: EL column name.
        ead_col: EAD column name.

    Returns:
        DataFrame with portfolio statistics.
    """
    if el_col not in df.columns:
        raise KeyError(f"Column '{el_col}' not found. Run compute_el() first.")

    total_ead = df[ead_col].sum() if ead_col in df.columns else np.nan
    total_el = df[el_col].sum()
    el_rate = total_el / total_ead if total_ead > 0 else np.nan

    summary = pd.DataFrame(
        [
            {
                "total_ead": total_ead,
                "total_el": total_el,
                "el_rate_%": el_rate * 100,
                "el_mean": df[el_col].mean(),
                "el_median": df[el_col].median(),
                "el_p95": df[el_col].quantile(0.95),
                "el_p99": df[el_col].quantile(0.99),
                "n_contracts": len(df),
            }
        ]
    )

    logger.info(f"\n{summary.T.to_string()}")
    return summary


def el_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    el_col: str = "expected_loss",
    ead_col: str = "AMT_CREDIT",
) -> pd.DataFrame:
    """EL breakdown by segment.

    Args:
        df: DataFrame with computed EL.
        segment_col: Segmentation column (e.g.: product_type, tenor_months).
        el_col: EL column.
        ead_col: EAD column.

    Returns:
        DataFrame with EL aggregated by segment.
    """
    agg = (
        df.groupby(segment_col)
        .agg(
            n_contracts=(el_col, "count"),
            total_ead=(ead_col, "sum"),
            total_el=(el_col, "sum"),
            el_mean=(el_col, "mean"),
        )
        .assign(el_rate_pct=lambda x: x["total_el"] / x["total_ead"] * 100)
        .sort_values("el_rate_pct", ascending=False)
        .reset_index()
    )
    return agg


def add_el_to_df(
    df: pd.DataFrame,
    pd_col: str = "pd_proba",
    lgd_col: str = "lgd_pred",
    ead_col: str = "AMT_CREDIT",
) -> pd.DataFrame:
    """Adds expected_loss column to the DataFrame.

    Args:
        df: DataFrame with PD, LGD, and EAD columns.
        pd_col: Predicted PD column.
        lgd_col: Predicted LGD column.
        ead_col: EAD column.

    Returns:
        DataFrame with 'expected_loss' column added.
    """
    for col in (pd_col, lgd_col, ead_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    df["expected_loss"] = compute_el(df[pd_col], df[lgd_col], df[ead_col])
    return df
