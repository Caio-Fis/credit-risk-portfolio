"""Vintage analysis.

Measures cumulative default rate by origination cohort and maturity.
Allows identifying whether the model is deteriorating or the portfolio has changed.

Terminology:
- Vintage: origination month of the contract
- Maturity: months since origination
- DPD: Days Past Due

Main functions:
- build_vintage_matrix: builds default matrix (vintage × maturity)
- plot_vintage_curves: plots default curves by vintage
- compare_vintages: compares recent vs. older vintages
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def build_vintage_matrix(
    df: pd.DataFrame,
    vintage_col: str = "vintage",
    maturity_col: str = "maturity_month",
    default_col: str = "is_default",
    min_observations: int = 30,
) -> pd.DataFrame:
    """Builds cumulative default matrix by vintage and maturity.

    Args:
        df: DataFrame with vintage, maturity, and default flag columns.
        vintage_col: Column indicating the origination period.
        maturity_col: Column with months since origination.
        default_col: Binary column (1 = default occurred up to this month).
        min_observations: Vintages with fewer samples are excluded.

    Returns:
        Pivot table: rows = vintage, columns = maturity, values = default rate.
    """
    for col in (vintage_col, maturity_col, default_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Filter vintages with sufficient observations
    vintage_counts = df.groupby(vintage_col)[default_col].count()
    valid_vintages = vintage_counts[vintage_counts >= min_observations].index
    df_filtered = df[df[vintage_col].isin(valid_vintages)]

    if len(df_filtered) < len(df):
        n_removed = len(df) - len(df_filtered)
        logger.warning(
            f"{n_removed} records removed (vintages with < {min_observations} obs)."
        )

    # Cumulative default rate by vintage × maturity
    matrix = (
        df_filtered.groupby([vintage_col, maturity_col])[default_col]
        .mean()
        .unstack(maturity_col)
        .sort_index()
    )

    # Ensure monotonicity: default rate is cumulative
    matrix = matrix.cummax(axis=1)

    logger.info(
        f"Vintage matrix: {matrix.shape[0]} vintages × {matrix.shape[1]} maturities"
    )
    return matrix


def plot_vintage_curves(
    vintage_matrix: pd.DataFrame,
    highlight_recent: int = 3,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plots default curves by vintage.

    Args:
        vintage_matrix: Result of build_vintage_matrix().
        highlight_recent: Number of most recent vintages to highlight.
        save_path: If provided, saves the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_vintages = len(vintage_matrix)
    colors = cm.Blues(np.linspace(0.3, 1.0, n_vintages))

    for i, (vintage, row) in enumerate(vintage_matrix.iterrows()):
        values = row.dropna()
        is_recent = i >= (n_vintages - highlight_recent)

        ax.plot(
            values.index,
            values.values * 100,
            color=colors[i],
            linewidth=2.5 if is_recent else 1.0,
            alpha=1.0 if is_recent else 0.4,
            label=str(vintage) if is_recent else None,
        )

    ax.set_xlabel("Maturity (months)")
    ax.set_ylabel("Cumulative Default Rate (%)")
    ax.set_title("Vintage Analysis — Default Rate by Origination Cohort")
    ax.grid(alpha=0.3)

    if highlight_recent > 0:
        ax.legend(title="Recent vintages", loc="upper left")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Vintage curves saved at {save_path}")

    return fig


def compare_vintages(
    vintage_matrix: pd.DataFrame,
    reference_period: int = 6,
    comparison_maturity: int = 12,
) -> pd.DataFrame:
    """Compares default rate at a fixed maturity across vintages.

    Useful for identifying deterioration over time.

    Args:
        vintage_matrix: Result of build_vintage_matrix().
        reference_period: Number of older vintages to use as reference.
        comparison_maturity: Maturity (months) used for comparison.

    Returns:
        DataFrame with default rate per vintage at the specified maturity.
    """
    if comparison_maturity not in vintage_matrix.columns:
        available = vintage_matrix.columns.tolist()
        nearest = min(available, key=lambda x: abs(x - comparison_maturity))
        logger.warning(
            f"Maturity {comparison_maturity} not available. Using {nearest}."
        )
        comparison_maturity = nearest

    rates = vintage_matrix[comparison_maturity].dropna().reset_index()
    rates.columns = ["vintage", "default_rate"]
    rates["default_rate_pct"] = rates["default_rate"] * 100

    ref_rate = rates.head(reference_period)["default_rate"].mean()
    rates["vs_reference_pp"] = (rates["default_rate"] - ref_rate) * 100

    logger.info(
        f"Reference rate (first {reference_period} vintages) at {comparison_maturity}m: "
        f"{ref_rate:.2%}"
    )

    return rates


def simulate_vintage_data(
    n_contracts: int = 10000,
    n_vintages: int = 12,
    max_maturity: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic vintage data for demonstration.

    Simulates gradual deterioration: newer vintages have slightly higher PD.

    Args:
        n_contracts: Total number of contracts.
        n_vintages: Number of vintages (months).
        max_maturity: Maximum maturity in months.
        seed: Random seed.

    Returns:
        DataFrame with vintage, maturity_month, is_default.
    """
    rng = np.random.default_rng(seed)
    records = []

    base_pd = 0.08
    drift_per_vintage = 0.005  # 0.5pp deterioration per vintage

    contracts_per_vintage = n_contracts // n_vintages

    for v in range(n_vintages):
        vintage_pd = base_pd + v * drift_per_vintage
        maturities_observed = n_vintages - v  # newer vintages have less history

        for _ in range(contracts_per_vintage):
            # Default month (if it occurs)
            default_month = (
                int(rng.geometric(p=vintage_pd / max_maturity))
                if rng.random() < vintage_pd
                else None
            )

            for m in range(1, min(maturities_observed, max_maturity) + 1):
                records.append(
                    {
                        "vintage": v + 1,
                        "maturity_month": m,
                        "is_default": 1
                        if (default_month is not None and default_month <= m)
                        else 0,
                    }
                )

    return pd.DataFrame(records)
