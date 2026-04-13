"""Data drift detection and reporting.

Separates two types of deterioration:
1. Population drift: change in feature distributions (PSI)
2. Model drift: metric degradation (AUROC, KS)

Main functions:
- detect_drift: classifies drift status for a PSI series
- drift_report: full drift report for a period
- plot_psi_heatmap: PSI heatmap by feature and period
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from src.config import PSI_ATTENTION, PSI_STABLE
from src.monitoring.psi import psi_all_features


def detect_drift(psi_series: pd.Series) -> pd.Series:
    """Classifies drift status for a series of PSI values.

    Args:
        psi_series: pd.Series with PSI per feature (index = feature, value = PSI).

    Returns:
        pd.Series with status: 'stable', 'attention' or 'drift'.
    """

    def _classify(v: float) -> str:
        if v < PSI_STABLE:
            return "stable"
        elif v < PSI_ATTENTION:
            return "attention"
        return "drift"

    return psi_series.map(_classify)


def drift_report(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    period_label: str = "current",
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Generates full drift report between two populations.

    Args:
        ref_df: Reference DataFrame (training).
        curr_df: Current DataFrame.
        period_label: Label for the current period (e.g.: "2024-Q1").
        save_path: If provided, saves the report as CSV.

    Returns:
        DataFrame with feature, psi, status, delta_mean, delta_std.
    """
    psi_df = psi_all_features(ref_df, curr_df)

    numeric_cols = [
        c
        for c in psi_df["feature"].tolist()
        if c in ref_df.columns and c in curr_df.columns
    ]

    delta_stats = []
    for col in numeric_cols:
        ref_mean = ref_df[col].mean()
        curr_mean = curr_df[col].mean()
        ref_std = ref_df[col].std()
        curr_std = curr_df[col].std()
        delta_stats.append(
            {
                "feature": col,
                "ref_mean": ref_mean,
                "curr_mean": curr_mean,
                "delta_mean": curr_mean - ref_mean,
                "ref_std": ref_std,
                "curr_std": curr_std,
            }
        )

    delta_df = pd.DataFrame(delta_stats)
    report = psi_df.merge(delta_df, on="feature", how="left")
    report["period"] = period_label

    n_drift = (report["status"] == "drift").sum()
    n_attention = (report["status"] == "attention").sum()
    n_stable = (report["status"] == "stable").sum()

    logger.info(
        f"Drift report [{period_label}] — "
        f"Drift: {n_drift} | Attention: {n_attention} | Stable: {n_stable}"
    )

    if n_drift > 0:
        drift_features = report[report["status"] == "drift"]["feature"].tolist()
        logger.warning(f"Features in drift: {drift_features}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(save_path, index=False)
        logger.info(f"Report saved at {save_path}")

    return report


def plot_psi_heatmap(
    reports: list[tuple[str, pd.DataFrame]],
    top_n: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """PSI heatmap by feature and period.

    Args:
        reports: List of (period_label, drift_report_df).
        top_n: Number of features with highest mean PSI to display.
        save_path: If provided, saves the figure.

    Returns:
        Matplotlib figure.
    """

    pivot = pd.concat(
        [
            df[["feature", "psi"]].set_index("feature").rename(columns={"psi": label})
            for label, df in reports
        ],
        axis=1,
    )

    # Select top N by mean PSI
    pivot = pivot.loc[pivot.mean(axis=1).nlargest(top_n).index]

    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 1.5), max(6, top_n * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.25)
    plt.colorbar(im, ax=ax, label="PSI")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Threshold lines
    for threshold, color, label in [
        (PSI_STABLE, "green", f"Stable (<{PSI_STABLE})"),
        (PSI_ATTENTION, "orange", f"Attention (<{PSI_ATTENTION})"),
    ]:
        pass  # lines would be on the colorbar — keeping it simple

    ax.set_title("PSI by Feature and Period")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PSI heatmap saved at {save_path}")

    return fig
