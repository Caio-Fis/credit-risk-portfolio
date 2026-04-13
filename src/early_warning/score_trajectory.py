"""Score trajectory monitoring for early warning.

Detects credit deterioration before default through sharp drops
in the PD score over time.

Operational rule:
- Drop > SCORE_DROP_THRESHOLD points in SCORE_DROP_WINDOW_DAYS days → alert

Main functions:
- compute_score_trajectory: calculates score change per entity/contract
- flag_score_drop: identifies entities with drop above threshold
- score_trend: linear regression of score to detect trend
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.config import SCORE_DROP_THRESHOLD, SCORE_DROP_WINDOW_DAYS


def compute_score_trajectory(
    df: pd.DataFrame,
    entity_col: str = "SK_ID_CURR",
    score_col: str = "pd_score",
    date_col: str = "reference_date",
    window: int = SCORE_DROP_WINDOW_DAYS,
) -> pd.DataFrame:
    """Calculates the score change for each entity within the time window.

    The monitoring score is defined as (1 - PD) × 1000 to
    maintain the convention "higher score = better" (bureau style).

    Args:
        df: DataFrame with score history per entity and date.
        entity_col: Entity identifier column.
        score_col: Score column (pd_score = (1-PD)×1000).
        date_col: Reference date column.
        window: Window in days for calculating the change.

    Returns:
        DataFrame with additional columns: score_start, score_end, score_drop.
    """
    for col in (entity_col, score_col, date_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([entity_col, date_col])

    cutoff_date = df[date_col].max() - pd.Timedelta(days=window)

    results = []
    for entity_id, group in df.groupby(entity_col):
        recent = group[group[date_col] >= cutoff_date]
        historic = group[group[date_col] < cutoff_date]

        if recent.empty or historic.empty:
            continue

        score_end = recent[score_col].iloc[-1]
        score_start = historic[score_col].iloc[-1]
        drop = score_start - score_end  # positive = drop

        results.append(
            {
                entity_col: entity_id,
                "score_start": score_start,
                "score_end": score_end,
                "score_drop": drop,
                "date_start": historic[date_col].iloc[-1],
                "date_end": recent[date_col].iloc[-1],
            }
        )

    trajectory = pd.DataFrame(results)
    logger.info(
        f"Trajectory computed for {len(trajectory):,} entities. "
        f"Mean drop: {trajectory['score_drop'].mean():.1f} pts"
    )
    return trajectory


def flag_score_drop(
    trajectory: pd.DataFrame,
    threshold: int = SCORE_DROP_THRESHOLD,
    entity_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Identifies entities with score drop above the threshold.

    Args:
        trajectory: Result of compute_score_trajectory().
        threshold: Minimum drop in points to trigger an alert.
        entity_col: Entity identifier column.

    Returns:
        Filtered DataFrame with only entities in alert,
        sorted by score_drop descending.
    """
    alerts = (
        trajectory[trajectory["score_drop"] >= threshold]
        .sort_values("score_drop", ascending=False)
        .reset_index(drop=True)
    )
    alerts["alert_level"] = alerts["score_drop"].apply(
        lambda x: "critical" if x >= threshold * 2 else "attention"
    )

    logger.warning(
        f"Early warning: {len(alerts)} entities with score drop "
        f">= {threshold} pts in the last {SCORE_DROP_WINDOW_DAYS}d"
    )
    return alerts


def score_trend(
    df: pd.DataFrame,
    entity_id,
    entity_col: str = "SK_ID_CURR",
    score_col: str = "pd_score",
    date_col: str = "reference_date",
) -> dict:
    """Calculates linear trend of the score for an entity.

    Args:
        df: DataFrame with score history.
        entity_id: Entity ID to analyse.
        entity_col: Identifier column.
        score_col: Score column.
        date_col: Date column.

    Returns:
        Dictionary with slope (pts/month), r_squared and trend_label.
    """
    entity_df = df[df[entity_col] == entity_id].copy()
    entity_df[date_col] = pd.to_datetime(entity_df[date_col])
    entity_df = entity_df.sort_values(date_col)

    if len(entity_df) < 3:
        return {"slope": 0.0, "r_squared": 0.0, "trend_label": "insufficient"}

    x = (entity_df[date_col] - entity_df[date_col].min()).dt.days.values.astype(float)
    y = entity_df[score_col].values.astype(float)

    coeffs = np.polyfit(x, y, deg=1)
    slope_per_day = coeffs[0]
    slope_per_month = slope_per_day * 30

    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if slope_per_month < -10:
        trend_label = "deteriorating"
    elif slope_per_month > 10:
        trend_label = "improving"
    else:
        trend_label = "stable"

    return {
        "slope_pts_per_month": float(slope_per_month),
        "r_squared": float(r2),
        "trend_label": trend_label,
    }


def plot_score_trajectory(
    df: pd.DataFrame,
    entity_id,
    entity_col: str = "SK_ID_CURR",
    score_col: str = "pd_score",
    date_col: str = "reference_date",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plots the score trajectory for an entity with a trend line.

    Args:
        df: DataFrame with score history.
        entity_id: Entity ID.
        entity_col: Identifier column.
        score_col: Score column.
        date_col: Date column.
        save_path: If provided, saves the figure.

    Returns:
        Matplotlib figure.
    """
    entity_df = df[df[entity_col] == entity_id].copy()
    entity_df[date_col] = pd.to_datetime(entity_df[date_col])
    entity_df = entity_df.sort_values(date_col)

    trend = score_trend(df, entity_id, entity_col, score_col, date_col)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        entity_df[date_col],
        entity_df[score_col],
        "o-",
        color="steelblue",
        label="Score",
    )
    ax.axhline(
        y=entity_df[score_col].iloc[-1],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Current score",
    )

    ax.set_title(
        f"Score Trajectory — {entity_col}={entity_id} | Trend: {trend['trend_label']}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Score (0–1000)")
    ax.set_ylim(0, 1000)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
