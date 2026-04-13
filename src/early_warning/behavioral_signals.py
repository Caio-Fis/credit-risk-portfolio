"""Behavioural triggers for early warning.

Complements the score trajectory with operational signals that indicate
deterioration before the score captures it:

- Transaction volume drop: reduction in revenue / invoices issued
- Protests: protest registration at a notary
- Delays in other obligations: rising DPD in other operations

In the Home Credit dataset, these signals are approximated by:
- Drop in the number of installments paid in the month
- Increase in DPD in bureau_balance
- Increase in contracts with "Bad debt" status in bureau

Main functions:
- flag_volume_drop: transaction volume drop
- flag_protests: protests / bad debt in bureau
- aggregate_signals: aggregates all signals into a composite risk score
"""

import numpy as np
import pandas as pd
from loguru import logger

# Volume thresholds (configurable)
VOLUME_DROP_PCT = 0.30  # 30% volume drop → alert
PROTEST_FLAG_THRESHOLD = 1  # any protest → alert


def flag_volume_drop(
    df: pd.DataFrame,
    entity_col: str = "SK_ID_CURR",
    volume_col: str = "monthly_payment_count",
    date_col: str = "reference_date",
    window_months: int = 3,
    threshold_pct: float = VOLUME_DROP_PCT,
) -> pd.DataFrame:
    """Detects sharp drop in transaction/payment volume.

    Compares the mean volume over the last `window_months` with the prior period.

    Args:
        df: DataFrame with volume history per entity and month.
        entity_col: Entity identifier column.
        volume_col: Monthly volume column (e.g.: payments made).
        date_col: Reference date column.
        window_months: Comparison window in months.
        threshold_pct: Drop percentage that triggers the alert.

    Returns:
        DataFrame with entities in volume drop alert.
    """
    for col in (entity_col, volume_col, date_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([entity_col, date_col])

    alerts = []
    cutoff = df[date_col].max() - pd.DateOffset(months=window_months)

    for entity_id, group in df.groupby(entity_col):
        recent = group[group[date_col] >= cutoff][volume_col].mean()
        historic = group[group[date_col] < cutoff][volume_col].mean()

        if pd.isna(recent) or pd.isna(historic) or historic == 0:
            continue

        drop_pct = (historic - recent) / historic
        if drop_pct >= threshold_pct:
            alerts.append(
                {
                    entity_col: entity_id,
                    "volume_recent": recent,
                    "volume_hist": historic,
                    "volume_drop_pct": drop_pct,
                    "signal_volume": True,
                }
            )

    result = pd.DataFrame(alerts)
    logger.info(
        f"Volume drop: {len(result)} entities in alert (>{threshold_pct:.0%})"
    )
    return result


def flag_protests(
    bureau_df: pd.DataFrame,
    entity_col: str = "SK_ID_CURR",
    status_col: str = "CREDIT_ACTIVE",
    bad_status_values: list[str] | None = None,
) -> pd.DataFrame:
    """Detects contracts with protest or bad debt status in the bureau.

    In Home Credit, uses CREDIT_ACTIVE == 'Bad debt' as a protest proxy.

    Args:
        bureau_df: Bureau DataFrame with active credit statuses.
        entity_col: Identifier column.
        status_col: Credit status column in the bureau.
        bad_status_values: Values that indicate protest/bad debt.

    Returns:
        DataFrame with protested entities, sorted by count.
    """
    if bad_status_values is None:
        bad_status_values = ["Bad debt", "Sold"]

    if status_col not in bureau_df.columns:
        logger.warning(f"Column '{status_col}' not found. No protest signal.")
        return pd.DataFrame(columns=[entity_col, "protest_count", "signal_protest"])

    bad_df = bureau_df[bureau_df[status_col].isin(bad_status_values)]

    result = (
        bad_df.groupby(entity_col)
        .size()
        .reset_index(name="protest_count")
        .assign(signal_protest=True)
    )

    logger.info(f"Protests: {len(result)} entities with bad debt/sold in bureau")
    return result


def aggregate_signals(
    trajectory_alerts: pd.DataFrame,
    volume_alerts: pd.DataFrame | None = None,
    protest_alerts: pd.DataFrame | None = None,
    entity_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Aggregates all signals into a composite risk score per entity.

    Composite score (0–3):
    - +1 if score drop > threshold
    - +1 if volume drop > threshold
    - +1 if protest/bad debt in bureau

    Args:
        trajectory_alerts: Result of flag_score_drop().
        volume_alerts: Result of flag_volume_drop() (optional).
        protest_alerts: Result of flag_protests() (optional).
        entity_col: Identifier column.

    Returns:
        DataFrame with entities in alert, composite score and active signals.
    """
    # Base: entities with score drop
    base = trajectory_alerts[[entity_col, "score_drop", "alert_level"]].copy()
    base["signal_score"] = True
    base["composite_risk"] = 1

    if (
        volume_alerts is not None
        and not volume_alerts.empty
        and entity_col in volume_alerts.columns
    ):
        vol_subset = volume_alerts[[entity_col, "volume_drop_pct", "signal_volume"]]
        base = base.merge(vol_subset, on=entity_col, how="left")
        base["signal_volume"] = base["signal_volume"].fillna(False)
        base["composite_risk"] += base["signal_volume"].astype(int)
    else:
        base["signal_volume"] = False

    if (
        protest_alerts is not None
        and not protest_alerts.empty
        and entity_col in protest_alerts.columns
    ):
        prot_subset = protest_alerts[[entity_col, "protest_count", "signal_protest"]]
        base = base.merge(prot_subset, on=entity_col, how="left")
        base["signal_protest"] = base["signal_protest"].fillna(False)
        base["composite_risk"] += base["signal_protest"].astype(int)
    else:
        base["signal_protest"] = False

    base["risk_label"] = (
        base["composite_risk"]
        .map(
            {
                1: "moderate",
                2: "high",
                3: "critical",
            }
        )
        .fillna("moderate")
    )

    result = base.sort_values(
        ["composite_risk", "score_drop"], ascending=False
    ).reset_index(drop=True)

    logger.warning(
        f"Composite alert: {len(result)} entities | "
        f"critical={(result['composite_risk'] == 3).sum()} | "
        f"high={(result['composite_risk'] == 2).sum()} | "
        f"moderate={(result['composite_risk'] == 1).sum()}"
    )
    return result


def simulate_behavioral_data(
    n_entities: int = 500,
    n_months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic behavioural data for demonstration.

    Args:
        n_entities: Number of entities.
        n_months: Number of months of history.
        seed: Random seed.

    Returns:
        DataFrame with entity_id, reference_date, pd_score, monthly_payment_count.
    """
    rng = np.random.default_rng(seed)
    records = []
    base_date = pd.Timestamp("2023-01-01")

    for i in range(n_entities):
        # Initial score (500–900)
        score_init = rng.integers(500, 900)
        # Trend: 80% stable, 15% drop, 5% improve
        trend_type = rng.choice(["stable", "drop", "improve"], p=[0.80, 0.15, 0.05])

        # Monthly payment volume
        vol_base = rng.integers(3, 15)

        for m in range(n_months):
            date = base_date + pd.DateOffset(months=m)

            if trend_type == "drop" and m >= n_months // 2:
                noise = rng.normal(0, 15)
                score = max(100, score_init - (m - n_months // 2) * 20 + noise)
                volume = max(
                    0, vol_base - (m - n_months // 2) * 2 + rng.integers(-1, 2)
                )
            elif trend_type == "improve":
                noise = rng.normal(0, 10)
                score = min(999, score_init + m * 5 + noise)
                volume = min(20, vol_base + m // 3 + rng.integers(0, 2))
            else:
                noise = rng.normal(0, 20)
                score = float(np.clip(score_init + noise, 100, 999))
                volume = int(np.clip(vol_base + rng.integers(-2, 3), 0, 20))

            records.append(
                {
                    "SK_ID_CURR": i,
                    "reference_date": date,
                    "pd_score": round(score, 1),
                    "monthly_payment_count": volume,
                }
            )

    return pd.DataFrame(records)
