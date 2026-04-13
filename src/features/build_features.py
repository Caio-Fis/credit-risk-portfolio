"""Batch feature store construction with 30/90/365-day time windows.

Main functions:
- merge_tables: joins application_train with Home Credit auxiliary tables
- build_temporal_features: aggregates features by time windows
- save_feature_store: persists the feature store as parquet
- load_feature_store: loads the already-built feature store

Large tables (>10M rows) are aggregated via batch chunked processing in
batch_aggregations.py to avoid OOM — pandas never materialises the
27M or 13.6M rows at once; processes in batches of 1M rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROCESSED_DIR, RAW_DIR, TEMPORAL_WINDOWS
from src.features.batch_aggregations import (
    build_bureau_features_batch,
    build_credit_card_features_batch,
    build_installment_features_batch,
    build_pos_cash_features_batch,
)

HOME_CREDIT_DIR = RAW_DIR / "home_credit"


def _load_raw(filename: str, data_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {data_dir}. Run `make data` first."
        )
    logger.debug(f"Loading {filename}...")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Previous application features (small table — stays in pandas)
# ---------------------------------------------------------------------------


def _build_prev_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Counts applications and refusals by window (proxy: DAYS_DECISION)."""
    prev = prev.copy()
    prev["is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    rows = []
    for w in TEMPORAL_WINDOWS:
        mask = prev["DAYS_DECISION"] >= -w
        sub = prev[mask]

        agg = (
            sub.groupby("SK_ID_CURR")
            .agg(
                cnt_applications=("SK_ID_PREV", "nunique"),
                cnt_refused=("is_refused", "sum"),
            )
            .rename(columns=lambda c: f"prev_{c}_{w}d")
            .reset_index()
        )
        rows.append(agg)

    result = rows[0]
    for df in rows[1:]:
        result = result.merge(df, on="SK_ID_CURR", how="outer")

    return result.fillna(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_tables(data_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    """Joins application_train with features from auxiliary tables.

    Small tables (<2M rows) are processed in pandas.
    Large tables (bureau_balance 27M, installments 13.6M) are delegated
    to batch chunked aggregations to avoid OOM.

    Args:
        data_dir: Directory with raw CSVs.

    Returns:
        Wide DataFrame with features from all sources.
    """
    app = _load_raw("application_train.csv", data_dir)
    logger.info(f"application_train: {app.shape}")

    # Pandas: previous_application is small (~887K rows), no OOM risk
    prev = _load_raw("previous_application.csv", data_dir)
    prev_feats = _build_prev_application_features(prev)
    del prev  # free before starting batch processing

    # Batch chunked: bureau_balance (27M) + installments (13.6M) in 1M-row batches
    bureau_feats = build_bureau_features_batch(
        bureau_path=data_dir / "bureau.csv",
        bureau_bal_path=data_dir / "bureau_balance.csv",
        windows=TEMPORAL_WINDOWS,
    )
    inst_feats = build_installment_features_batch(
        inst_path=data_dir / "installments_payments.csv",
        windows=TEMPORAL_WINDOWS,
    )
    pos_feats = build_pos_cash_features_batch(
        pos_path=data_dir / "POS_CASH_balance.csv",
        windows=TEMPORAL_WINDOWS,
    )
    cc_feats = build_credit_card_features_batch(
        cc_path=data_dir / "credit_card_balance.csv",
        windows=TEMPORAL_WINDOWS,
    )

    df = (
        app.merge(bureau_feats, on="SK_ID_CURR", how="left")
        .merge(prev_feats, on="SK_ID_CURR", how="left")
        .merge(inst_feats, on="SK_ID_CURR", how="left")
        .merge(pos_feats, on="SK_ID_CURR", how="left")
        .merge(cc_feats, on="SK_ID_CURR", how="left")
    )

    logger.info(f"Unified table: {df.shape}")
    return df


def build_temporal_features(
    df: pd.DataFrame,
    windows: list[int] = TEMPORAL_WINDOWS,
) -> pd.DataFrame:
    """Adds derived features to the already-unified DataFrame.

    Includes:
    - Credit ratios (credit/income, annuity/income)
    - Anomalous employment flags
    - Mean external score

    Args:
        df: DataFrame result of merge_tables().
        windows: Time windows to consider (informational, does not recompute).

    Returns:
        DataFrame with additional features.
    """
    df = df.copy()

    # Financial ratios
    df["credit_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["credit_annuity_ratio"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    # Age and employment duration in years
    df["age_years"] = -df["DAYS_BIRTH"] / 365
    df["employed_years"] = np.where(
        df["DAYS_EMPLOYED"] == 365243,  # code for "unemployed"
        0,
        -df["DAYS_EMPLOYED"] / 365,
    )
    df["employed_to_age_ratio"] = df["employed_years"] / (df["age_years"] + 1)

    # External score: aggregations and interactions
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available = [c for c in ext_cols if c in df.columns]
    if available:
        df["ext_source_mean"] = df[available].mean(axis=1)
        df["ext_source_min"] = df[available].min(axis=1)
        df["ext_source_max"] = df[available].max(axis=1)
        # Product of EXT_SOURCEs — captures joint signal; top feature in Kaggle Home Credit
        df["ext_source_product"] = df[available].prod(axis=1)
        # Pairwise interaction between the two most predictive scores
        if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
            df["ext_source_2x3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # Bureau overdue rate by window
    for w in windows:
        cnt_col = f"bureau_cnt_credits_{w}d"
        ovd_col = f"bureau_cnt_overdue_{w}d"
        if cnt_col in df.columns and ovd_col in df.columns:
            df[f"bureau_overdue_rate_{w}d"] = (
                df[ovd_col] / (df[cnt_col] + 1)
            )

    # Refusal rate in previous applications by window
    for w in windows:
        app_col = f"prev_cnt_applications_{w}d"
        ref_col = f"prev_cnt_refused_{w}d"
        if app_col in df.columns and ref_col in df.columns:
            df[f"prev_refusal_rate_{w}d"] = (
                df[ref_col] / (df[app_col] + 1)
            )

    # Social circle default rate (risk network proxy)
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["social_circle_default_rate"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
        )

    # LTV: credit / financed asset ratio
    if "AMT_GOODS_PRICE" in df.columns:
        df["credit_goods_ratio"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    # Per-capita family income
    if "CNT_FAM_MEMBERS" in df.columns:
        df["income_per_family_member"] = (
            df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
        )

    # Explicit unemployed flag (365243 is a special code — the flag has a different signal than 0)
    df["flag_unemployed"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)

    logger.info(f"Features built: {df.shape[1]} columns")
    return df


def save_feature_store(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DIR / "feature_store.parquet",
) -> Path:
    """Persists the feature store as parquet.

    Args:
        df: DataFrame with all features.
        output_path: Output path.

    Returns:
        Path to the saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(
        f"Feature store saved: {output_path} ({df.shape[0]:,} × {df.shape[1]})"
    )
    return output_path


def load_feature_store(
    path: Path = PROCESSED_DIR / "feature_store.parquet",
) -> pd.DataFrame:
    """Loads the already-built feature store.

    Args:
        path: Path to the parquet file.

    Returns:
        DataFrame with all features.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {path}. Run `make features`."
        )
    df = pd.read_parquet(path)
    logger.info(f"Feature store loaded: {df.shape[0]:,} × {df.shape[1]}")
    return df


if __name__ == "__main__":
    df_raw = merge_tables()
    df_features = build_temporal_features(df_raw)
    save_feature_store(df_features)
