"""Temporal-safe feature engineering for the LendingClub 2007-2018 dataset.

The Zenodo distribution (record 11295916) is already curated to the
*loan-granting* slice: 15 columns, no post-origination leakage, and a
pre-derived binary ``Default`` target. This module:

  1. Parses ``issue_d`` into a datetime (canonical temporal axis).
  2. Parses ``emp_length`` from LendingClub's bucketed strings to a numeric.
  3. Drops constant / free-text / identifier columns.
  4. Renames ``Default`` → ``target`` for downstream consistency.
  5. Merges FRED US macro features via ``merge_asof`` (no look-ahead).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import LENDINGCLUB_FEATURES, LENDINGCLUB_PARQUET
from src.features.macro_features import merge_macro_features
from src.ingestion.download_lendingclub import load_lendingclub


def _parse_emp_length(s: pd.Series) -> pd.Series:
    """'10+ years' → 10, '< 1 year' → 0, '5 years' → 5, 'NI'/'n/a' → NaN."""
    mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "10+ years": 10,
        "n/a": np.nan,
        "NI": np.nan,
    }
    out = s.map(mapping)
    extracted = pd.to_numeric(
        s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    return out.where(~out.isna() | s.isin(mapping), extracted).astype("float32")


def transform(df: pd.DataFrame, add_macro: bool = True) -> pd.DataFrame:
    """End-to-end feature pipeline.

    Output guarantees:
        - 'target' column: int8 in {0, 1}, no NaN
        - 'issue_d' column: datetime, no NaN — canonical temporal axis
        - Categorical columns kept as object/category for LightGBM native handling
        - Macro features merged via merge_asof backward (no look-ahead)
        - Identifier / free-text / constant columns dropped
    """
    df = df.copy()

    # --- Target -------------------------------------------------------------
    if "target" not in df.columns:
        if "Default" not in df.columns:
            raise ValueError("Neither 'target' nor 'Default' column present.")
        df = df.rename(columns={"Default": "target"})
    df["target"] = df["target"].astype("int8")

    # --- Temporal axis ------------------------------------------------------
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    n_bad = df["issue_d"].isna().sum()
    if n_bad:
        logger.warning(f"Dropping {n_bad:,} rows with unparseable issue_d")
        df = df.dropna(subset=["issue_d"]).reset_index(drop=True)

    # --- emp_length ---------------------------------------------------------
    if "emp_length" in df.columns:
        df["emp_length"] = _parse_emp_length(df["emp_length"])

    # --- Drop noise ---------------------------------------------------------
    drop_cols: list[str] = []
    for c in ("id", "title", "desc"):
        if c in df.columns:
            drop_cols.append(c)
    # Drop columns with zero variance (e.g. experience_c is constant=1 here)
    for c in df.columns:
        if c in ("target", "issue_d"):
            continue
        if df[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
            logger.info(f"Dropping constant column: {c}")
    df = df.drop(columns=list(dict.fromkeys(drop_cols)))

    # --- Macro merge --------------------------------------------------------
    if add_macro:
        df = merge_macro_features(df, date_col="issue_d")

    logger.success(
        f"Features built: {len(df):,} rows × {df.shape[1]} cols "
        f"({df['issue_d'].min().date()} → {df['issue_d'].max().date()}), "
        f"default rate {df['target'].mean():.2%}"
    )
    return df


def build_features(
    parquet_in: Path = LENDINGCLUB_PARQUET,
    parquet_out: Path = LENDINGCLUB_FEATURES,
    add_macro: bool = True,
    force: bool = False,
) -> Path:
    """Reads raw parquet, applies transform, writes feature parquet."""
    if not force and parquet_out.exists():
        logger.info(f"Features parquet already exists: {parquet_out}")
        return parquet_out

    df = load_lendingclub(parquet_in)
    out = transform(df, add_macro=add_macro)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_out, index=False)
    logger.success(f"Features written: {parquet_out} ({parquet_out.stat().st_size / 1e6:.1f} MB)")
    return parquet_out


if __name__ == "__main__":
    build_features(force=True)
