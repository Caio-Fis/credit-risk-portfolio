"""Enrichment with macroeconomic series from the Brazilian Central Bank (SGS API).

Series used:
- 432:   Selic rate (% p.a.)
- 21082: Corporate default rate — total (%)
- 24364: IBC-Br — Economic Activity Index

Main functions:
- fetch_bcb_series: downloads a time series from the SGS/BCB API
- fetch_all_macro: downloads all required macroeconomic series
- merge_macro_features: merges macro features into the main DataFrame by date
"""

from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger

from src.config import BCB_IBC_BR, BCB_INADIMPLENCIA_PJ, BCB_SELIC, PROCESSED_DIR

BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
MACRO_CACHE_PATH = PROCESSED_DIR / "macro_features.parquet"

# Default period: last 10 years
DEFAULT_START = (datetime.now() - timedelta(days=10 * 365)).strftime("%d/%m/%Y")
DEFAULT_END = datetime.now().strftime("%d/%m/%Y")


def fetch_bcb_series(
    series_id: int,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    timeout: int = 30,
) -> pd.Series:
    """Downloads a time series from the Brazilian Central Bank SGS API.

    Args:
        series_id: Series code in SGS (e.g.: 432 for Selic).
        start_date: Start date in DD/MM/YYYY format.
        end_date: End date in DD/MM/YYYY format.
        timeout: HTTP timeout in seconds.

    Returns:
        pd.Series with date index and series values.
    """
    url = BCB_SGS_URL.format(series_id=series_id)
    params = {"formato": "json", "dataInicial": start_date, "dataFinal": end_date}

    logger.info(f"Fetching BCB series {series_id} ({start_date} → {end_date})...")
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not data:
        logger.warning(f"Series {series_id} returned empty for the requested period.")
        return pd.Series(dtype=float, name=f"bcb_{series_id}")

    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.set_index("data").sort_index()

    series = df["valor"].rename(f"bcb_{series_id}")
    logger.success(f"Series {series_id}: {len(series)} observations")
    return series


def fetch_all_macro(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    cache: bool = True,
) -> pd.DataFrame:
    """Downloads and combines all macroeconomic series.

    Args:
        start_date: Start date in DD/MM/YYYY format.
        end_date: End date in DD/MM/YYYY format.
        cache: If True, uses parquet cache if available.

    Returns:
        DataFrame with columns: selic_rate, pj_default_rate, ibc_br_yoy
    """
    if cache and MACRO_CACHE_PATH.exists():
        logger.info(f"Loading macro features from cache: {MACRO_CACHE_PATH}")
        return pd.read_parquet(MACRO_CACHE_PATH)

    series_map = {
        BCB_SELIC: "selic_rate",
        BCB_INADIMPLENCIA_PJ: "pj_default_rate",
        BCB_IBC_BR: "ibc_br",
    }

    frames = {}
    for series_id, col_name in series_map.items():
        try:
            s = fetch_bcb_series(series_id, start_date, end_date)
            frames[col_name] = s
        except Exception as exc:
            logger.warning(
                f"Failed to fetch series {series_id}: {exc}. Filling with NaN."
            )
            frames[col_name] = pd.Series(dtype=float, name=col_name)

    macro = pd.DataFrame(frames)
    macro.index.name = "date"

    # Ensure DatetimeIndex even when all series failed
    if not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)

    # Calculate IBC-Br year-over-year change
    if "ibc_br" in macro.columns:
        macro["ibc_br_yoy"] = macro["ibc_br"].pct_change(periods=12, fill_method=None) * 100
        macro = macro.drop(columns=["ibc_br"])

    # Monthly resampling only if there is data; fallback to empty series with NaN
    if not macro.empty:
        macro = macro.resample("ME").last().ffill()
    else:
        logger.warning("No macro series available; macro features will be NaN.")

    if cache:
        MACRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        macro.to_parquet(MACRO_CACHE_PATH)
        logger.success(f"Macro features cached at {MACRO_CACHE_PATH}")

    return macro


def merge_macro_features(
    df: pd.DataFrame,
    date_col: str = "reference_date",
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merges macroeconomic features into the main DataFrame by date.

    Home Credit has no absolute date. If `date_col` does not exist,
    macro features are added with the most recent available value
    (static snapshot — simulates a credit decision at a fixed point in time).

    Args:
        df: Main DataFrame (feature store).
        date_col: Date column for temporal merge. If absent, uses snapshot.
        macro: Macro features DataFrame (if None, calls fetch_all_macro()).

    Returns:
        DataFrame with macro columns added.
    """
    if macro is None:
        macro = fetch_all_macro()

    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        macro_reset = macro.reset_index().rename(columns={"date": date_col})
        df = pd.merge_asof(
            df,
            macro_reset.sort_values(date_col),
            on=date_col,
            direction="backward",
        )
        logger.info("Macro features merged by date (merge_asof).")
    else:
        snapshot = macro.iloc[-1]
        for col, val in snapshot.items():
            df[col] = val
        logger.info(
            f"Macro features added as snapshot ({macro.index[-1].date()})."
        )

    return df


if __name__ == "__main__":
    macro_df = fetch_all_macro(cache=True)
    logger.info(f"\n{macro_df.tail()}")
