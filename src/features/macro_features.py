"""US macroeconomic enrichment via FRED (St. Louis Fed) CSV endpoints.

LendingClub is a US-domiciled marketplace; macro context comes from FRED.
The CSV endpoint requires no API key, which keeps the project trivially
reproducible.

Series used (configured in src.config.FRED_SERIES):
- FEDFUNDS: Effective Federal Funds Rate (monthly, %)
- UNRATE:   Unemployment Rate (monthly, %)
- GDPC1:    Real GDP (quarterly, billions chained 2017$) → YoY derived
- VIXCLS:   VIX Close (daily, index) → resampled monthly
- DGS10:    10-Year Treasury Constant Maturity Rate (daily, %) → resampled monthly
"""

from io import StringIO

import pandas as pd
import requests
from loguru import logger

from src.config import FRED_CSV_URL, FRED_SERIES, MACRO_CACHE_PATH


def fetch_fred_series(series_id: str, timeout: int = 30) -> pd.Series:
    """Downloads a single FRED series via the CSV endpoint (no auth).

    Returns a pd.Series indexed by date, named after the series_id.
    Missing values come back as '.' in FRED CSVs — coerced to NaN.
    """
    url = FRED_CSV_URL.format(series_id=series_id)
    logger.info(f"Fetching FRED series {series_id}...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    # FRED CSVs come with two columns: observation_date (or DATE) and the series id.
    date_col = df.columns[0]
    value_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    series = df.set_index(date_col)[value_col].rename(series_id)
    logger.success(f"  {series_id}: {len(series)} observations ({series.index.min().date()} → {series.index.max().date()})")
    return series


def fetch_all_macro(cache: bool = True, force: bool = False) -> pd.DataFrame:
    """Downloads all configured FRED series and returns a monthly DataFrame.

    Output schema:
        Index: date (month-end)
        Columns: fed_funds_rate, us_unemployment, us_real_gdp_yoy,
                 vix_close, us_10y_treasury
    """
    if cache and not force and MACRO_CACHE_PATH.exists():
        logger.info(f"Loading macro features from cache: {MACRO_CACHE_PATH}")
        return pd.read_parquet(MACRO_CACHE_PATH)

    frames: dict[str, pd.Series] = {}
    for series_id, col_name in FRED_SERIES.items():
        try:
            s = fetch_fred_series(series_id)
            frames[col_name] = s
        except Exception as exc:
            logger.warning(f"Failed to fetch FRED {series_id}: {exc}. Filling with NaN.")
            frames[col_name] = pd.Series(dtype=float, name=col_name)

    macro = pd.concat(frames.values(), axis=1, keys=frames.keys())
    macro.index.name = "date"

    if not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)

    # GDP is quarterly → forward-fill to monthly then compute YoY
    if "us_real_gdp" in macro.columns:
        gdp = macro["us_real_gdp"].resample("ME").last().ffill()
        macro = macro.resample("ME").last()
        macro["us_real_gdp"] = gdp
        macro["us_real_gdp_yoy"] = macro["us_real_gdp"].pct_change(periods=12, fill_method=None) * 100
        macro = macro.drop(columns=["us_real_gdp"])
    else:
        macro = macro.resample("ME").last()

    macro = macro.ffill()

    if cache:
        MACRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        macro.to_parquet(MACRO_CACHE_PATH)
        logger.success(f"Macro features cached: {MACRO_CACHE_PATH}")

    return macro


def merge_macro_features(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merges macro features into `df` via merge_asof (backward).

    The macro snapshot used is the latest available *before or on* each loan's
    `date_col`. This prevents look-ahead leakage.
    """
    if macro is None:
        macro = fetch_all_macro()

    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(
            f"Column '{date_col}' not in DataFrame. Cannot merge macro without a date."
        )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    macro_reset = macro.reset_index().rename(columns={"date": date_col})
    macro_reset[date_col] = pd.to_datetime(macro_reset[date_col])
    macro_reset = macro_reset.sort_values(date_col)

    merged = pd.merge_asof(df, macro_reset, on=date_col, direction="backward")
    logger.info(f"Macro features merged on '{date_col}' (merge_asof backward).")
    return merged


if __name__ == "__main__":
    macro_df = fetch_all_macro(cache=True, force=True)
    logger.info(f"\n{macro_df.tail()}")
    logger.info(f"Range: {macro_df.index.min().date()} → {macro_df.index.max().date()}")
