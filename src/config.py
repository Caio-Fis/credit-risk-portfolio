"""Global project configuration: paths, logging, and operational thresholds."""

import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMAS_DIR = DATA_DIR / "schemas"
MODELS_DIR = PROJECT_ROOT / "mlruns"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in (RAW_DIR, PROCESSED_DIR, SCHEMAS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging (loguru)
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="INFO",
)
logger.add(
    PROJECT_ROOT / "logs" / "pipeline.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")
MLFLOW_EXPERIMENT_PD = "pd_model"
MLFLOW_EXPERIMENT_LGD = "lgd_model"
MLFLOW_EXPERIMENT_PD_ONLINE = "pd_model_online"

# ---------------------------------------------------------------------------
# Operational thresholds — never hardcode outside this file
# ---------------------------------------------------------------------------

# PSI (Population Stability Index)
PSI_STABLE = 0.10
PSI_ATTENTION = 0.20  # between PSI_STABLE and PSI_ATTENTION → attention

# Early warning: score drop
SCORE_DROP_THRESHOLD = 50  # points
SCORE_DROP_WINDOW_DAYS = 30

# Contextual score: normalisation to 0–1000 scale
SCORE_MIN = 0
SCORE_MAX = 1000

# ---------------------------------------------------------------------------
# Valid products and tenors (Module 3 — contextual, synthetic)
# ---------------------------------------------------------------------------
PRODUCTS = ["working_capital", "investment", "receivables_advance"]
TENORS_MONTHS = [1, 3, 6, 12, 24, 36, 48]

# ---------------------------------------------------------------------------
# LendingClub dataset (Zenodo record 11295916, CC-BY 4.0)
# Loans issued 2007-2018, final status only, ~1.35M rows.
# ---------------------------------------------------------------------------
LENDINGCLUB_URL = "https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv?download=1"
LENDINGCLUB_MD5 = "b019384d6bc65bf2a3e839362e4ff502"
LENDINGCLUB_RAW = RAW_DIR / "lendingclub" / "loans.csv"
LENDINGCLUB_PARQUET = PROCESSED_DIR / "lendingclub_raw.parquet"
LENDINGCLUB_FEATURES = PROCESSED_DIR / "lendingclub_features.parquet"
LENDINGCLUB_SCHEMA = SCHEMAS_DIR / "lendingclub.json"

# ---------------------------------------------------------------------------
# FRED — US macroeconomic series (no API key, CSV endpoint)
# Aligned to LendingClub time range 2007-2018.
# ---------------------------------------------------------------------------
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",       # monthly, %
    "UNRATE": "us_unemployment",        # monthly, %
    "GDPC1": "us_real_gdp",             # quarterly, billions chained 2017$ → YoY derived
    "VIXCLS": "vix_close",              # daily, index → resampled monthly
    "DGS10": "us_10y_treasury",         # daily, % → resampled monthly
}
MACRO_CACHE_PATH = PROCESSED_DIR / "macro_features.parquet"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TEMPORAL_WINDOWS = [30, 90, 365]  # days

# ---------------------------------------------------------------------------
# Online learning — challenger configuration
# ---------------------------------------------------------------------------
ONLINE_LABEL_DELAY_DAYS = 90      # simulated delay before label is available
ONLINE_DRIFT_ADWIN_DELTA = 0.002  # ADWIN sensitivity (lower = more sensitive)
ONLINE_DRIFT_KSWIN_ALPHA = 0.005  # KSWIN significance level
ONLINE_CALIB_WINDOW_MONTHS = 6    # sliding window for isotonic recalibration
