"""Configurações globais do projeto: paths, logging e thresholds operacionais."""

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

# Cria diretórios necessários ao importar config
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

# ---------------------------------------------------------------------------
# Thresholds operacionais — nunca hardcodar fora daqui
# ---------------------------------------------------------------------------

# PSI (Population Stability Index)
PSI_STABLE = 0.10
PSI_ATTENTION = 0.20  # entre PSI_STABLE e PSI_ATTENTION → atenção

# Early warning: queda de score
SCORE_DROP_THRESHOLD = 50  # pontos
SCORE_DROP_WINDOW_DAYS = 30

# Score contextual: normalização para escala 0–1000
SCORE_MIN = 0
SCORE_MAX = 1000

# ---------------------------------------------------------------------------
# Produtos e prazos válidos (Módulo 3)
# ---------------------------------------------------------------------------
PRODUCTS = ["capital_de_giro", "investimento", "antecipacao_recebiveis"]
TENORS_MONTHS = [1, 3, 6, 12, 24, 36, 48]

# ---------------------------------------------------------------------------
# BCB SGS — IDs das séries macroeconômicas
# ---------------------------------------------------------------------------
BCB_SELIC = 432
BCB_INADIMPLENCIA_PJ = 21082
BCB_IBC_BR = 24364

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TEMPORAL_WINDOWS = [30, 90, 365]  # dias
