"""Service configuration via environment variables.

Pydantic BaseSettings keeps env vars typed and discoverable. Override via
``CREDIT_RISK_<FIELD>`` env vars (e.g. ``CREDIT_RISK_MODEL_PATH=/foo``).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import (
    ARTIFACTS_DIR,
    LENDINGCLUB_FEATURES,
    MACRO_CACHE_PATH,
)


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CREDIT_RISK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------- Service identity --------
    service_name: str = "credit-risk-api"
    api_version: str = "v1"
    environment: str = Field("development", description="dev / staging / production")

    # -------- Model artifacts --------
    model_path: Path = ARTIFACTS_DIR / "pd_model_lc.joblib"
    calibrator_path: Path = ARTIFACTS_DIR / "pd_calibrator_lc.joblib"
    macro_cache_path: Path = MACRO_CACHE_PATH
    feature_store_path: Path = LENDINGCLUB_FEATURES

    # -------- API behaviour --------
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout_seconds: int = 30

    # -------- Observability --------
    enable_metrics: bool = True
    log_level: str = "INFO"

    # -------- Drift monitoring --------
    drift_state_path: Path = ARTIFACTS_DIR / "drift_state.json"
    recalibration_cadence_days: int = 7


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Memoised settings accessor. Use as FastAPI dependency."""
    return Settings()
