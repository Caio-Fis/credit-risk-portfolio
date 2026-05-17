"""Shared dependencies: model registry + macro lookup cache.

A single ``ModelRegistry`` instance is created at app startup and stored
on ``app.state``. FastAPI dependencies expose it to routes without
relying on module-level globals (cleaner for tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
from fastapi import Request
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.isotonic import IsotonicRegression

from src.api.settings import Settings

CATEGORICAL_COLS = ["purpose", "home_ownership_n", "addr_state", "zip_code"]
FEATURE_ORDER = [
    "revenue", "dti_n", "loan_amnt", "fico_n", "experience_c", "emp_length",
    "purpose", "home_ownership_n", "addr_state", "zip_code",
    "fed_funds_rate", "us_unemployment", "vix_close", "us_10y_treasury", "us_real_gdp_yoy",
]


@dataclass
class ModelRegistry:
    """Holds the loaded artefacts and macro lookup table."""
    model: LGBMClassifier
    calibrator: IsotonicRegression
    macro: pd.DataFrame              # indexed by month-end date
    macro_latest_snapshot: dict[str, float]
    model_loaded_at: datetime
    model_path: Path
    feature_names: list[str]
    train_period: str
    explainer: Any = field(default=None, repr=False)  # shap.TreeExplainer (lazy)

    def lookup_macro(self, issue_d: date) -> dict[str, float]:
        """Backward merge_asof: pick the most recent macro snapshot ≤ issue_d."""
        ts = pd.Timestamp(issue_d)
        idx = self.macro.index[self.macro.index <= ts]
        if len(idx) == 0:
            # Pre-history of macro series; fall back to earliest available
            row = self.macro.iloc[0]
        else:
            row = self.macro.loc[idx[-1]]
        return {c: float(row[c]) for c in self.macro.columns}


def build_registry(settings: Settings) -> ModelRegistry:
    """Loads model + calibrator + macro from disk."""
    logger.info(f"Loading model from {settings.model_path}")
    model = joblib.load(settings.model_path)
    logger.info(f"Loading calibrator from {settings.calibrator_path}")
    calibrator = joblib.load(settings.calibrator_path)

    logger.info(f"Loading macro snapshot from {settings.macro_cache_path}")
    macro = pd.read_parquet(settings.macro_cache_path)
    if not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()
    latest = macro.iloc[-1]
    latest_snapshot = {c: float(latest[c]) for c in macro.columns}

    # LightGBM exposes feature names via booster
    try:
        feature_names = list(model.booster_.feature_name())
    except Exception:
        feature_names = FEATURE_ORDER

    logger.info("Initialising TreeExplainer (tree_path_dependent, no background)")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    registry = ModelRegistry(
        model=model,
        calibrator=calibrator,
        macro=macro,
        macro_latest_snapshot=latest_snapshot,
        model_loaded_at=datetime.utcnow(),
        model_path=settings.model_path,
        feature_names=feature_names,
        train_period="2007-06 → 2014-12 (train); 2015 (val); 2016-2017 (test)",
        explainer=explainer,
    )
    logger.success(
        f"Model registry ready — {len(feature_names)} features; "
        f"macro {macro.index.min().date()}..{macro.index.max().date()}; SHAP explainer cached"
    )
    return registry


def get_registry(request: Request) -> ModelRegistry:
    """Dependency that retrieves the per-app ModelRegistry."""
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("ModelRegistry not initialised — check app lifespan.")
    return registry


def get_drift_monitor(request: Request):  # type: ignore[no-untyped-def]
    """Live drift monitor singleton (see src.api.state.LiveDriftMonitor)."""
    return getattr(request.app.state, "drift_monitor", None)
