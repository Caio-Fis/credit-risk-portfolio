"""Prediction endpoints — single + batch.

The model expects 15 features in a specific order with two categoricals
that must carry the trained category dtype. The router (a) merges macro
from the registry's cached FRED snapshot keyed by ``issue_d``, (b) builds
a DataFrame in the trained column order, (c) coerces categoricals.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from src.api.dependencies import (
    CATEGORICAL_COLS,
    ModelRegistry,
    get_drift_monitor,
    get_registry,
)
from src.api.metrics import PREDICTION_PD, PREDICTIONS_TOTAL
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    LoanFeatures,
    PredictionResponse,
)
from src.api.state import LiveDriftMonitor

router = APIRouter(prefix="/v1", tags=["predict"])

RISK_BANDS = [
    (0.05, "low"),
    (0.10, "medium"),
    (0.20, "high"),
    (1.01, "very_high"),
]


def _risk_band(pd_calibrated: float) -> str:
    for threshold, label in RISK_BANDS:
        if pd_calibrated < threshold:
            return label
    return "very_high"


def _build_feature_frame(
    loans: list[LoanFeatures],
    registry: ModelRegistry,
) -> tuple[pd.DataFrame, list[dict[str, float]], list[date]]:
    """Assemble the DataFrame the model expects, merging macro per-row."""
    macro_snapshots: list[dict[str, float]] = []
    issue_dates: list[date] = []
    rows: list[dict] = []

    today = date.today()
    for loan in loans:
        used_date = loan.issue_d or today
        issue_dates.append(used_date)
        macro = registry.lookup_macro(used_date)
        macro_snapshots.append(macro)

        row = loan.model_dump(exclude={"issue_d"})
        # Enum fields → string
        row["purpose"] = row["purpose"].value if hasattr(row["purpose"], "value") else row["purpose"]
        row["home_ownership_n"] = row["home_ownership_n"].value if hasattr(row["home_ownership_n"], "value") else row["home_ownership_n"]
        if row["emp_length"] is None:
            row["emp_length"] = np.nan
        row.update(macro)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Order columns exactly as model expects, then enforce category dtype
    df = df.reindex(columns=registry.feature_names, fill_value=np.nan)
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df, macro_snapshots, issue_dates


def _predict_batch(
    loans: list[LoanFeatures],
    registry: ModelRegistry,
) -> list[PredictionResponse]:
    df, macro_snapshots, issue_dates = _build_feature_frame(loans, registry)
    raw = registry.model.predict_proba(df)[:, 1]
    calibrated = registry.calibrator.transform(raw)

    responses = []
    for i, p_cal in enumerate(calibrated):
        p_cal_f = float(np.clip(p_cal, 0.0, 1.0))
        p_raw_f = float(np.clip(raw[i], 0.0, 1.0))
        responses.append(
            PredictionResponse(
                pd_calibrated=p_cal_f,
                pd_raw=p_raw_f,
                score_0_1000=int(round(1000 * (1 - p_cal_f))),
                risk_band=_risk_band(p_cal_f),
                model_version=registry.model_path.stem,
                issue_d_used=issue_dates[i],
                macro_snapshot=macro_snapshots[i],
            )
        )
    return responses


def _record_prediction_metrics(preds: list[PredictionResponse], endpoint: str) -> None:
    PREDICTIONS_TOTAL.labels(endpoint=endpoint).inc(len(preds))
    for p in preds:
        PREDICTION_PD.observe(p.pd_calibrated)


def _feed_drift(
    preds: list[PredictionResponse],
    loans: list[LoanFeatures],
    drift_mon: LiveDriftMonitor | None,
) -> None:
    if drift_mon is None:
        return
    for pred, loan in zip(preds, loans):
        feature_row = loan.model_dump(exclude={"issue_d"})
        feature_row.update(pred.macro_snapshot)
        drift_mon.observe(pred.pd_calibrated, feature_row)


@router.post("/predict", response_model=PredictionResponse, summary="Calibrated PD for one loan application")
def predict_one(
    loan: LoanFeatures,
    registry: ModelRegistry = Depends(get_registry),
    drift_mon: LiveDriftMonitor | None = Depends(get_drift_monitor),
) -> PredictionResponse:
    preds = _predict_batch([loan], registry)
    _record_prediction_metrics(preds, endpoint="predict")
    _feed_drift(preds, [loan], drift_mon)
    return preds[0]


@router.post("/predict/batch", response_model=BatchPredictionResponse, summary="Calibrated PD for up to 1000 loans")
def predict_batch(
    body: BatchPredictionRequest,
    registry: ModelRegistry = Depends(get_registry),
    drift_mon: LiveDriftMonitor | None = Depends(get_drift_monitor),
) -> BatchPredictionResponse:
    t0 = time.perf_counter()
    preds = _predict_batch(body.loans, registry)
    _record_prediction_metrics(preds, endpoint="predict_batch")
    _feed_drift(preds, body.loans, drift_mon)
    latency = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(predictions=preds, n=len(preds), latency_ms=round(latency, 2))
