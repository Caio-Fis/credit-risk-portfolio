"""Drift and calibration monitoring endpoints.

These read offline-computed artefacts under ``data/processed/`` —
``arf_drifts_lc.csv`` for drift event timeline and ``sliding_calibration_lc.csv``
for rolling calibration metrics. Block D will add live streaming detector
state on top of this read-only API.

Recalibration trigger queues a background task; the runner is wired in
Block D. For now it logs the intent and returns 202 immediately.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException

from src.api.dependencies import get_drift_monitor
from src.api.metrics import RECALIBRATIONS_TOTAL
from src.api.schemas import (
    CalibrationMonitorResponse,
    CalibrationYearly,
    ChampionChallengerResponse,
    ChampionChallengerYearly,
    DriftEvent,
    DriftMonitorResponse,
    LiveDriftResponse,
    RecalibrationTriggerResponse,
    RollingOOTYearly,
    RollingVsFrozenResponse,
)
from src.api.settings import Settings, get_settings
from src.api.state import LiveDriftMonitor
from src.config import PROCESSED_DIR

router = APIRouter(prefix="/v1/monitor", tags=["monitor"])

DRIFT_CSV = PROCESSED_DIR / "arf_drifts_lc.csv"
CALIBRATION_CSV = PROCESSED_DIR / "sliding_calibration_lc.csv"
ARF_YEARLY_CSV = PROCESSED_DIR / "arf_yearly_lc.csv"
ROLLING_OOT_CSV = PROCESSED_DIR / "rolling_oot_lc.csv"
FROZEN_OOT_CSV = PROCESSED_DIR / "frozen_oot_lc.csv"


@router.get("/drift", response_model=DriftMonitorResponse, summary="Drift event timeline (ADWIN + KSWIN)")
def drift_status() -> DriftMonitorResponse:
    if not DRIFT_CSV.exists():
        raise HTTPException(status_code=503, detail=f"Drift artefact missing: {DRIFT_CSV.name}")
    df = pd.read_csv(DRIFT_CSV, parse_dates=["timestamp"])
    by_detector = df.groupby("detector").size().to_dict()
    by_year = df.groupby(df["timestamp"].dt.year.astype(str)).size().to_dict()
    recent = df.sort_values("timestamp", ascending=False).head(20)
    events = [
        DriftEvent(timestamp=row.timestamp, detector=str(row.detector), type=str(row.type))
        for row in recent.itertuples(index=False)
    ]
    return DriftMonitorResponse(
        total_events=len(df),
        by_detector={k: int(v) for k, v in by_detector.items()},
        by_year={k: int(v) for k, v in by_year.items()},
        recent_events=events,
        last_observation=df["timestamp"].max().to_pydatetime() if not df.empty else None,
        source=DRIFT_CSV.name,
    )


@router.get("/drift/live", response_model=LiveDriftResponse, summary="Live drift state (current process)")
def drift_live(drift_mon: LiveDriftMonitor | None = Depends(get_drift_monitor)) -> LiveDriftResponse:
    if drift_mon is None:
        raise HTTPException(status_code=503, detail="Live drift monitor not initialised.")
    snap = drift_mon.snapshot()
    return LiveDriftResponse(**snap)


@router.get(
    "/champion-vs-challenger",
    response_model=ChampionChallengerResponse,
    summary="Yearly metrics for the ARF challenger replay",
)
def champion_vs_challenger() -> ChampionChallengerResponse:
    if not ARF_YEARLY_CSV.exists():
        raise HTTPException(status_code=503, detail=f"ARF yearly artefact missing: {ARF_YEARLY_CSV.name}")
    df = pd.read_csv(ARF_YEARLY_CSV)
    yearly = [
        ChampionChallengerYearly(
            year=int(row.year),
            n_test=int(row.n_test),
            base_rate_test=float(row.base_rate_test),
            auroc=float(row.auroc),
            ks=float(row.ks),
            brier=float(row.brier),
            calib_slope=float(row.calib_slope),
        )
        for row in df.itertuples(index=False)
    ]
    summary = {
        "auroc_mean": float(df["auroc"].mean()),
        "ks_mean": float(df["ks"].mean()),
        "brier_mean": float(df["brier"].mean()),
        "years_covered": int(df["year"].nunique()),
    }
    return ChampionChallengerResponse(
        yearly=yearly,
        summary=summary,
        note=(
            "ARFClassifier (River) replayed in test-then-train with 90d label delay. "
            "Underperforms the static LightGBM champion (mean AUROC ~0.54 vs ~0.65) due to "
            "cold start, the 11-feature signal ceiling, and lack of interaction modelling. "
            "Retained as a drift detector and a data-driven argument for retraining, not as the primary model."
        ),
        source=ARF_YEARLY_CSV.name,
    )


@router.get(
    "/rolling-vs-frozen",
    response_model=RollingVsFrozenResponse,
    summary="LightGBM retrained yearly vs frozen at 2013",
)
def rolling_vs_frozen() -> RollingVsFrozenResponse:
    if not ROLLING_OOT_CSV.exists() or not FROZEN_OOT_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Rolling/frozen artefacts missing: {ROLLING_OOT_CSV.name}, {FROZEN_OOT_CSV.name}",
        )
    rolling_df = pd.read_csv(ROLLING_OOT_CSV)
    frozen_df = pd.read_csv(FROZEN_OOT_CSV)

    def _to_yearly(df: pd.DataFrame) -> list[RollingOOTYearly]:
        return [
            RollingOOTYearly(
                year=int(row.year),
                n_test=int(row.n_test),
                auroc=float(row.auroc),
                ks=float(row.ks),
                brier=float(row.brier),
                calib_slope=float(row.calib_slope),
            )
            for row in df.itertuples(index=False)
        ]

    overlap = rolling_df.merge(frozen_df, on="year", suffixes=("_rolling", "_frozen"))
    summary = {
        "auroc_uplift_pp": float(100 * (overlap["auroc_rolling"] - overlap["auroc_frozen"]).mean()),
        "ks_uplift_pp": float(100 * (overlap["ks_rolling"] - overlap["ks_frozen"]).mean()),
        "brier_delta_pp": float(100 * (overlap["brier_rolling"] - overlap["brier_frozen"]).mean()),
        "years_overlap": int(len(overlap)),
    }
    return RollingVsFrozenResponse(
        rolling=_to_yearly(rolling_df),
        frozen=_to_yearly(frozen_df),
        summary=summary,
        source={"rolling": ROLLING_OOT_CSV.name, "frozen": FROZEN_OOT_CSV.name},
    )


@router.get("/calibration", response_model=CalibrationMonitorResponse, summary="Rolling calibration metrics")
def calibration_status() -> CalibrationMonitorResponse:
    if not CALIBRATION_CSV.exists():
        raise HTTPException(status_code=503, detail=f"Calibration artefact missing: {CALIBRATION_CSV.name}")
    df = pd.read_csv(CALIBRATION_CSV)
    yearly = [CalibrationYearly(**row) for row in df.to_dict(orient="records")]
    summary = {
        "auroc_rolling_mean": float(df["auroc_raw"].mean()),
        "auroc_sliding_mean": float(df["auroc_sliding"].mean()),
        "brier_static_mean": float(df["brier_static"].mean()),
        "brier_sliding_mean": float(df["brier_sliding"].mean()),
        "brier_improvement_pct": float(
            100.0 * (df["brier_static"].mean() - df["brier_sliding"].mean()) / df["brier_static"].mean()
        ),
    }
    mtime = datetime.fromtimestamp(CALIBRATION_CSV.stat().st_mtime)
    return CalibrationMonitorResponse(yearly=yearly, summary=summary, last_refit_at=mtime)


def _do_recalibrate(job_id: str, trigger: str) -> None:
    """Placeholder body — wired to a real worker in Block D."""
    from loguru import logger
    logger.bind(job_id=job_id, trigger=trigger).info("recalibration job started")
    RECALIBRATIONS_TOTAL.labels(trigger=trigger).inc()
    # Block D will: refit SlidingWindowIsotonic over recent labelled data,
    # persist new calibrator, and tick LAST_RECALIBRATION_TIMESTAMP.


@router.post(
    "/recalibrate",
    response_model=RecalibrationTriggerResponse,
    status_code=202,
    summary="Trigger a sliding-window calibrator refit (background)",
)
def trigger_recalibration(
    background_tasks: BackgroundTasks,
    trigger: Annotated[str, Body(embed=True)] = "manual",
    settings: Settings = Depends(get_settings),
) -> RecalibrationTriggerResponse:
    if trigger not in {"manual", "scheduled", "drift"}:
        raise HTTPException(status_code=400, detail=f"Unknown trigger {trigger!r}")
    job_id = uuid.uuid4().hex
    background_tasks.add_task(_do_recalibrate, job_id, trigger)
    return RecalibrationTriggerResponse(
        job_id=job_id,
        status="accepted",
        triggered_at=datetime.utcnow(),
        detail=f"Recalibration enqueued (trigger={trigger}); cadence {settings.recalibration_cadence_days}d.",
    )
