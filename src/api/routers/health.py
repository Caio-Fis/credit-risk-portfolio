"""Liveness, readiness, and service version endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, Response

from src.api import metrics
from src.api.dependencies import ModelRegistry, get_registry
from src.api.schemas import HealthResponse, ModelInfoResponse
from src.api.settings import Settings, get_settings

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        service=settings.service_name,
        version=settings.api_version,
    )


@router.get("/version", response_model=HealthResponse, summary="Alias for /health, kept for backwards compatibility")
def version(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return health(settings)


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)


@router.get("/v1/models/info", response_model=ModelInfoResponse, summary="Loaded model metadata")
def model_info(registry: ModelRegistry = Depends(get_registry)) -> ModelInfoResponse:
    # OOT metrics are persisted from the training run — for now we surface
    # the headline numbers documented in README. A future task can read
    # mlflow runs directly.
    return ModelInfoResponse(
        name="pd_model_lc (LightGBM + isotonic)",
        version=registry.model_path.stem,
        trained_at=registry.model_loaded_at,
        train_period=registry.train_period,
        feature_count=len(registry.feature_names),
        metrics={
            "auroc_test_calibrated": 0.6548,
            "brier_test_calibrated": 0.1697,
            "rolling_oot_2014_2017_mean": 0.6534,
            "frozen_at_2013_2014_2017_mean": 0.6277,
        },
    )
