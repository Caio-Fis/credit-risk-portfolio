"""FastAPI application factory + uvicorn entrypoint.

Use the factory pattern so tests can build the app with overridden
settings without monkey-patching globals.

Run locally::

    uv run uvicorn src.api.main:app --reload
    # or
    uv run credit-risk-api
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from src.api.dependencies import build_registry
from src.api.logging_config import configure_logging
from src.api.metrics import MODEL_LOADED_TIMESTAMP
from src.api.middleware import RequestContextMiddleware
from src.api.routers import explain, health, monitor, predict
from src.api.settings import Settings, get_settings
from src.api.state import (
    NUMERIC_FEATURES_TO_MONITOR,
    LiveDriftMonitor,
    build_psi_reference,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts at startup; release on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_format=settings.environment != "development")
    logger.info(
        f"Starting {settings.service_name} ({settings.environment}, api {settings.api_version})"
    )
    registry = build_registry(settings)
    MODEL_LOADED_TIMESTAMP.set(registry.model_loaded_at.timestamp())
    app.state.registry = registry
    app.state.settings = settings

    # Live drift monitor — fed by every prediction
    psi_ref = build_psi_reference(settings.feature_store_path, NUMERIC_FEATURES_TO_MONITOR)
    monitor_obj = LiveDriftMonitor(state_path=settings.drift_state_path)
    monitor_obj.initialise(psi_reference=psi_ref or None)
    app.state.drift_monitor = monitor_obj

    try:
        yield
    finally:
        logger.info("Shutting down — persisting drift state and releasing registry")
        try:
            monitor_obj.persist()
        except Exception as exc:
            logger.warning(f"Drift state persistence failed: {exc}")
        app.state.registry = None
        app.state.drift_monitor = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory for the FastAPI app. Tests may override settings."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Credit Risk PD API",
        version=settings.api_version,
        description=(
            "Calibrated probability of default with drift-aware adaptive "
            "explainability. v2 of the credit-risk-portfolio project."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect bare-URL visits to the Swagger UI."""
        return RedirectResponse(url="/docs", status_code=307)

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(monitor.router)
    return app


app = create_app()


def run() -> None:
    """uvicorn entrypoint registered as the `credit-risk-api` script."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=get_settings().log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    run()
