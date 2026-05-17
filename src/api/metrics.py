"""Prometheus collectors exposed at ``/metrics``.

Three categories:
  * HTTP-level: request count, latency histogram, in-flight gauge.
  * ML-level: predictions count, calibrated-PD distribution, drift events.
  * State: model loaded timestamp, last recalibration timestamp.

Collectors are module-level singletons (Prometheus's expected pattern);
imports are idempotent.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# HTTP-level metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "credit_risk_requests_total",
    "Total HTTP requests handled.",
    labelnames=("route", "method", "status"),
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "credit_risk_request_latency_seconds",
    "End-to-end request latency in seconds.",
    labelnames=("route", "method"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
REQUEST_IN_FLIGHT = Gauge(
    "credit_risk_requests_in_flight",
    "Number of requests currently being processed.",
    labelnames=("route", "method"),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# ML-level metrics
# ---------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "credit_risk_predictions_total",
    "Total predictions issued by /v1/predict and /v1/predict/batch.",
    labelnames=("endpoint",),
    registry=REGISTRY,
)
PREDICTION_PD = Histogram(
    "credit_risk_prediction_pd_calibrated",
    "Distribution of calibrated default probabilities returned.",
    buckets=(0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80),
    registry=REGISTRY,
)
DRIFT_EVENTS_TOTAL = Counter(
    "credit_risk_drift_events_total",
    "Drift detections raised at runtime by ADWIN / KSWIN / PSI.",
    labelnames=("detector",),
    registry=REGISTRY,
)
RECALIBRATIONS_TOTAL = Counter(
    "credit_risk_recalibrations_total",
    "Number of sliding-window recalibrations executed.",
    labelnames=("trigger",),  # 'manual' | 'scheduled' | 'drift'
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# State gauges
# ---------------------------------------------------------------------------
MODEL_LOADED_TIMESTAMP = Gauge(
    "credit_risk_model_loaded_timestamp_seconds",
    "Unix timestamp of the most recent model load.",
    registry=REGISTRY,
)
LAST_RECALIBRATION_TIMESTAMP = Gauge(
    "credit_risk_last_recalibration_timestamp_seconds",
    "Unix timestamp of the most recent calibrator refit.",
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """Return ``(payload, content_type)`` for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
