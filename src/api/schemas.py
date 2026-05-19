"""Pydantic request / response models.

Schemas mirror ``data/schemas/lendingclub.json`` for input validation and
expose calibrated probabilities + metadata for downstream consumers.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums — mirror schema enums
# ---------------------------------------------------------------------------
class Purpose(str, Enum):
    debt_consolidation = "debt_consolidation"
    credit_card = "credit_card"
    home_improvement = "home_improvement"
    other = "other"
    major_purchase = "major_purchase"
    medical = "medical"
    small_business = "small_business"
    car = "car"


class HomeOwnership(str, Enum):
    mortgage = "MORTGAGE"
    rent = "RENT"
    own = "OWN"
    other = "OTHER"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class LoanFeatures(BaseModel):
    """Borrower-side features available at loan origination.

    Macro context (``fed_funds_rate``, ``us_unemployment``, etc.) is merged
    server-side from FRED based on ``issue_d``. Clients should not supply
    macro fields — they are derived.
    """

    model_config = ConfigDict(extra="forbid")

    revenue: Annotated[float, Field(ge=0, description="Annual income, USD.")]
    dti_n: Annotated[float, Field(ge=0, le=999, description="Debt-to-income ratio. 999 used as sentinel for missing.")]
    loan_amnt: Annotated[float, Field(ge=500, le=40_000, description="Requested loan amount, USD.")]
    fico_n: Annotated[float, Field(ge=300, le=850, description="FICO score at application.")]
    experience_c: Annotated[int, Field(ge=0, le=1, description="LendingClub experience class indicator.")]
    emp_length: Annotated[float | None, Field(ge=0, le=10, description="Employment length in years (None if not reported).")] = None
    purpose: Purpose
    home_ownership_n: HomeOwnership
    addr_state: Annotated[str, Field(min_length=2, max_length=2, description="US two-letter state code.")]
    zip_code: Annotated[str, Field(min_length=3, max_length=6, description="First 3 digits of ZIP (LendingClub format, e.g. '350xx').")]
    issue_d: date | None = Field(None, description="Origination date for macro lookup. Defaults to today.")


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loans: Annotated[list[LoanFeatures], Field(min_length=1, max_length=1000)]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Calibrated probability of default + transparency metadata."""

    pd_calibrated: Annotated[float, Field(ge=0, le=1, description="Sliding-window calibrated default probability.")]
    pd_raw: Annotated[float, Field(ge=0, le=1, description="Raw LightGBM probability (pre-calibration).")]
    score_0_1000: Annotated[int, Field(ge=0, le=1000, description="Convenience score: round(1000 * (1 - pd_calibrated)).")]
    risk_band: str = Field(description="Bucket: low / medium / high / very_high.")
    model_version: str
    issue_d_used: date
    macro_snapshot: dict[str, float] = Field(description="Macro values merged for this prediction.")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Service metadata
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    trained_at: datetime | None
    train_period: str = Field(description="Train slice used by the static champion (e.g. '2007-06 → 2014-12').")
    feature_count: int
    metrics: dict[str, float]
    references: list[str] = Field(
        default_factory=lambda: [
            "Gomes et al. 2017 (ARF)",
            "Shivogo John 2025 — Fair and Explainable Credit-Scoring under Concept Drift (arXiv:2511.03807)",
            "Lundberg & Lee 2017 (SHAP)",
        ]
    )


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    feature: str
    value: float | str | None
    shap_value: float
    direction: str = Field(description="'risk_up' if shap_value > 0, else 'risk_down'.")


class ExplanationResponse(BaseModel):
    pd_calibrated: float
    pd_raw: float
    base_value: float = Field(description="Model's expected log-odds before applying any feature contribution.")
    contributions: list[FeatureContribution]
    top_drivers: list[FeatureContribution] = Field(description="Top 5 by |shap_value|.")
    model_version: str
    issue_d_used: date


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
class DriftEvent(BaseModel):
    timestamp: datetime
    detector: str  # 'ADWIN' | 'KSWIN'
    type: str      # 'error' | 'score'


class DriftMonitorResponse(BaseModel):
    total_events: int
    by_detector: dict[str, int]
    by_year: dict[str, int]
    recent_events: list[DriftEvent] = Field(description="Last 20 events, newest first.")
    last_observation: datetime | None
    source: str = Field(description="Stream replay this state was computed from (e.g. 'arf_drifts_lc.csv').")


class LiveDriftEvent(BaseModel):
    timestamp: datetime
    detector: str
    type: str
    value: float | None = None


class LiveDriftResponse(BaseModel):
    samples_seen: int
    started_at: datetime
    score_buffer_size: int
    events_total: int
    events_by_detector: dict[str, int]
    events_recent: list[LiveDriftEvent]
    last_psi: dict[str, float] = Field(
        default_factory=dict,
        description="Last PSI value per monitored numeric feature (PSI > 0.20 fires a drift event).",
    )


class CalibrationYearly(BaseModel):
    year: int
    auroc_raw: float
    auroc_static: float
    auroc_sliding: float
    brier_raw: float
    brier_static: float
    brier_sliding: float
    slope_static: float
    slope_sliding: float


class CalibrationMonitorResponse(BaseModel):
    yearly: list[CalibrationYearly]
    summary: dict[str, float] = Field(description="Mean across rolling years for headline metrics.")
    last_refit_at: datetime | None


class RecalibrationTriggerResponse(BaseModel):
    job_id: str
    status: str = Field(description="'accepted' | 'rejected' | 'running'")
    triggered_at: datetime
    detail: str | None = None


# ---------------------------------------------------------------------------
# Champion vs challenger (ARF replay)
# ---------------------------------------------------------------------------
class ChampionChallengerYearly(BaseModel):
    year: int
    n_test: int
    base_rate_test: float
    auroc: float
    ks: float
    brier: float
    calib_slope: float


class ChampionChallengerResponse(BaseModel):
    yearly: list[ChampionChallengerYearly]
    summary: dict[str, float] = Field(
        description="Mean AUROC/KS/Brier across years for the ARF challenger replay."
    )
    note: str = Field(
        description="Editorial note: ARF underperforms the LightGBM champion on this dataset (~0.54 vs ~0.65 AUROC). Kept as drift detector, not primary model."
    )
    source: str


# ---------------------------------------------------------------------------
# Rolling vs frozen baseline (LightGBM retrained yearly vs frozen 2013)
# ---------------------------------------------------------------------------
class RollingOOTYearly(BaseModel):
    year: int
    n_test: int
    auroc: float
    ks: float
    brier: float
    calib_slope: float


class RollingVsFrozenResponse(BaseModel):
    rolling: list[RollingOOTYearly] = Field(description="LightGBM retrained yearly, evaluated on next year.")
    frozen: list[RollingOOTYearly] = Field(description="LightGBM frozen at 2013 cut, evaluated on later years.")
    summary: dict[str, float] = Field(description="Mean uplift of rolling over frozen (AUROC, KS, Brier).")
    source: dict[str, str]


# ---------------------------------------------------------------------------
# Adaptive SHAP (offline-computed: heatmap + per-decile + ridge surrogate)
# ---------------------------------------------------------------------------
class ShapHeatmapCell(BaseModel):
    month: str = Field(description="YYYY-MM bucket.")
    feature: str
    mean_abs_shap: float


class ShapDecileCell(BaseModel):
    decile: int = Field(ge=0, le=9, description="Risk decile (0 = lowest predicted PD, 9 = highest).")
    feature: str
    mean_abs_shap: float


class RidgeCoefRow(BaseModel):
    month: str
    coefs: dict[str, float]


class AdaptiveShapResponse(BaseModel):
    heatmap: list[ShapHeatmapCell] = Field(
        description="Mean |SHAP| per (month, feature). Background rebased monthly."
    )
    by_decile: list[ShapDecileCell] = Field(
        description="Per-decile SHAP attribution for the latest scoring window."
    )
    ridge_surrogate: list[RidgeCoefRow] = Field(
        description="Monthly incremental Ridge surrogate coefficients in logit space."
    )
    top_features: list[str] = Field(description="Top features by overall mean |SHAP|.")
    months: list[str] = Field(description="Sorted month buckets present in the heatmap.")
    deciles: list[int] = Field(description="Sorted deciles present in by_decile.")
    references: list[str] = Field(
        default_factory=list,
        description="Papers/methods this surface replicates.",
    )


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
