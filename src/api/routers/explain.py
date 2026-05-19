"""On-demand SHAP explanation for a single loan application.

Uses the registry's cached ``TreeExplainer`` (tree_path_dependent — the
only mode that handles LightGBM categorical splits). Returns per-feature
SHAP contributions plus a top-5 driver list ranked by absolute impact.

Also exposes the offline-computed adaptive SHAP surfaces (monthly heatmap,
per-decile attribution, ridge surrogate coefficients) replicating
arXiv:2511.03807.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    CATEGORICAL_COLS,
    ModelRegistry,
    get_registry,
)
from src.api.schemas import (
    AdaptiveShapResponse,
    ExplanationResponse,
    FeatureContribution,
    LoanFeatures,
    RidgeCoefRow,
    ShapDecileCell,
    ShapHeatmapCell,
)
from src.config import PROCESSED_DIR

router = APIRouter(prefix="/v1", tags=["explain"])

ADAPTIVE_SHAP_MONTHLY_CSV = PROCESSED_DIR / "adaptive_shap_monthly.csv"
ADAPTIVE_SHAP_DECILE_CSV = PROCESSED_DIR / "adaptive_shap_by_decile.csv"
RIDGE_SURROGATE_CSV = PROCESSED_DIR / "ridge_surrogate_coefs.csv"
ADAPTIVE_SHAP_TOP_N = 12
ADAPTIVE_SHAP_REFERENCES = [
    "arXiv:2511.03807 — Adaptive SHAP with rebased background",
    "Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions",
]


def _build_single_row(loan: LoanFeatures, registry: ModelRegistry) -> tuple[pd.DataFrame, date, dict[str, float]]:
    used_date = loan.issue_d or date.today()
    macro = registry.lookup_macro(used_date)
    row = loan.model_dump(exclude={"issue_d"})
    row["purpose"] = row["purpose"].value if hasattr(row["purpose"], "value") else row["purpose"]
    row["home_ownership_n"] = row["home_ownership_n"].value if hasattr(row["home_ownership_n"], "value") else row["home_ownership_n"]
    if row["emp_length"] is None:
        row["emp_length"] = np.nan
    row.update(macro)
    df = pd.DataFrame([row]).reindex(columns=registry.feature_names, fill_value=np.nan)
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df, used_date, macro


@router.post("/explain", response_model=ExplanationResponse, summary="SHAP waterfall for a single loan")
def explain_one(
    loan: LoanFeatures,
    registry: ModelRegistry = Depends(get_registry),
) -> ExplanationResponse:
    df, used_date, _ = _build_single_row(loan, registry)
    p_raw = float(registry.model.predict_proba(df)[0, 1])
    p_cal = float(np.clip(registry.calibrator.transform([p_raw])[0], 0.0, 1.0))

    explanation = registry.explainer(df)
    vals = explanation.values
    base = explanation.base_values
    if vals.ndim == 3:                 # multi-class output
        vals = vals[0, :, 1]
        base_val = float(base[0, 1] if base.ndim > 1 else base[0])
    else:
        vals = vals[0]
        base_val = float(base[0] if hasattr(base, "__len__") else base)

    contributions = []
    for feature, value, shap_val in zip(registry.feature_names, df.iloc[0].tolist(), vals.tolist()):
        # Categorical/nan-safe value coercion for the response
        if value is None or (isinstance(value, float) and np.isnan(value)):
            v_out: float | str | None = None
        elif isinstance(value, (int, float, np.integer, np.floating)):
            v_out = float(value)
        else:
            v_out = str(value)
        contributions.append(
            FeatureContribution(
                feature=feature,
                value=v_out,
                shap_value=float(shap_val),
                direction="risk_up" if shap_val > 0 else "risk_down",
            )
        )
    top = sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)[:5]

    return ExplanationResponse(
        pd_calibrated=p_cal,
        pd_raw=p_raw,
        base_value=base_val,
        contributions=contributions,
        top_drivers=top,
        model_version=registry.model_path.stem,
        issue_d_used=used_date,
    )


@router.get(
    "/explain/adaptive-shap",
    response_model=AdaptiveShapResponse,
    summary="Adaptive SHAP surfaces — monthly heatmap, per-decile attribution, ridge surrogate",
)
def adaptive_shap() -> AdaptiveShapResponse:
    missing = [p.name for p in (ADAPTIVE_SHAP_MONTHLY_CSV, ADAPTIVE_SHAP_DECILE_CSV, RIDGE_SURROGATE_CSV) if not p.exists()]
    if missing:
        raise HTTPException(status_code=503, detail=f"Adaptive SHAP artefacts missing: {', '.join(missing)}")

    monthly_df = pd.read_csv(ADAPTIVE_SHAP_MONTHLY_CSV)
    decile_df = pd.read_csv(ADAPTIVE_SHAP_DECILE_CSV)
    ridge_df = pd.read_csv(RIDGE_SURROGATE_CSV)

    top_features = (
        monthly_df.groupby("feature")["mean_abs_shap"].mean()
        .sort_values(ascending=False)
        .head(ADAPTIVE_SHAP_TOP_N)
        .index.tolist()
    )
    heatmap_df = monthly_df[monthly_df["feature"].isin(top_features)]
    months = sorted(heatmap_df["month"].unique().tolist())

    by_decile_df = decile_df[decile_df["feature"].isin(top_features)]
    deciles = sorted(int(d) for d in by_decile_df["decile"].unique())

    ridge_features = [c for c in ridge_df.columns if c != "month"]
    ridge_rows = [
        RidgeCoefRow(
            month=str(row["month"]),
            coefs={f: float(row[f]) for f in ridge_features},
        )
        for _, row in ridge_df.sort_values("month").iterrows()
    ]

    heatmap = [
        ShapHeatmapCell(
            month=str(row.month),
            feature=str(row.feature),
            mean_abs_shap=float(row.mean_abs_shap),
        )
        for row in heatmap_df.itertuples(index=False)
    ]
    by_decile = [
        ShapDecileCell(
            decile=int(row.decile),
            feature=str(row.feature),
            mean_abs_shap=float(row.mean_abs_shap),
        )
        for row in by_decile_df.itertuples(index=False)
    ]

    return AdaptiveShapResponse(
        heatmap=heatmap,
        by_decile=by_decile,
        ridge_surrogate=ridge_rows,
        top_features=top_features,
        months=months,
        deciles=deciles,
        references=ADAPTIVE_SHAP_REFERENCES,
    )
