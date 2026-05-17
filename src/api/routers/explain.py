"""On-demand SHAP explanation for a single loan application.

Uses the registry's cached ``TreeExplainer`` (tree_path_dependent — the
only mode that handles LightGBM categorical splits). Returns per-feature
SHAP contributions plus a top-5 driver list ranked by absolute impact.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from src.api.dependencies import (
    CATEGORICAL_COLS,
    ModelRegistry,
    get_registry,
)
from src.api.schemas import (
    ExplanationResponse,
    FeatureContribution,
    LoanFeatures,
)

router = APIRouter(prefix="/v1", tags=["explain"])


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
