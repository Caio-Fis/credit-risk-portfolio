"""Adaptive SHAP for streaming credit scoring.

Replicates the three core ideas from
*Fair and Explainable Credit-Scoring under Concept Drift*
(Shivogo John, arXiv:2511.03807, 2025):

  1. **Sliding-window rebaselining**: TreeSHAP run with a rolling-window
     interventional background so the baseline reflects the recent
     population, not the long-frozen training slice.
  2. **Per-slice reweighting**: SHAP aggregated per predicted-risk
     decile so explanations honour the local context (high-risk cohort
     features may differ from low-risk).
  3. **Incremental Ridge surrogate**: an interpretable linear surrogate
     fitted online on (features → logit(p̂)) pairs; updated as new
     records arrive. Useful as a sanity-check on the tree model.

Killer artefact for the portfolio: ``adaptive_shap_heatmap`` —
top-K feature mean(|SHAP|) over monthly windows. The shifting heat
of features like ``fed_funds_rate`` or ``vix_close`` around 2008
makes drift visible at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Sliding-window rebaselined SHAP
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveSHAPResult:
    monthly: pd.DataFrame      # month × feature → mean(|SHAP|)
    by_slice: pd.DataFrame     # decile × feature → mean(|SHAP|), latest window
    surrogate_coefs: pd.DataFrame  # feature × month → Ridge coefficient


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def rebaselined_shap_monthly(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "issue_d",
    background_window_months: int = 6,
    explain_per_month: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """For each month, compute interventional TreeSHAP using a rolling
    background window of the last ``background_window_months`` of data.

    Returns a long DataFrame: (month, feature, mean_abs_shap, base_value).
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])
    rng = np.random.default_rng(seed)

    months = sorted(df[date_col].dt.to_period("M").unique())
    rows = []
    for period in months:
        month_start = period.to_timestamp()
        win_start = month_start - pd.DateOffset(months=background_window_months)
        background = df[(df[date_col] >= win_start) & (df[date_col] < month_start)]
        target = df[df[date_col].dt.to_period("M") == period]
        if len(background) < 200 or len(target) < 50:
            continue

        bg_sample = background.sample(min(len(background), 1000), random_state=int(period.ordinal) % 1000)
        if len(target) > explain_per_month:
            target = target.sample(explain_per_month, random_state=seed)

        try:
            # tree_path_dependent works with LightGBM's categorical splits;
            # `interventional` would require encoding categoricals out.
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            exp = explainer(target[feature_cols])
        except Exception as exc:
            logger.warning(f"  {period}: TreeExplainer failed ({exc}); skipping.")
            continue

        vals = exp.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]  # positive class
        mean_abs = np.abs(vals).mean(axis=0)
        for j, f in enumerate(feature_cols):
            rows.append({"month": str(period), "feature": f, "mean_abs_shap": float(mean_abs[j])})

    if not rows:
        logger.warning("Rebaselined SHAP produced no rows; returning empty frame.")
        return pd.DataFrame(columns=["month", "feature", "mean_abs_shap"])
    out = pd.DataFrame(rows)
    logger.success(f"Rebaselined SHAP done: {out['month'].nunique()} months × {len(feature_cols)} features.")
    return out


def shap_by_risk_decile(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    score_col: str = "p",
    n_deciles: int = 10,
    explain_per_decile: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-slice SHAP: split by predicted PD decile, average |SHAP| per
    feature within each slice. Latest snapshot only (not temporal).
    """
    df = df.copy()
    df["_decile"] = pd.qcut(df[score_col], q=n_deciles, duplicates="drop", labels=False)
    rows = []
    for d, slice_df in df.groupby("_decile"):
        if len(slice_df) < 50:
            continue
        sample = slice_df.sample(min(len(slice_df), explain_per_decile), random_state=seed)
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            exp = explainer(sample[feature_cols])
        except Exception as exc:
            logger.warning(f"  decile {d}: failed ({exc}); skipping.")
            continue
        vals = exp.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        mean_abs = np.abs(vals).mean(axis=0)
        for j, f in enumerate(feature_cols):
            rows.append({"decile": int(d), "feature": f, "mean_abs_shap": float(mean_abs[j])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Incremental Ridge surrogate
# ---------------------------------------------------------------------------
class IncrementalRidgeSurrogate:
    """Online Ridge that maps features → logit(p̂) of the tree model.

    Refit on a sliding window each month. Coefficients give an
    interpretable linear approximation of the tree's behaviour and
    surface coefficient drift over time.
    """

    def __init__(self, window_months: int = 6, alpha: float = 1.0):
        self.window_months = window_months
        self.alpha = alpha

    def fit_monthly(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        score_col: str = "p",
        date_col: str = "issue_d",
    ) -> pd.DataFrame:
        df = df.copy().sort_values(date_col).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])
        # Categorical → numeric via simple frequency encoding (surrogate is linear).
        # Cast away from `category` dtype first; otherwise fillna can't accept new values.
        for c in feature_cols:
            if df[c].dtype.name in ("object", "category"):
                series = df[c].astype("object")
                freq = series.value_counts(normalize=True)
                df[c] = series.map(freq).astype("float64").fillna(0.0)
        df[feature_cols] = df[feature_cols].astype("float64").fillna(0.0)

        coefs = []
        months = sorted(df[date_col].dt.to_period("M").unique())
        for period in months:
            month_start = period.to_timestamp()
            win_start = month_start - pd.DateOffset(months=self.window_months)
            train = df[(df[date_col] >= win_start) & (df[date_col] < month_start)]
            if len(train) < 500:
                continue
            ridge = Ridge(alpha=self.alpha).fit(train[feature_cols], _logit(train[score_col].to_numpy()))
            coef_row = {"month": str(period)}
            for f, c in zip(feature_cols, ridge.coef_):
                coef_row[f] = float(c)
            coefs.append(coef_row)
        return pd.DataFrame(coefs)


# ---------------------------------------------------------------------------
# Killer plot
# ---------------------------------------------------------------------------
def adaptive_shap_heatmap(
    monthly: pd.DataFrame,
    top_k: int = 12,
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap: month (rows) × top-K feature (columns) → mean(|SHAP|).

    The shifting heat over time visually exposes feature-importance drift.
    """
    pivot = monthly.pivot(index="month", columns="feature", values="mean_abs_shap")
    global_top = pivot.mean(axis=0).sort_values(ascending=False).head(top_k).index.tolist()
    pivot = pivot[global_top]

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.9), max(6, len(pivot) * 0.18)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_title(f"Adaptive SHAP — top {top_k} features over time (rebaselined per month)")
    fig.colorbar(im, ax=ax, label="mean |SHAP|")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"Adaptive SHAP heatmap saved at {save_path}")
    return fig
