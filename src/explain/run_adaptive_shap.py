"""Driver: produce the adaptive-SHAP artefacts on the frozen LightGBM v2.

Outputs:
  artifacts/adaptive_shap_heatmap.png   — top-K mean|SHAP| month × feature
  artifacts/shap_by_decile_lc.png        — per-risk-decile SHAP top-features
  data/processed/adaptive_shap_monthly.csv
  data/processed/adaptive_shap_by_decile.csv
  data/processed/ridge_surrogate_coefs.csv

Uses the v2 champion (frozen LightGBM at 2013) loaded from
``artifacts/pd_model_lc.joblib`` and the LendingClub features parquet.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.config import ARTIFACTS_DIR, LENDINGCLUB_FEATURES, PROCESSED_DIR
from src.explain.shap_adaptive import (
    IncrementalRidgeSurrogate,
    adaptive_shap_heatmap,
    rebaselined_shap_monthly,
    shap_by_risk_decile,
)
from src.models.pd_model_lc import (
    CATEGORICAL_COLS,
    DATE_COL,
    TARGET_COL,
    load_pd_lc,
)


def main() -> None:
    df = pd.read_parquet(LENDINGCLUB_FEATURES)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, DATE_COL}]
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    model, _calibrator = load_pd_lc()

    # --- Monthly rebaselined SHAP on test period 2014-2017 ----------------
    test = df[df[DATE_COL].dt.year.between(2014, 2017)].reset_index(drop=True)
    t0 = time.time()
    monthly = rebaselined_shap_monthly(
        model,
        test,
        feature_cols=feature_cols,
        date_col=DATE_COL,
        background_window_months=6,
        explain_per_month=1500,
    )
    logger.info(f"Monthly SHAP done in {time.time() - t0:.1f}s, {len(monthly)} rows.")
    monthly.to_csv(PROCESSED_DIR / "adaptive_shap_monthly.csv", index=False)
    adaptive_shap_heatmap(monthly, top_k=12, save_path=ARTIFACTS_DIR / "adaptive_shap_heatmap.png")

    # --- Per-risk-decile SHAP (latest year) -------------------------------
    latest = test[test[DATE_COL].dt.year == 2017].copy()
    latest["p"] = model.predict_proba(latest[feature_cols])[:, 1]
    by_dec = shap_by_risk_decile(
        model, latest, feature_cols=feature_cols, score_col="p", n_deciles=10, explain_per_decile=400
    )
    by_dec.to_csv(PROCESSED_DIR / "adaptive_shap_by_decile.csv", index=False)

    # Plot decile heatmap (decile × feature)
    if not by_dec.empty:
        pivot = by_dec.pivot(index="decile", columns="feature", values="mean_abs_shap")
        top_glob = pivot.mean(axis=0).sort_values(ascending=False).head(12).index.tolist()
        pivot = pivot[top_glob]
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"decile {d}" for d in pivot.index])
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_title("SHAP by predicted-risk decile — 2017 cohort")
        fig.colorbar(im, ax=ax, label="mean |SHAP|")
        plt.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "shap_by_decile_lc.png", dpi=140, bbox_inches="tight")

    # --- Incremental Ridge surrogate --------------------------------------
    surrogate_df = df[df[DATE_COL].dt.year.between(2013, 2017)].copy()
    surrogate_df["p"] = model.predict_proba(surrogate_df[feature_cols])[:, 1]
    ridge = IncrementalRidgeSurrogate(window_months=6, alpha=1.0)
    coefs = ridge.fit_monthly(surrogate_df, feature_cols=feature_cols, score_col="p", date_col=DATE_COL)
    coefs.to_csv(PROCESSED_DIR / "ridge_surrogate_coefs.csv", index=False)
    if not coefs.empty:
        top_abs = coefs.drop(columns=["month"]).abs().mean().sort_values(ascending=False).head(10).index.tolist()
        fig, ax = plt.subplots(figsize=(11, 5))
        for f in top_abs:
            ax.plot(coefs["month"], coefs[f], label=f, lw=1.4)
        ax.set_xticks(range(0, len(coefs), max(1, len(coefs) // 12)))
        ax.set_xticklabels([coefs["month"].iloc[i] for i in range(0, len(coefs), max(1, len(coefs) // 12))],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title("Incremental Ridge surrogate coefficients — top 10 features")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "ridge_surrogate_coefs.png", dpi=140, bbox_inches="tight")

    logger.success("Adaptive SHAP artefacts written.")


if __name__ == "__main__":
    main()
