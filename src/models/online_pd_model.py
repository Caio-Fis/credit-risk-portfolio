"""Online PD challenger — Adaptive Random Forest (River) on LendingClub.

This is the v2 challenger. It learns sample-by-sample in chronological
``issue_d`` order with two production-realistic constraints:

  1. **Test-then-train**: every record is first predicted (out-of-sample),
     then used to update the model. The prediction is what enters the metric.
  2. **Label delay**: defaults are not observed at origination — only after
     ``ONLINE_LABEL_DELAY_DAYS`` (default 90d). A record's label is queued
     and released to ``learn_one`` only when the simulated clock crosses
     ``issue_d + delay``.

Yearly windows on the prediction timeline yield a metric set directly
comparable with ``rolling_oot_evaluation`` / ``frozen_oot_evaluation``
in ``src.evaluate.metrics``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from river import drift, forest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import (
    LENDINGCLUB_FEATURES,
    ONLINE_DRIFT_ADWIN_DELTA,
    ONLINE_DRIFT_KSWIN_ALPHA,
    ONLINE_LABEL_DELAY_DAYS,
)

TARGET_COL = "target"
DATE_COL = "issue_d"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def row_to_dict(row: pd.Series, feature_cols: list[str]) -> dict:
    """River expects {feature: value}; NaN → 0.0, categoricals → str."""
    out: dict = {}
    for c in feature_cols:
        v = row[c]
        if isinstance(v, float) and np.isnan(v):
            out[c] = 0.0
        elif pd.api.types.is_number(v):
            out[c] = float(v)
        else:
            out[c] = str(v)
    return out


def build_arf(seed: int = 42, n_models: int = 10) -> forest.ARFClassifier:
    """ARF tuned for tabular credit streams.

    Defaults from River with a slightly looser drift delta to avoid
    over-reaction to monthly noise, and a fixed seed for reproducibility.
    """
    return forest.ARFClassifier(
        n_models=n_models,
        seed=seed,
        drift_detector=drift.ADWIN(delta=ONLINE_DRIFT_ADWIN_DELTA),
        warning_detector=drift.ADWIN(delta=ONLINE_DRIFT_ADWIN_DELTA * 5),
        leaf_prediction="nba",  # naive-Bayes-adaptive — better calibration than majority
    )


# ---------------------------------------------------------------------------
# Stream evaluation
# ---------------------------------------------------------------------------
@dataclass
class StreamResult:
    predictions: pd.DataFrame   # one row per sample, with issue_d, y, p
    yearly: pd.DataFrame        # one row per year, with auroc/ks/brier/slope
    drift_events: list[dict]    # {timestamp, type, detector}


def _ks_from_arrays(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p")
    n_pos = int(y_true.sum()) or 1
    n_neg = int(len(y_true) - y_true.sum()) or 1
    return float(
        ((df["y"] == 1).cumsum() / n_pos - (df["y"] == 0).cumsum() / n_neg).abs().max()
    )


def stream_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_delay_days: int = ONLINE_LABEL_DELAY_DAYS,
    samples_per_month: int | None = 500,
    seed: int = 42,
    n_models: int = 10,
    log_every: int = 5000,
) -> StreamResult:
    """Process the DataFrame as a chronological stream.

    Parameters
    ----------
    df : DataFrame with ``issue_d``, ``target``, and feature columns.
    feature_cols : list[str]
    label_delay_days : int — labels released only after this many days.
    samples_per_month : int | None — stratified subsample for tractability
        on a laptop. None = use everything (~1.35M rows, ~hours).
    n_models : int — number of trees in the ARF ensemble.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if samples_per_month:
        df = (
            df.groupby(df[DATE_COL].dt.to_period("M"), group_keys=False)
            .apply(lambda g: g.sample(min(len(g), samples_per_month), random_state=seed))
            .reset_index(drop=True)
            .sort_values(DATE_COL)
            .reset_index(drop=True)
        )
        logger.info(f"Sub-sampled to {len(df):,} rows ({samples_per_month}/month max)")

    model = build_arf(seed=seed, n_models=n_models)
    score_drift = drift.KSWIN(alpha=ONLINE_DRIFT_KSWIN_ALPHA, window_size=200, stat_size=50, seed=seed)

    # Delayed-label queue: list of (release_date, x, y, idx)
    pending: deque = deque()
    drift_events: list[dict] = []

    pred_rows = []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        issue_d = pd.Timestamp(getattr(row, DATE_COL))
        x = row_to_dict(pd.Series(row._asdict()), feature_cols)
        y = int(getattr(row, TARGET_COL))

        # 1) Release any labels whose delay has expired
        while pending and pending[0][0] <= issue_d:
            _, x_old, y_old, _ = pending.popleft()
            model.learn_one(x_old, y_old)

        # 2) Predict (test-then-train style — prediction is OOS)
        proba = model.predict_proba_one(x)
        p = float(proba.get(1, 0.5)) if proba else 0.5

        # 3) Unsupervised drift on score distribution (KSWIN)
        score_drift.update(p)
        if score_drift.drift_detected:
            drift_events.append({"timestamp": issue_d, "type": "score_kswin", "detector": "KSWIN"})

        pred_rows.append({"issue_d": issue_d, "y": y, "p": p})

        # 4) Queue label for delayed release
        pending.append((issue_d + pd.Timedelta(days=label_delay_days), x, y, i))

        if (i + 1) % log_every == 0:
            recent_auc = roc_auc_score(
                [r["y"] for r in pred_rows[-log_every:]],
                [r["p"] for r in pred_rows[-log_every:]],
            ) if len({r["y"] for r in pred_rows[-log_every:]}) > 1 else float("nan")
            logger.info(f"  {i + 1:,}/{n:,} | recent {log_every} AUROC={recent_auc:.4f} | drifts so far={len(drift_events)}")

    # Drain any remaining pending labels so the model is fully trained
    while pending:
        _, x_old, y_old, _ = pending.popleft()
        model.learn_one(x_old, y_old)

    preds = pd.DataFrame(pred_rows)
    preds["year"] = preds["issue_d"].dt.year

    yearly_rows = []
    eps = 1e-6
    for year, g in preds.groupby("year"):
        if g["y"].nunique() < 2 or len(g) < 200:
            continue
        p = g["p"].to_numpy()
        y = g["y"].to_numpy()
        logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        slope = float(LinearRegression().fit(logit.reshape(-1, 1), y).coef_[0])
        yearly_rows.append({
            "year": int(year),
            "n_test": len(y),
            "base_rate_test": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)),
            "ks": _ks_from_arrays(y, p),
            "brier": float(brier_score_loss(y, p)),
            "calib_slope": slope,
        })
    yearly = pd.DataFrame(yearly_rows)

    logger.success(
        f"Stream done: {len(preds):,} predictions, {len(drift_events)} KSWIN drifts, "
        f"AUROC overall {roc_auc_score(preds['y'], preds['p']):.4f}"
    )
    return StreamResult(preds, yearly, drift_events)


if __name__ == "__main__":
    df = pd.read_parquet(LENDINGCLUB_FEATURES)
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, DATE_COL}]
    result = stream_evaluate(df, feature_cols, samples_per_month=500, n_models=10)
    print(result.yearly.to_string(index=False))
