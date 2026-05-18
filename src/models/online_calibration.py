"""Sliding-window probability calibration for streaming PD models.

Standard isotonic / Platt calibration assumes the (raw_score → true_rate)
mapping is stationary. Under macro-driven concept drift it is not: the
same raw score in 2008 means a much higher realised default rate than
the same raw score in 2014.

This module refits the calibrator over a rolling window of the most
recent ``window_months`` of labelled history. It is used downstream of
both the static champion (refresh monthly) and the ARF challenger
(refresh per-month or per-drift-event).

Two variants:
  * SlidingWindowIsotonic — non-parametric, no functional form. Default.
  * SlidingWindowPlatt    — logistic; cheaper, more stable on tiny windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import ONLINE_CALIB_WINDOW_MONTHS


@dataclass
class _Window:
    """Rolling buffer of (date, raw_score, label) tuples."""
    window_months: int
    raw: deque
    y: deque
    ts: deque

    @classmethod
    def empty(cls, window_months: int) -> "_Window":
        return cls(window_months=window_months, raw=deque(), y=deque(), ts=deque())

    def add(self, raw_score: float, label: int, ts: pd.Timestamp) -> None:
        self.raw.append(float(raw_score))
        self.y.append(int(label))
        self.ts.append(ts)
        cutoff = ts - pd.DateOffset(months=self.window_months)
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
            self.raw.popleft()
            self.y.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.raw), np.asarray(self.y)


class SlidingWindowIsotonic:
    """Isotonic regression refit on the trailing window.

    Refits are expensive (O(n log n)) so the calibrator is rebuilt only when
    the month changes — about once per ~30 days of stream, regardless of
    sample volume. The calibrator handles per-sample transforms with a
    cheap interpolation.
    """

    def __init__(self, window_months: int = ONLINE_CALIB_WINDOW_MONTHS, min_samples: int = 500):
        self.window = _Window.empty(window_months)
        self.min_samples = min_samples
        self._cal: IsotonicRegression | None = None
        self._last_refit_month: pd.Period | None = None

    def update(self, raw_score: float, label: int, ts: pd.Timestamp) -> None:
        self.window.add(raw_score, label, ts)
        current_month = pd.Period(ts, freq="M")
        if self._last_refit_month == current_month:
            return
        x, y = self.window.arrays()
        if len(x) >= self.min_samples and (y == 0).any() and (y == 1).any():
            self._cal = IsotonicRegression(out_of_bounds="clip").fit(x, y)
            self._last_refit_month = current_month

    def transform(self, raw_score: float) -> float:
        if self._cal is None:
            return float(raw_score)
        return float(self._cal.transform([raw_score])[0])


class SlidingWindowPlatt:
    """Single-feature logistic regression on the trailing window (monthly refit)."""

    def __init__(self, window_months: int = ONLINE_CALIB_WINDOW_MONTHS, min_samples: int = 500):
        self.window = _Window.empty(window_months)
        self.min_samples = min_samples
        self._cal: LogisticRegression | None = None
        self._last_refit_month: pd.Period | None = None

    def update(self, raw_score: float, label: int, ts: pd.Timestamp) -> None:
        self.window.add(raw_score, label, ts)
        current_month = pd.Period(ts, freq="M")
        if self._last_refit_month == current_month:
            return
        x, y = self.window.arrays()
        if len(x) >= self.min_samples and (y == 0).any() and (y == 1).any():
            self._cal = LogisticRegression(C=10.0).fit(x.reshape(-1, 1), y)
            self._last_refit_month = current_month

    def transform(self, raw_score: float) -> float:
        if self._cal is None:
            return float(raw_score)
        return float(self._cal.predict_proba(np.array([[raw_score]]))[0, 1])


def apply_sliding_calibration(
    preds: pd.DataFrame,
    raw_col: str = "p",
    label_col: str = "y",
    date_col: str = "issue_d",
    method: str = "isotonic",
    window_months: int = ONLINE_CALIB_WINDOW_MONTHS,
    label_delay_days: int = 90,
) -> pd.DataFrame:
    """Apply a sliding-window calibrator to an already-scored stream.

    Vectorised by month: at the start of each calendar month the calibrator
    is refit on the last ``window_months`` of observations whose label
    would already have arrived (i.e. issued ≤ month_start − label_delay).
    All predictions falling in that month are then transformed with that
    single calibrator. This is the production-realistic equivalent of the
    sample-by-sample version, but ~3 orders of magnitude faster.

    Note: the returned frame is sorted by ``date_col`` and re-indexed.
    Don't assign ``.values`` back to the original frame — work with the
    returned DataFrame directly or join on a stable key.
    """
    if method == "isotonic":
        Calibrator = lambda x, y: IsotonicRegression(out_of_bounds="clip").fit(x, y)  # noqa: E731
        Predict = lambda c, x: c.transform(x)  # noqa: E731
    elif method == "platt":
        Calibrator = lambda x, y: LogisticRegression(C=10.0).fit(x.reshape(-1, 1), y)  # noqa: E731
        Predict = lambda c, x: c.predict_proba(x.reshape(-1, 1))[:, 1]  # noqa: E731
    else:
        raise ValueError(f"Unknown method {method!r}")

    preds = preds.copy().sort_values(date_col).reset_index(drop=True)
    preds[date_col] = pd.to_datetime(preds[date_col])
    preds["_month"] = preds[date_col].values.astype("datetime64[M]")

    out_col = f"p_cal_{method}"
    preds[out_col] = preds[raw_col].astype("float64")

    months = sorted(preds["_month"].unique())
    cal = None
    delay = pd.Timedelta(days=label_delay_days)
    win_offset = pd.DateOffset(months=window_months)

    for m in months:
        m_start = pd.Timestamp(m)
        # Training window: observations whose label would have arrived
        usable_until = m_start - delay
        usable_from = usable_until - win_offset
        train_mask = (preds[date_col] >= usable_from) & (preds[date_col] <= usable_until)
        train = preds.loc[train_mask]
        if len(train) >= 500 and train[label_col].nunique() == 2:
            cal = Calibrator(train[raw_col].to_numpy(), train[label_col].to_numpy())

        cur_mask = preds["_month"] == m
        if cal is None:
            continue  # leave raw scores as the placeholder
        preds.loc[cur_mask, out_col] = Predict(cal, preds.loc[cur_mask, raw_col].to_numpy())

    preds = preds.drop(columns=["_month"])
    return preds


if __name__ == "__main__":
    import time

    from sklearn.metrics import brier_score_loss

    from src.config import PROCESSED_DIR

    pred_path = PROCESSED_DIR / "arf_predictions_lc.parquet"
    if not pred_path.exists():
        logger.warning(f"{pred_path} missing — run src.models.online_pd_model first.")
        raise SystemExit(0)

    preds = pd.read_parquet(pred_path)
    t0 = time.time()
    cal = apply_sliding_calibration(preds, method="isotonic", window_months=6, label_delay_days=90)
    logger.info(f"Calibration applied in {time.time() - t0:.1f}s")

    raw_brier = brier_score_loss(cal["y"], cal["p"])
    cal_brier = brier_score_loss(cal["y"], cal["p_cal_isotonic"])
    logger.info(f"Brier — raw: {raw_brier:.4f} | sliding isotonic: {cal_brier:.4f}")
    cal.to_parquet(PROCESSED_DIR / "arf_predictions_lc_calibrated.parquet", index=False)
