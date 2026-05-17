"""Streaming drift detection: ADWIN (supervised, on prediction error) and
KSWIN (unsupervised, on score distribution).

ADWIN reacts to changes in *model error* — needs labels, so it lags by
``ONLINE_LABEL_DELAY_DAYS`` in production. KSWIN reacts to changes in the
score distribution itself — does not need labels, fires earlier when input
distribution shifts, but cannot tell whether the shift hurts performance.

The combination is the standard "watch both" pattern: KSWIN flags
suspicious shifts immediately, ADWIN later confirms whether they actually
degraded the model.
"""

import pandas as pd
from loguru import logger
from river import drift

from src.config import ONLINE_DRIFT_ADWIN_DELTA, ONLINE_DRIFT_KSWIN_ALPHA


def detect_drifts_on_stream(
    predictions: pd.DataFrame,
    score_col: str = "p",
    label_col: str = "y",
    date_col: str = "issue_d",
    adwin_delta: float = ONLINE_DRIFT_ADWIN_DELTA,
    kswin_alpha: float = ONLINE_DRIFT_KSWIN_ALPHA,
    kswin_window: int = 200,
    kswin_stat: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Replay a stream of (date, score, label) and record drift events.

    Returns a DataFrame with one row per drift event:
        timestamp, type, detector_value, ix
    """
    adwin = drift.ADWIN(delta=adwin_delta)
    kswin = drift.KSWIN(alpha=kswin_alpha, window_size=kswin_window, stat_size=kswin_stat, seed=seed)

    events: list[dict] = []
    for ix, row in enumerate(predictions.itertuples(index=False)):
        p = float(getattr(row, score_col))
        y = int(getattr(row, label_col))
        ts = pd.Timestamp(getattr(row, date_col))

        kswin.update(p)
        if kswin.drift_detected:
            events.append({"timestamp": ts, "type": "score", "detector": "KSWIN", "value": p, "ix": ix})

        # 0/1 error feed for ADWIN — higher = worse model
        err = abs(y - p)
        adwin.update(err)
        if adwin.drift_detected:
            events.append({"timestamp": ts, "type": "error", "detector": "ADWIN", "value": err, "ix": ix})

    out = pd.DataFrame(events)
    if not out.empty:
        logger.info(
            f"Drift events: total={len(out)} | "
            f"KSWIN={int((out['detector'] == 'KSWIN').sum())} | "
            f"ADWIN={int((out['detector'] == 'ADWIN').sum())}"
        )
    else:
        logger.info("No drift events detected.")
    return out
