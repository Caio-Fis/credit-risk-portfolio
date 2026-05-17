"""Live drift monitoring state.

A single ``LiveDriftMonitor`` is built at app startup and fed by every
prediction. It keeps three streaming structures:

  * **ADWIN on score** — supervised drift on the (unsupervised proxy)
    of the prediction itself. Fires when the running mean of predicted
    PD shifts meaningfully — useful as an early signal of input drift.
  * **KSWIN on score** — unsupervised; non-parametric Kolmogorov-Smirnov
    over a sliding window of the most recent predictions.
  * **Rolling PSI** — per numeric feature, comparing the most recent
    window against a static training reference distribution.

State persists to ``settings.drift_state_path`` periodically so a
restart does not lose history. Drift firings are also reflected on the
``credit_risk_drift_events_total`` Prometheus counter.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from river import drift

from src.api.metrics import DRIFT_EVENTS_TOTAL
from src.monitoring.psi import compute_psi

# Numeric features we'll monitor (categoricals would need OHE first;
# skip for now — the score-level KSWIN is what catches their effect).
NUMERIC_FEATURES_TO_MONITOR = [
    "revenue", "dti_n", "loan_amnt", "fico_n", "emp_length",
    "fed_funds_rate", "us_unemployment", "vix_close", "us_10y_treasury", "us_real_gdp_yoy",
]


@dataclass
class DriftEvent:
    timestamp: datetime
    detector: str
    type: str
    value: float | None = None


@dataclass
class LiveDriftMonitor:
    """Streaming drift monitor fed by every API prediction."""

    state_path: Path
    adwin_delta: float = 0.002
    kswin_alpha: float = 0.005
    kswin_window: int = 200
    kswin_stat: int = 50
    score_buffer_max: int = 5000
    feature_buffer_max: int = 5000
    psi_refresh_every: int = 200
    seed: int = 42

    # Built lazily so the dataclass remains pickle-friendly
    adwin: drift.ADWIN = field(default=None, init=False)  # type: ignore[assignment]
    kswin: drift.KSWIN = field(default=None, init=False)  # type: ignore[assignment]
    score_buffer: deque = field(default=None, init=False)  # type: ignore[assignment]
    feature_buffer: deque = field(default=None, init=False)  # type: ignore[assignment]
    events: deque = field(default=None, init=False)  # type: ignore[assignment]
    psi_reference: dict[str, np.ndarray] | None = None
    _last_psi_value: dict[str, float] = field(default_factory=dict, init=False)
    _samples_since_psi: int = field(default=0, init=False)
    _samples_seen: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    started_at: datetime = field(default_factory=datetime.utcnow, init=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialise(self, psi_reference: dict[str, np.ndarray] | None = None) -> None:
        self.adwin = drift.ADWIN(delta=self.adwin_delta)
        self.kswin = drift.KSWIN(
            alpha=self.kswin_alpha,
            window_size=self.kswin_window,
            stat_size=self.kswin_stat,
            seed=self.seed,
        )
        self.score_buffer = deque(maxlen=self.score_buffer_max)
        self.feature_buffer = deque(maxlen=self.feature_buffer_max)
        self.events = deque(maxlen=500)
        self.psi_reference = psi_reference
        logger.info(
            f"LiveDriftMonitor ready (ADWIN δ={self.adwin_delta}, "
            f"KSWIN α={self.kswin_alpha}, PSI features={len(self.psi_reference or {})})"
        )

    # ------------------------------------------------------------------
    # Update path — called per prediction
    # ------------------------------------------------------------------
    def observe(self, score: float, feature_row: dict[str, Any]) -> list[DriftEvent]:
        if self.adwin is None:
            return []
        fired: list[DriftEvent] = []
        now = datetime.utcnow()
        with self._lock:
            self._samples_seen += 1
            self.score_buffer.append(score)
            self.feature_buffer.append({k: feature_row.get(k) for k in NUMERIC_FEATURES_TO_MONITOR})

            # Score-level detectors
            self.adwin.update(score)
            if self.adwin.drift_detected:
                ev = DriftEvent(timestamp=now, detector="ADWIN", type="score_mean", value=score)
                self.events.append(ev)
                DRIFT_EVENTS_TOTAL.labels(detector="ADWIN").inc()
                fired.append(ev)
            self.kswin.update(score)
            if self.kswin.drift_detected:
                ev = DriftEvent(timestamp=now, detector="KSWIN", type="score_distribution", value=score)
                self.events.append(ev)
                DRIFT_EVENTS_TOTAL.labels(detector="KSWIN").inc()
                fired.append(ev)

            # Periodic PSI refresh — every N samples (avoid O(n) on each call)
            self._samples_since_psi += 1
            if (
                self.psi_reference is not None
                and self._samples_since_psi >= self.psi_refresh_every
                and len(self.feature_buffer) >= self.kswin_window
            ):
                self._refresh_psi(now, fired)
                self._samples_since_psi = 0
        return fired

    def _refresh_psi(self, now: datetime, fired: list[DriftEvent]) -> None:
        rows = list(self.feature_buffer)
        cur = pd.DataFrame(rows)
        for feature, ref_vals in self.psi_reference.items():
            if feature not in cur.columns:
                continue
            psi_val = compute_psi(ref_vals, cur[feature].to_numpy(dtype=float))
            self._last_psi_value[feature] = psi_val
            if psi_val > 0.20:
                ev = DriftEvent(timestamp=now, detector="PSI", type=f"feature:{feature}", value=psi_val)
                self.events.append(ev)
                DRIFT_EVENTS_TOTAL.labels(detector="PSI").inc()
                fired.append(ev)

    # ------------------------------------------------------------------
    # Snapshot / read path
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "samples_seen": self._samples_seen,
                "started_at": self.started_at.isoformat(),
                "score_buffer_size": len(self.score_buffer) if self.score_buffer else 0,
                "events_total": len(self.events) if self.events else 0,
                "events_by_detector": self._counts("detector"),
                "events_recent": [self._event_to_dict(e) for e in list(self.events)[-20:][::-1]]
                                  if self.events else [],
                "last_psi": dict(self._last_psi_value),
            }

    def _counts(self, attr: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for ev in (self.events or []):
            key = getattr(ev, attr)
            out[key] = out.get(key, 0) + 1
        return out

    @staticmethod
    def _event_to_dict(ev: DriftEvent) -> dict[str, Any]:
        return {
            "timestamp": ev.timestamp.isoformat(),
            "detector": ev.detector,
            "type": ev.type,
            "value": ev.value,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def persist(self) -> None:
        if not self.events:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "samples_seen": self._samples_seen,
            "started_at": self.started_at.isoformat(),
            "events": [self._event_to_dict(e) for e in self.events],
            "last_psi": self._last_psi_value,
        }
        with self.state_path.open("w") as fh:
            json.dump(payload, fh)
        logger.debug(f"Drift state persisted to {self.state_path}")


def build_psi_reference(features_path: Path, columns: list[str], sample_n: int = 50_000) -> dict[str, np.ndarray]:
    """Sample the feature parquet to obtain a reference distribution per feature."""
    if not features_path.exists():
        logger.warning(f"Feature store not available at {features_path}; PSI disabled.")
        return {}
    df = pd.read_parquet(features_path, columns=[c for c in columns if c])
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    return {c: df[c].to_numpy(dtype=float) for c in columns if c in df.columns}
