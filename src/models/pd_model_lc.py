"""PD model — v2 champion on LendingClub 2007-2018 (FRED-enriched).

Differences from v1 (Home Credit, src.models.pd_model):
  * Strict temporal split (no random shuffle): train ≤ 2014, val = 2015,
    test = 2016-2017. 2018 is excluded — the Zenodo dump filtered loans to
    final status only, so 2018 loans are under-represented for default.
  * Calibration is fit on the validation slice (not via random-CV) to
    preserve temporal ordering — sliding-window calibration is added later
    in src.models.online_calibration.
  * LightGBM uses native categorical handling (no LabelEncoder).
  * Logs to a dedicated MLflow experiment so v1 metrics are not overwritten.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import (
    ARTIFACTS_DIR,
    LENDINGCLUB_FEATURES,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TARGET_COL = "target"
DATE_COL = "issue_d"
CATEGORICAL_COLS = ["purpose", "home_ownership_n", "addr_state", "zip_code"]

TRAIN_END = "2014-12-31"          # inclusive
VAL_START, VAL_END = "2015-01-01", "2015-12-31"
TEST_START, TEST_END = "2016-01-01", "2017-12-31"
# 2018 excluded — see module docstring.

LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 200,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

OOS_PATH = PROCESSED_DIR / "oos_predictions_lc.parquet"
MODEL_PATH = ARTIFACTS_DIR / "pd_model_lc.joblib"
CALIBRATOR_PATH = ARTIFACTS_DIR / "pd_calibrator_lc.joblib"


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
@dataclass
class TemporalSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: list[str]


def _to_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def temporal_split(df: pd.DataFrame) -> TemporalSplit:
    """Splits the feature parquet by `issue_d` into train/val/test."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    train = df[df[DATE_COL] <= TRAIN_END]
    val = df[(df[DATE_COL] >= VAL_START) & (df[DATE_COL] <= VAL_END)]
    test = df[(df[DATE_COL] >= TEST_START) & (df[DATE_COL] <= TEST_END)]

    feature_cols = [c for c in df.columns if c not in {TARGET_COL, DATE_COL}]

    def _xy(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x = _to_categoricals(part[feature_cols], CATEGORICAL_COLS)
        return x, part[TARGET_COL].astype("int8")

    X_tr, y_tr = _xy(train)
    X_va, y_va = _xy(val)
    X_te, y_te = _xy(test)

    logger.info(
        f"Temporal split — train ≤ {TRAIN_END}: {len(y_tr):,} ({y_tr.mean():.2%}) | "
        f"val {VAL_START[:7]}: {len(y_va):,} ({y_va.mean():.2%}) | "
        f"test {TEST_START[:7]}..{TEST_END[:7]}: {len(y_te):,} ({y_te.mean():.2%})"
    )
    return TemporalSplit(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_cols)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_pd_lc(
    df: pd.DataFrame | None = None,
    run_name: str = "lgbm_isotonic_lc",
) -> tuple[LGBMClassifier, IsotonicRegression]:
    """Trains LightGBM + isotonic calibration with strict temporal split."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("pd_model_lc")

    if df is None:
        df = pd.read_parquet(LENDINGCLUB_FEATURES)

    split = temporal_split(df)

    lgbm = LGBMClassifier(**LGBM_PARAMS)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("train_end", TRAIN_END)
        mlflow.log_param("val_period", f"{VAL_START} → {VAL_END}")
        mlflow.log_param("test_period", f"{TEST_START} → {TEST_END}")
        mlflow.log_param("n_train", len(split.y_train))
        mlflow.log_param("n_val", len(split.y_val))
        mlflow.log_param("n_test", len(split.y_test))
        mlflow.log_param("default_rate_train", float(split.y_train.mean()))

        lgbm.fit(
            split.X_train,
            split.y_train,
            eval_set=[(split.X_val, split.y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(50), log_evaluation(period=0)],
            categorical_feature=CATEGORICAL_COLS,
        )

        # Calibrate on validation — preserves temporal ordering
        p_val_raw = lgbm.predict_proba(split.X_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, split.y_val)

        # Evaluate uncalibrated and calibrated on val + test
        p_val_cal = calibrator.transform(p_val_raw)
        p_test_raw = lgbm.predict_proba(split.X_test)[:, 1]
        p_test_cal = calibrator.transform(p_test_raw)

        metrics = {
            "auroc_val_raw": roc_auc_score(split.y_val, p_val_raw),
            "auroc_val_cal": roc_auc_score(split.y_val, p_val_cal),
            "auroc_test_raw": roc_auc_score(split.y_test, p_test_raw),
            "auroc_test_cal": roc_auc_score(split.y_test, p_test_cal),
            "brier_test_raw": brier_score_loss(split.y_test, p_test_raw),
            "brier_test_cal": brier_score_loss(split.y_test, p_test_cal),
            "best_iteration": int(lgbm.best_iteration_),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        logger.info("Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        # Persist OOS preds for evaluate.metrics consumption
        oos_df = pd.DataFrame({
            "issue_d": pd.to_datetime(df[df[DATE_COL] >= TEST_START][DATE_COL].head(0)),
        })  # placeholder structure
        test_meta = df[(df[DATE_COL] >= TEST_START) & (df[DATE_COL] <= TEST_END)].reset_index(drop=True)
        oos_df = pd.DataFrame({
            "issue_d": test_meta[DATE_COL].values,
            "y_true": split.y_test.values,
            "y_pred_raw": p_test_raw,
            "y_pred": p_test_cal,
        })
        OOS_PATH.parent.mkdir(parents=True, exist_ok=True)
        oos_df.to_parquet(OOS_PATH, index=False)
        mlflow.log_artifact(str(OOS_PATH), artifact_path="oos")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lgbm, MODEL_PATH)
        joblib.dump(calibrator, CALIBRATOR_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(CALIBRATOR_PATH))

        logger.success(f"v2 champion trained — run_id={mlflow.active_run().info.run_id}")

    return lgbm, calibrator


def predict_pd_lc(
    model: LGBMClassifier,
    calibrator: IsotonicRegression,
    df: pd.DataFrame,
) -> np.ndarray:
    """Calibrated PD for new data. Feature engineering must be applied upstream."""
    X = _to_categoricals(df.drop(columns=[c for c in (TARGET_COL, DATE_COL) if c in df.columns]),
                          CATEGORICAL_COLS)
    p_raw = model.predict_proba(X)[:, 1]
    return calibrator.transform(p_raw)


def load_pd_lc(model_path: Path = MODEL_PATH, calib_path: Path = CALIBRATOR_PATH):
    return joblib.load(model_path), joblib.load(calib_path)


if __name__ == "__main__":
    train_pd_lc()
