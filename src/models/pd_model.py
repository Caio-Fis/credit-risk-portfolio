"""Probability of Default (PD) model.

Architecture:
- LightGBM as ranker (good AUROC/KS but poorly calibrated probabilities)
- CalibratedClassifierCV with method='isotonic' to calibrate probabilities
  to real scale — required for computing EL in currency units

Platt scaling (sigmoid) was discarded after Hosmer-Lemeshow and Basel binomial
tests revealed systematic miscalibration in the 5–20% PD range:
the model underpredicts risk in that range by ~20–35%. Isotonic regression
has no functional form restriction and corrects the curvature. Overfitting
of isotonic only risks with <1000 samples — we have 246K training samples.

Main functions:
- train_pd: trains LightGBM + isotonic and logs to MLflow
- predict_pd: returns calibrated probabilities
- load_pd_model: loads model from MLflow
"""

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    MLFLOW_EXPERIMENT_PD,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
)

# ---------------------------------------------------------------------------
# Default hyperparameters (never hardcode outside this file)
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict[str, Any] = {
    # Params optimised via Optuna (50 trials, TPE, 5-fold CV on train split)
    # Baseline before tuning: OOS AUROC 0.7747 | After tuning: 0.7760 (params only)
    # With additional feature engineering: target ≥ 0.78
    "n_estimators": 1000,
    "learning_rate": 0.020151,
    "num_leaves": 38,
    "max_depth": -1,
    "min_child_samples": 104,
    "min_split_gain": 0.214926,
    "subsample": 0.784504,
    "colsample_bytree": 0.586299,
    "reg_alpha": 1.434127,
    "reg_lambda": 1.357588,
    "random_state": 42,
    "n_jobs": 2,   # -1 replicates data across N_CPU simultaneous workers → OOM
    "verbose": -1,
}

CALIBRATION_CV = 3  # calibration folds; 3 is sufficient and saves RAM
TARGET_COL = "TARGET"
TEST_SIZE = 0.2      # 20% hold-out for out-of-sample evaluation
OOS_PATH = PROCESSED_DIR / "oos_predictions.parquet"


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Removes non-feature columns and encodes categories."""
    drop_cols = [TARGET_COL, "SK_ID_CURR", "reference_date"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # Encode object/category columns with LabelEncoder
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # float32 reduces memory usage ~50% vs float64 (307K × 100 cols: 250 MB → 125 MB)
    X = X.astype({c: "float32" for c in X.select_dtypes("float64").columns})

    return X, feature_cols


def train_pd(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    run_name: str = "lgbm_isotonic",
) -> CalibratedClassifierCV:
    """Trains LightGBM + isotonic calibration and logs artefacts to MLflow.

    Stratified 80/20 split: trains on train split, evaluates out-of-sample
    on test split. Saves OOS predictions to OOS_PATH for evaluate to use.

    Args:
        df: Feature store with TARGET column.
        target_col: Target column name.
        run_name: MLflow run name.

    Returns:
        Calibrated model ready for predict_proba (trained on train split).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PD)

    X, feature_cols = _prepare_features(df)
    y = df[target_col].values

    # Stratified split — preserves default rate in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    logger.info(
        f"Training PD — train: {len(y_train):,} | test (OOS): {len(y_test):,} | "
        f"features: {X.shape[1]} | default rate: {y.mean():.2%}"
    )

    lgbm = LGBMClassifier(**LGBM_PARAMS)

    # Isotonic calibration: trains LGBM on (n-1) folds, calibrates on remaining fold
    calibrated_model = CalibratedClassifierCV(
        estimator=lgbm,
        method="isotonic",  # corrects non-linear curvature; Platt caused miscalibration at 5–20%
        cv=CALIBRATION_CV,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("calibration_method", "isotonic")
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("default_rate_train", float(y_train.mean()))

        # CV on train split (does not leak test)
        cv_scores = cross_val_score(
            calibrated_model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
            n_jobs=1,  # parallel folds double memory; sequential is safe
        )
        logger.info(f"CV AUROC (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        mlflow.log_metric("cv_auroc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_auroc_std", float(cv_scores.std()))

        # Train final model on train split
        calibrated_model.fit(X_train, y_train)

        # Evaluate on hold-out (OOS) and save predictions for metrics.py
        y_pred_train = calibrated_model.predict_proba(X_train)[:, 1]
        y_pred_test = calibrated_model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import roc_auc_score
        auroc_train = roc_auc_score(y_train, y_pred_train)
        auroc_test = roc_auc_score(y_test, y_pred_test)
        logger.info(f"AUROC — train: {auroc_train:.4f} | OOS: {auroc_test:.4f}")
        mlflow.log_metric("auroc_train", float(auroc_train))
        mlflow.log_metric("auroc_oos", float(auroc_test))

        oos_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test})
        oos_df.to_parquet(OOS_PATH, index=False)
        mlflow.log_artifact(str(OOS_PATH), artifact_path="oos")
        logger.info(f"OOS predictions saved at {OOS_PATH}")

        mlflow.sklearn.log_model(
            calibrated_model,
            artifact_path="pd_model",
            registered_model_name="pd_lgbm_isotonic",
        )
        run_id = mlflow.active_run().info.run_id

    logger.success(f"PD model trained. Run ID: {run_id}")
    return calibrated_model


def predict_pd(
    model: CalibratedClassifierCV,
    df: pd.DataFrame,
) -> np.ndarray:
    """Returns calibrated default probabilities.

    Args:
        model: Model trained via train_pd().
        df: Features DataFrame (without TARGET).

    Returns:
        Array of P(default) probabilities in [0, 1].
    """
    X, _ = _prepare_features(df)
    proba = model.predict_proba(X)[:, 1]
    logger.debug(
        f"PD predicted: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}"
    )
    return proba


def load_pd_model(
    model_name: str = "pd_lgbm_isotonic",
    stage: str = "latest",
) -> CalibratedClassifierCV:
    """Loads PD model from MLflow Model Registry.

    Args:
        model_name: Registered model name.
        stage: Stage ('latest', 'Production', 'Staging').

    Returns:
        Loaded model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading PD model: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    from src.features.build_features import load_feature_store

    df = load_feature_store()
    model = train_pd(df)
