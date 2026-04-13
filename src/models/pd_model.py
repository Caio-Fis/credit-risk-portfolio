"""Modelo de Probabilidade de Default (PD).

Arquitetura:
- LightGBM como ranqueador (bom AUROC/KS mas probabilidades mal calibradas)
- CalibratedClassifierCV com method='isotonic' para calibrar as probabilidades
  em escala real — necessário para calcular EL em R$

Platt scaling (sigmoid) foi descartado após testes de Hosmer-Lemeshow e
binomial Basel mostrarem miscalibração sistemática na faixa 5–20% de PD:
o modelo subprediz risco nessa faixa em ~20–35%. Isotonic regression
não tem restrição de forma funcional e corrige a curvatura. Overfitting
do isotônico só é risco com <1000 amostras — temos 246K de treino.

Funções principais:
- train_pd: treina LightGBM + isotonic e loga no MLflow
- predict_pd: retorna probabilidades calibradas
- load_pd_model: carrega modelo do MLflow
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
# Hiperparâmetros default (não hardcodar fora daqui)
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict[str, Any] = {
    # Params otimizados via Optuna (50 trials, TPE, CV 5-fold no train split)
    # Baseline antes do tuning: AUROC OOS 0.7747 | Após tuning: 0.7760 (só params)
    # Com feature engineering adicional: alvo ≥ 0.78
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
    "n_jobs": 2,   # -1 replica os dados em N_CPU workers simultâneos → OOM
    "verbose": -1,
}

CALIBRATION_CV = 3  # folds para calibração; 3 é suficiente e poupa RAM
TARGET_COL = "TARGET"
TEST_SIZE = 0.2      # 20% hold-out para avaliação out-of-sample
OOS_PATH = PROCESSED_DIR / "oos_predictions.parquet"


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove colunas não-feature e encoda categorias."""
    drop_cols = [TARGET_COL, "SK_ID_CURR", "reference_date"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # Encoda colunas object/category com LabelEncoder
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # float32 reduz uso de memória ~50% vs float64 (307K × 100 cols: 250 MB → 125 MB)
    X = X.astype({c: "float32" for c in X.select_dtypes("float64").columns})

    return X, feature_cols


def train_pd(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    run_name: str = "lgbm_platt",
) -> CalibratedClassifierCV:
    """Treina LightGBM + Platt scaling e loga artefatos no MLflow.

    Split estratificado 80/20: treina no train split, avalia out-of-sample
    no test split. Salva predições OOS em OOS_PATH para evaluate usar.

    Args:
        df: Feature store com coluna TARGET.
        target_col: Nome da coluna alvo.
        run_name: Nome do run no MLflow.

    Returns:
        Modelo calibrado pronto para predict_proba (treinado no train split).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PD)

    X, feature_cols = _prepare_features(df)
    y = df[target_col].values

    # Split estratificado — mantém proporção de default em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    logger.info(
        f"Treinando PD — train: {len(y_train):,} | test (OOS): {len(y_test):,} | "
        f"features: {X.shape[1]} | default rate: {y.mean():.2%}"
    )

    lgbm = LGBMClassifier(**LGBM_PARAMS)

    # Platt scaling: treina LGBM em (n-1) folds, calibra no fold restante
    calibrated_model = CalibratedClassifierCV(
        estimator=lgbm,
        method="isotonic",  # corrige curvatura não-linear; Platt causava miscalibração 5–20%
        cv=CALIBRATION_CV,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("calibration_method", "platt_sigmoid")
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("default_rate_train", float(y_train.mean()))

        # CV no train split (não vaza o test)
        cv_scores = cross_val_score(
            calibrated_model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
            n_jobs=1,  # folds em paralelo duplicam a memória; sequencial é seguro
        )
        logger.info(f"CV AUROC (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        mlflow.log_metric("cv_auroc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_auroc_std", float(cv_scores.std()))

        # Treina modelo final no train split
        calibrated_model.fit(X_train, y_train)

        # Avalia no hold-out (OOS) e salva predições para metrics.py
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
        logger.info(f"Predições OOS salvas em {OOS_PATH}")

        mlflow.sklearn.log_model(
            calibrated_model,
            artifact_path="pd_model",
            registered_model_name="pd_lgbm_platt",
        )
        run_id = mlflow.active_run().info.run_id

    logger.success(f"Modelo PD treinado. Run ID: {run_id}")
    return calibrated_model


def predict_pd(
    model: CalibratedClassifierCV,
    df: pd.DataFrame,
) -> np.ndarray:
    """Retorna probabilidades calibradas de default.

    Args:
        model: Modelo treinado via train_pd().
        df: DataFrame de features (sem TARGET).

    Returns:
        Array de probabilidades P(default) em [0, 1].
    """
    X, _ = _prepare_features(df)
    proba = model.predict_proba(X)[:, 1]
    logger.debug(
        f"PD predito: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}"
    )
    return proba


def load_pd_model(
    model_name: str = "pd_lgbm_platt",
    stage: str = "latest",
) -> CalibratedClassifierCV:
    """Carrega modelo PD do MLflow Model Registry.

    Args:
        model_name: Nome do modelo registrado.
        stage: Estágio ('latest', 'Production', 'Staging').

    Returns:
        Modelo carregado.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Carregando modelo PD: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    from src.features.build_features import load_feature_store

    df = load_feature_store()
    model = train_pd(df)
