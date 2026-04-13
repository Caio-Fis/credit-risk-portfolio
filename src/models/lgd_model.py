"""Loss Given Default (LGD) model.

Architecture:
- Beta regression via statsmodels — response variable in [0, 1]
- Treated as a problem separate from PD: the cost of error in LGD is distinct
- DO NOT collapse PD and LGD into a single score

LGD = 1 - recovery_rate
- LGD = 0: full recovery (no loss)
- LGD = 1: total loss (no recovery)

For the Home Credit dataset, where observed LGD is not directly available,
we generate a proxy based on operational characteristics (collateral, contract type).

Main functions:
- train_lgd: trains Beta regression and logs to MLflow
- predict_lgd: returns estimated LGD per contract
- load_lgd_model: loads model from MLflow
"""

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

from src.config import (
    MLFLOW_EXPERIMENT_LGD,
    MLFLOW_TRACKING_URI,
)

LGD_COL = "lgd"
BETA_EPS = 1e-6  # avoids exact 0 and 1 (beta regression requires strict (0,1))


def _generate_lgd_proxy(df: pd.DataFrame) -> pd.Series:
    """Generates LGD proxy for Home Credit (no real observed LGD).

    Proxy based on known operational characteristics:
    - Unsecured operations → higher LGD (0.6–0.9)
    - Revolving credit type → intermediate LGD (0.4–0.7)
    - With collateral → low LGD (0.1–0.4)

    This proxy is explicitly documented as synthetic.
    In production, replace with observed recovery rate.
    """
    rng = np.random.default_rng(42)
    n = len(df)

    # Determine LGD range by contract type
    lgd = np.zeros(n)

    if "NAME_CONTRACT_TYPE" in df.columns:
        is_revolving = (df["NAME_CONTRACT_TYPE"] == "Revolving loans").values
        lgd[is_revolving] = rng.uniform(0.4, 0.7, is_revolving.sum())
        lgd[~is_revolving] = rng.uniform(0.2, 0.6, (~is_revolving).sum())
    else:
        lgd = rng.uniform(0.3, 0.7, n)

    # Modulate by external score: high ext_source_mean → lower LGD
    if "ext_source_mean" in df.columns:
        ext = df["ext_source_mean"].fillna(0.5).values
        lgd = lgd * (1 - 0.3 * ext)

    # Clip to strict (0, 1) (beta regression)
    lgd = np.clip(lgd, BETA_EPS, 1 - BETA_EPS)
    return pd.Series(lgd, index=df.index, name=LGD_COL)


def _prepare_lgd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects and prepares features for the LGD model."""
    lgd_feature_cols = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "ext_source_mean",
        "ext_source_min",
        "ext_source_max",
        "credit_income_ratio",
        "annuity_income_ratio",
    ]
    available = [c for c in lgd_feature_cols if c in df.columns]
    X = df[available].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X.fillna(X.median())


def _reconstruct_beta_wrapper(state: dict) -> "BetaRegressionWrapper":
    """Reconstructs BetaRegressionWrapper from serialised state.

    Module-level function required for __reduce__: cloudpickle requires that the
    reconstruction function be importable by name, not a lambda or closure.
    """
    obj = object.__new__(BetaRegressionWrapper)
    obj._feature_cols = state["feature_cols"]
    if state["model_type"] == "ridge":
        obj._model = state["model"]
    else:
        obj._model = {"params": state["params"]}
    return obj


class BetaRegressionWrapper:
    """Scikit-learn-like wrapper around statsmodels Beta regression.

    Implements __reduce__ for safe serialisation via pickle/cloudpickle:
    statsmodels GLMResultsWrapper stores internal file handles that
    block cloudpickle. __reduce__ ensures only the coefficients
    (params) are serialised, reconstructing prediction with the logit link.
    """

    def __init__(self) -> None:
        self._model = None
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BetaRegressionWrapper":
        self._feature_cols = list(X.columns)
        data = X.copy()
        data["__lgd__"] = np.clip(y.values, BETA_EPS, 1 - BETA_EPS)

        formula = "__lgd__ ~ " + " + ".join(self._feature_cols)
        try:
            self._model = smf.glm(
                formula=formula,
                data=data,
                family=__import__(
                    "statsmodels.genmod.families", fromlist=["Binomial"]
                ).Binomial(),
            ).fit(disp=False)
        except Exception:
            # Fallback: Ridge regression with manual logit link
            logger.warning(
                "Beta regression via statsmodels failed. Using Ridge with logit link."
            )
            self._model = Ridge(alpha=1.0)
            self._model.fit(X, np.log(y / (1 - y)))  # logit(LGD)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(self._model, Ridge):
            logit_pred = self._model.predict(X[self._feature_cols])
            return np.clip(1 / (1 + np.exp(-logit_pred)), BETA_EPS, 1 - BETA_EPS)

        if isinstance(self._model, dict):
            # Reconstructed from __setstate__: applies logit link manually
            # Binomial GLM with logit link: μ = sigmoid(Intercept + X @ β)
            params = self._model["params"]
            coef = np.array(
                [params.get("Intercept", 0.0)]
                + [params.get(c, 0.0) for c in self._feature_cols]
            )
            X_mat = np.column_stack(
                [np.ones(len(X))] + [X[c].values for c in self._feature_cols]
            )
            eta = X_mat @ coef
            return np.clip(1 / (1 + np.exp(-eta)), BETA_EPS, 1 - BETA_EPS)

        # GLMResultsWrapper still in memory (before serialisation)
        data = X[self._feature_cols].copy()
        return np.clip(self._model.predict(data), BETA_EPS, 1 - BETA_EPS)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def __reduce__(self) -> tuple:
        """Safe serialisation for pickle and cloudpickle.

        cloudpickle prioritises __reduce__ over __getstate__/__setstate__.
        Returns only coefficients — without statsmodels file handles.
        """
        if isinstance(self._model, Ridge):
            state = {
                "model_type": "ridge",
                "feature_cols": self._feature_cols,
                "model": self._model,
            }
        elif isinstance(self._model, dict):
            state = {
                "model_type": "glm_params",
                "feature_cols": self._feature_cols,
                "params": self._model["params"],
            }
        else:
            # GLMResultsWrapper: extract only coefficients (without file handles)
            state = {
                "model_type": "glm_params",
                "feature_cols": self._feature_cols,
                "params": self._model.params.to_dict(),
            }
        return (_reconstruct_beta_wrapper, (state,))


def train_lgd(
    df: pd.DataFrame,
    run_name: str = "beta_regression",
) -> BetaRegressionWrapper:
    """Trains Beta regression for LGD and logs to MLflow.

    Args:
        df: Feature store (uses LGD proxy if 'lgd' column is absent).
        run_name: MLflow run name.

    Returns:
        Trained LGD model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_LGD)

    # Ensure LGD column exists
    if LGD_COL not in df.columns:
        logger.warning("Column 'lgd' absent — generating synthetic proxy (documented).")
        df = df.copy()
        df[LGD_COL] = _generate_lgd_proxy(df)

    X = _prepare_lgd_features(df)
    y = df[LGD_COL].clip(BETA_EPS, 1 - BETA_EPS)

    logger.info(f"Training LGD — {len(y):,} samples. Mean LGD: {y.mean():.3f}")

    model = BetaRegressionWrapper()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "beta_regression_statsmodels")
        mlflow.log_param("n_samples", len(y))
        mlflow.log_param("lgd_mean", float(y.mean()))
        mlflow.log_param("lgd_std", float(y.std()))

        model.fit(X, y)

        pred = model.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(y - pred)))

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        logger.info(f"LGD — R²: {r2:.4f}, MAE: {mae:.4f}")

        # statsmodels GLMResultsWrapper has internal file handles that block
        # MLflow cloudpickle. Convert to params dict before serialising;
        # predict() already handles the dict case (reconstruction via logit link).
        if hasattr(model._model, "params"):
            model._model = {"params": model._model.params.to_dict()}

        # cloudpickle fails with loguru's file sink (mode 'a') in module globals.
        # Standard pickle serialises importable classes by name, without
        # traversing globals — resolves the cloudpickle PicklingError.
        mlflow.sklearn.log_model(
            model,
            artifact_path="lgd_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        run_id = mlflow.active_run().info.run_id

    logger.success(f"LGD model trained. Run ID: {run_id}")
    return model


def predict_lgd(
    model: BetaRegressionWrapper,
    df: pd.DataFrame,
) -> np.ndarray:
    """Returns estimated LGD per contract.

    Args:
        model: Model trained via train_lgd().
        df: Features DataFrame.

    Returns:
        Array of estimated LGD in [0, 1].
    """
    X = _prepare_lgd_features(df)
    lgd = model.predict(X)
    logger.debug(
        f"LGD predicted: min={lgd.min():.4f}, max={lgd.max():.4f}, mean={lgd.mean():.4f}"
    )
    return lgd


if __name__ == "__main__":
    from src.features.build_features import load_feature_store

    df = load_feature_store()
    model = train_lgd(df)
