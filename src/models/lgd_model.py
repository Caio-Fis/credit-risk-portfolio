"""Modelo de Loss Given Default (LGD).

Arquitetura:
- Regressão Beta via statsmodels — variável resposta em [0, 1]
- Tratado como problema separado do PD: o custo de erro em LGD é distinto
- NÃO colapsar PD e LGD num score único

LGD = 1 - taxa_de_recuperação
- LGD = 0: recuperação total (sem perda)
- LGD = 1: perda total (sem recuperação)

Para o dataset Home Credit, onde LGD observado não está disponível diretamente,
geramos um proxy com base em características da operação (garantia, tipo de contrato).

Funções principais:
- train_lgd: treina regressão Beta e loga no MLflow
- predict_lgd: retorna LGD estimado por contrato
- load_lgd_model: carrega modelo do MLflow
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
BETA_EPS = 1e-6  # evita 0 e 1 exatos (beta regression requer (0,1) estrito)


def _generate_lgd_proxy(df: pd.DataFrame) -> pd.Series:
    """Gera LGD proxy para Home Credit (sem LGD real observado).

    Proxy baseado em características conhecidas da operação:
    - Operações sem garantia → LGD mais alto (0.6–0.9)
    - Tipo de crédito revolvente → LGD intermediário (0.4–0.7)
    - Com garantia real → LGD baixo (0.1–0.4)

    Este proxy é explicitamente documentado como sintético.
    Em produção, substituir por taxa de recuperação observada.
    """
    rng = np.random.default_rng(42)
    n = len(df)

    # Determina faixa de LGD por tipo de contrato
    lgd = np.zeros(n)

    if "NAME_CONTRACT_TYPE" in df.columns:
        is_revolving = (df["NAME_CONTRACT_TYPE"] == "Revolving loans").values
        lgd[is_revolving] = rng.uniform(0.4, 0.7, is_revolving.sum())
        lgd[~is_revolving] = rng.uniform(0.2, 0.6, (~is_revolving).sum())
    else:
        lgd = rng.uniform(0.3, 0.7, n)

    # Modula pelo score externo: ext_source_mean alto → LGD menor
    if "ext_source_mean" in df.columns:
        ext = df["ext_source_mean"].fillna(0.5).values
        lgd = lgd * (1 - 0.3 * ext)

    # Clipa para (0, 1) estrito (beta regression)
    lgd = np.clip(lgd, BETA_EPS, 1 - BETA_EPS)
    return pd.Series(lgd, index=df.index, name=LGD_COL)


def _prepare_lgd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e prepara features para o modelo LGD."""
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
    """Reconstrói BetaRegressionWrapper a partir do estado serializado.

    Função de módulo necessária para __reduce__: cloudpickle exige que a função
    de reconstrução seja importável pelo nome, não pode ser lambda nem closure.
    """
    obj = object.__new__(BetaRegressionWrapper)
    obj._feature_cols = state["feature_cols"]
    if state["model_type"] == "ridge":
        obj._model = state["model"]
    else:
        obj._model = {"params": state["params"]}
    return obj


class BetaRegressionWrapper:
    """Wrapper scikit-learn-like em torno da regressão Beta do statsmodels.

    Implementa __reduce__ para serialização segura via pickle/cloudpickle:
    o GLMResultsWrapper do statsmodels guarda file handles internos que
    bloqueiam o cloudpickle. __reduce__ garante que apenas os coeficientes
    (params) sejam serializados, reconstruindo a predição com link logit.
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
            # Fallback: Ridge regression com link logit manual
            logger.warning(
                "Beta regression via statsmodels falhou. Usando Ridge com link logit."
            )
            self._model = Ridge(alpha=1.0)
            self._model.fit(X, np.log(y / (1 - y)))  # logit(LGD)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(self._model, Ridge):
            logit_pred = self._model.predict(X[self._feature_cols])
            return np.clip(1 / (1 + np.exp(-logit_pred)), BETA_EPS, 1 - BETA_EPS)

        if isinstance(self._model, dict):
            # Reconstruído do __setstate__: aplica link logit manualmente
            # Binomial GLM com link logit: μ = sigmoid(Intercept + X @ β)
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

        # GLMResultsWrapper ainda na memória (antes da serialização)
        data = X[self._feature_cols].copy()
        return np.clip(self._model.predict(data), BETA_EPS, 1 - BETA_EPS)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def __reduce__(self) -> tuple:
        """Serialização segura para pickle e cloudpickle.

        cloudpickle prioriza __reduce__ sobre __getstate__/__setstate__.
        Retorna apenas os coeficientes — sem file handles do statsmodels.
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
            # GLMResultsWrapper: extrai só coeficientes (sem file handles)
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
    """Treina regressão Beta para LGD e loga no MLflow.

    Args:
        df: Feature store (usa proxy LGD se coluna 'lgd' ausente).
        run_name: Nome do run no MLflow.

    Returns:
        Modelo LGD treinado.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_LGD)

    # Garante coluna de LGD
    if LGD_COL not in df.columns:
        logger.warning("Coluna 'lgd' ausente — gerando proxy sintético (documentado).")
        df = df.copy()
        df[LGD_COL] = _generate_lgd_proxy(df)

    X = _prepare_lgd_features(df)
    y = df[LGD_COL].clip(BETA_EPS, 1 - BETA_EPS)

    logger.info(f"Treinando LGD — {len(y):,} amostras. LGD médio: {y.mean():.3f}")

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

        # GLMResultsWrapper do statsmodels tem file handles internos que bloqueiam
        # o cloudpickle do MLflow. Converte para dict de params antes de serializar;
        # predict() já trata o caso dict (reconstrução via link logit).
        if hasattr(model._model, "params"):
            model._model = {"params": model._model.params.to_dict()}

        # cloudpickle falha com o file sink do loguru (modo 'a') nos globals do
        # módulo. pickle padrão serializa classes importáveis por nome, sem
        # percorrer globals — resolve o PicklingError do cloudpickle.
        mlflow.sklearn.log_model(
            model,
            artifact_path="lgd_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        run_id = mlflow.active_run().info.run_id

    logger.success(f"Modelo LGD treinado. Run ID: {run_id}")
    return model


def predict_lgd(
    model: BetaRegressionWrapper,
    df: pd.DataFrame,
) -> np.ndarray:
    """Retorna LGD estimado por contrato.

    Args:
        model: Modelo treinado via train_lgd().
        df: DataFrame de features.

    Returns:
        Array de LGD estimado em [0, 1].
    """
    X = _prepare_lgd_features(df)
    lgd = model.predict(X)
    logger.debug(
        f"LGD predito: min={lgd.min():.4f}, max={lgd.max():.4f}, mean={lgd.mean():.4f}"
    )
    return lgd


if __name__ == "__main__":
    from src.features.build_features import load_feature_store

    df = load_feature_store()
    model = train_lgd(df)
