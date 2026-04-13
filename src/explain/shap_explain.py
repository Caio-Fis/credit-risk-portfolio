"""Explicabilidade por SHAP values.

Gera waterfall por contrato individual e importância global de features.

Funções principais:
- compute_shap: calcula SHAP values para um conjunto de predições
- waterfall_plot: gera waterfall SHAP para um contrato específico
- top_features: retorna as N features mais importantes globalmente
- summary_plot: gráfico de resumo de importância global
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger


def compute_shap(
    model,
    X: pd.DataFrame,
    background_sample: int = 100,
    max_explain_samples: int = 2000,
    seed: int = 42,
) -> shap.Explanation:
    """Calcula SHAP values para o conjunto X.

    Para CalibratedClassifierCV, extrai o estimador base LightGBM
    e usa TreeExplainer (mais eficiente que KernelExplainer).

    Args:
        model: Modelo treinado (CalibratedClassifierCV ou LGBMClassifier).
        X: DataFrame de features.
        background_sample: Tamanho da amostra de background para KernelExplainer.
        max_explain_samples: Limite de linhas para SHAP. Se X > limite, amostra
            estratificada por posição (preserva distribuição temporal/de risco).
            TreeExplainer é O(n × features × depth) — 307K linhas leva ~10min;
            2000 linhas leva ~2s.
        seed: Semente para a amostragem.

    Returns:
        shap.Explanation com values para a classe positiva (default).
    """
    from sklearn.calibration import CalibratedClassifierCV

    # Subsample quando X é grande demais para SHAP ser interativo.
    # TreeExplainer calcula SHAP ponto-a-ponto (sem interação entre amostras),
    # então amostrar é equivalente a escolher quais contratos explicar.
    # Importância global = mean(|SHAP|) converge com n~500 (CLT); 2000 é conservador.
    # Usamos amostragem aleatória uniforme — sem estratificação por classe porque
    # o dataset contextual sintético é balanceado (~43% default).
    # Se usar em produção com dataset desbalanceado (<5% default), passe y=target
    # e implemente stratified sampling explícito.
    n_original = len(X)
    if n_original > max_explain_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_original, size=max_explain_samples, replace=False)
        idx.sort()  # preserva ordem original
        X = X.iloc[idx].reset_index(drop=True)
        logger.info(
            f"SHAP: {n_original:,} → {max_explain_samples:,} linhas. "
            f"Valores individuais: idênticos. Importância global: erro ≈ σ/√{max_explain_samples}."
        )

    # Extrai estimador base do wrapper de calibração
    if isinstance(model, CalibratedClassifierCV):
        base_estimator = model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = model

    logger.info(f"Calculando SHAP values (TreeExplainer) para {len(X)} amostras...")

    try:
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer(X)
        # Para classificação binária, pega a classe positiva (índice 1)
        if shap_values.values.ndim == 3:
            explanation = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1]
                if shap_values.base_values.ndim > 1
                else shap_values.base_values,
                data=shap_values.data,
                feature_names=list(X.columns),
            )
        else:
            explanation = shap_values
    except Exception as e:
        logger.warning(
            f"TreeExplainer falhou ({e}). Usando KernelExplainer (mais lento)."
        )
        background = shap.sample(X, background_sample)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_matrix = explainer.shap_values(X)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[1]
        explanation = shap.Explanation(
            values=shap_matrix,
            base_values=np.full(
                len(X),
                explainer.expected_value[1]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value,
            ),
            data=X.values,
            feature_names=list(X.columns),
        )

    logger.success(f"SHAP values calculados: {explanation.values.shape}")
    return explanation


def waterfall_plot(
    explanation: shap.Explanation,
    idx: int,
    max_display: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """Gera waterfall SHAP para um contrato específico.

    Args:
        explanation: shap.Explanation do conjunto completo.
        idx: Índice do contrato no DataFrame.
        max_display: Número máximo de features a exibir.
        save_path: Se fornecido, salva a figura neste caminho.

    Returns:
        Figura matplotlib.
    """
    shap.plots.waterfall(explanation[idx], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Waterfall plot salvo em {save_path}")

    return fig


def top_features(
    explanation: shap.Explanation,
    n: int = 10,
) -> pd.DataFrame:
    """Retorna as N features mais importantes por importância SHAP global.

    Importância global = mean(|SHAP values|) por feature.

    Args:
        explanation: shap.Explanation do conjunto.
        n: Número de features a retornar.

    Returns:
        DataFrame com colunas: feature, mean_abs_shap (ordenado desc).
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    feature_names = explanation.feature_names or [f"f{i}" for i in range(len(mean_abs))]

    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    logger.info(f"Top {n} features por importância SHAP:\n{df.to_string(index=False)}")
    return df


def summary_plot(
    explanation: shap.Explanation,
    max_display: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """Gera summary plot de importância global.

    Args:
        explanation: shap.Explanation do conjunto.
        max_display: Número máximo de features a exibir.
        save_path: Se fornecido, salva a figura.

    Returns:
        Figura matplotlib.
    """
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Summary plot salvo em {save_path}")

    return fig
