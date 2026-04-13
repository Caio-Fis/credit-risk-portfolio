"""Monitoramento de trajetória de score para early warning.

Detecta deterioração de crédito antes do default por meio de queda brusca
no score de PD ao longo do tempo.

Regra operacional:
- Queda > SCORE_DROP_THRESHOLD pontos em SCORE_DROP_WINDOW_DAYS dias → alerta

Funções principais:
- compute_score_trajectory: calcula variação de score por empresa/contrato
- flag_score_drop: identifica empresas com queda superior ao threshold
- score_trend: regressão linear do score para detectar tendência
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.config import SCORE_DROP_THRESHOLD, SCORE_DROP_WINDOW_DAYS


def compute_score_trajectory(
    df: pd.DataFrame,
    entity_col: str = "SK_ID_CURR",
    score_col: str = "pd_score",
    date_col: str = "reference_date",
    window: int = SCORE_DROP_WINDOW_DAYS,
) -> pd.DataFrame:
    """Calcula a variação de score para cada entidade na janela de tempo.

    O score de monitoramento é definido como (1 - PD) × 1000 para
    manter convenção de "score maior = melhor" (estilo bureau).

    Args:
        df: DataFrame com histórico de scores por entidade e data.
        entity_col: Coluna de identificador da entidade.
        score_col: Coluna de score (pd_score = (1-PD)×1000).
        date_col: Coluna de data de referência.
        window: Janela em dias para calcular a variação.

    Returns:
        DataFrame com colunas adicionais: score_start, score_end, score_drop.
    """
    for col in (entity_col, score_col, date_col):
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada no DataFrame.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([entity_col, date_col])

    cutoff_date = df[date_col].max() - pd.Timedelta(days=window)

    results = []
    for entity_id, group in df.groupby(entity_col):
        recent = group[group[date_col] >= cutoff_date]
        historic = group[group[date_col] < cutoff_date]

        if recent.empty or historic.empty:
            continue

        score_end = recent[score_col].iloc[-1]
        score_start = historic[score_col].iloc[-1]
        drop = score_start - score_end  # positivo = queda

        results.append(
            {
                entity_col: entity_id,
                "score_start": score_start,
                "score_end": score_end,
                "score_drop": drop,
                "date_start": historic[date_col].iloc[-1],
                "date_end": recent[date_col].iloc[-1],
            }
        )

    trajectory = pd.DataFrame(results)
    logger.info(
        f"Trajetória calculada para {len(trajectory):,} entidades. "
        f"Drop médio: {trajectory['score_drop'].mean():.1f} pts"
    )
    return trajectory


def flag_score_drop(
    trajectory: pd.DataFrame,
    threshold: int = SCORE_DROP_THRESHOLD,
    entity_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Identifica entidades com queda de score acima do threshold.

    Args:
        trajectory: Resultado de compute_score_trajectory().
        threshold: Queda mínima em pontos para disparar alerta.
        entity_col: Coluna de identificador da entidade.

    Returns:
        DataFrame filtrado com apenas as entidades em alerta,
        ordenado por score_drop decrescente.
    """
    alerts = (
        trajectory[trajectory["score_drop"] >= threshold]
        .sort_values("score_drop", ascending=False)
        .reset_index(drop=True)
    )
    alerts["alert_level"] = alerts["score_drop"].apply(
        lambda x: "crítico" if x >= threshold * 2 else "atenção"
    )

    logger.warning(
        f"Early warning: {len(alerts)} entidades com queda de score "
        f">= {threshold} pts nos últimos {SCORE_DROP_WINDOW_DAYS}d"
    )
    return alerts


def score_trend(
    df: pd.DataFrame,
    entity_id,
    entity_col: str = "SK_ID_CURR",
    score_col: str = "pd_score",
    date_col: str = "reference_date",
) -> dict:
    """Calcula tendência linear do score para uma entidade.

    Args:
        df: DataFrame com histórico de scores.
        entity_id: ID da entidade a analisar.
        entity_col: Coluna de identificador.
        score_col: Coluna de score.
        date_col: Coluna de data.

    Returns:
        Dicionário com slope (pts/mês), r_squared e trend_label.
    """
    entity_df = df[df[entity_col] == entity_id].copy()
    entity_df[date_col] = pd.to_datetime(entity_df[date_col])
    entity_df = entity_df.sort_values(date_col)

    if len(entity_df) < 3:
        return {"slope": 0.0, "r_squared": 0.0, "trend_label": "insuficiente"}

    x = (entity_df[date_col] - entity_df[date_col].min()).dt.days.values.astype(float)
    y = entity_df[score_col].values.astype(float)

    coeffs = np.polyfit(x, y, deg=1)
    slope_per_day = coeffs[0]
    slope_per_month = slope_per_day * 30

    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if slope_per_month < -10:
        trend_label = "deteriorando"
    elif slope_per_month > 10:
        trend_label = "melhorando"
    else:
        trend_label = "estável"

    return {
        "slope_pts_per_month": float(slope_per_month),
        "r_squared": float(r2),
        "trend_label": trend_label,
    }


def plot_score_trajectory(
    df: pd.DataFrame,
    entity_id,
    entity_col: str = "SK_ID_CURR",
    score_col: str = "pd_score",
    date_col: str = "reference_date",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plota a trajetória de score de uma entidade com linha de tendência.

    Args:
        df: DataFrame com histórico de scores.
        entity_id: ID da entidade.
        entity_col: Coluna de identificador.
        score_col: Coluna de score.
        date_col: Coluna de data.
        save_path: Se fornecido, salva a figura.

    Returns:
        Figura matplotlib.
    """
    entity_df = df[df[entity_col] == entity_id].copy()
    entity_df[date_col] = pd.to_datetime(entity_df[date_col])
    entity_df = entity_df.sort_values(date_col)

    trend = score_trend(df, entity_id, entity_col, score_col, date_col)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        entity_df[date_col],
        entity_df[score_col],
        "o-",
        color="steelblue",
        label="Score",
    )
    ax.axhline(
        y=entity_df[score_col].iloc[-1],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Score atual",
    )

    ax.set_title(
        f"Trajetória de Score — {entity_col}={entity_id} | Tendência: {trend['trend_label']}"
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Score (0–1000)")
    ax.set_ylim(0, 1000)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
