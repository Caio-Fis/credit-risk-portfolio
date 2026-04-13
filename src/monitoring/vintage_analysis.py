"""Análise de safra (vintage analysis).

Mede a inadimplência acumulada por coorte de originação e maturidade.
Permite identificar se o modelo está deteriorando ou se a carteira mudou.

Terminologia:
- Safra (vintage): mês de originação do contrato
- Maturidade: meses desde a originação
- DPD: Days Past Due (dias em atraso)

Funções principais:
- build_vintage_matrix: constrói matriz de inadimplência (safra × maturidade)
- plot_vintage_curves: plota curvas de inadimplência por safra
- compare_vintages: compara safras novas vs antigas
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def build_vintage_matrix(
    df: pd.DataFrame,
    vintage_col: str = "vintage",
    maturity_col: str = "maturity_month",
    default_col: str = "is_default",
    min_observations: int = 30,
) -> pd.DataFrame:
    """Constrói matriz de inadimplência acumulada por safra e maturidade.

    Args:
        df: DataFrame com colunas de safra, maturidade e flag de default.
        vintage_col: Coluna indicando o período de originação.
        maturity_col: Coluna com meses desde a originação.
        default_col: Coluna binária (1 = default ocorreu até este mês).
        min_observations: Safras com menos amostras são excluídas.

    Returns:
        Pivot table: linhas = safra, colunas = maturidade, valores = taxa de default.
    """
    for col in (vintage_col, maturity_col, default_col):
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada no DataFrame.")

    # Filtra safras com observações suficientes
    vintage_counts = df.groupby(vintage_col)[default_col].count()
    valid_vintages = vintage_counts[vintage_counts >= min_observations].index
    df_filtered = df[df[vintage_col].isin(valid_vintages)]

    if len(df_filtered) < len(df):
        n_removed = len(df) - len(df_filtered)
        logger.warning(
            f"{n_removed} registros removidos (safras com < {min_observations} obs)."
        )

    # Taxa de default acumulada por safra × maturidade
    matrix = (
        df_filtered.groupby([vintage_col, maturity_col])[default_col]
        .mean()
        .unstack(maturity_col)
        .sort_index()
    )

    # Garante monoticidade: taxa de default é acumulada
    matrix = matrix.cummax(axis=1)

    logger.info(
        f"Vintage matrix: {matrix.shape[0]} safras × {matrix.shape[1]} maturidades"
    )
    return matrix


def plot_vintage_curves(
    vintage_matrix: pd.DataFrame,
    highlight_recent: int = 3,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plota curvas de inadimplência por safra.

    Args:
        vintage_matrix: Resultado de build_vintage_matrix().
        highlight_recent: Número de safras mais recentes a destacar.
        save_path: Se fornecido, salva a figura.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_vintages = len(vintage_matrix)
    colors = cm.Blues(np.linspace(0.3, 1.0, n_vintages))

    for i, (vintage, row) in enumerate(vintage_matrix.iterrows()):
        values = row.dropna()
        is_recent = i >= (n_vintages - highlight_recent)

        ax.plot(
            values.index,
            values.values * 100,
            color=colors[i],
            linewidth=2.5 if is_recent else 1.0,
            alpha=1.0 if is_recent else 0.4,
            label=str(vintage) if is_recent else None,
        )

    ax.set_xlabel("Maturidade (meses)")
    ax.set_ylabel("Taxa de Default Acumulada (%)")
    ax.set_title("Análise de Safra — Inadimplência por Coorte de Originação")
    ax.grid(alpha=0.3)

    if highlight_recent > 0:
        ax.legend(title="Safras recentes", loc="upper left")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Vintage curves salvo em {save_path}")

    return fig


def compare_vintages(
    vintage_matrix: pd.DataFrame,
    reference_period: int = 6,
    comparison_maturity: int = 12,
) -> pd.DataFrame:
    """Compara taxa de default em maturidade fixa entre safras.

    Útil para identificar deterioração ao longo do tempo.

    Args:
        vintage_matrix: Resultado de build_vintage_matrix().
        reference_period: Número de safras antigas a usar como referência.
        comparison_maturity: Maturidade (meses) usada para comparação.

    Returns:
        DataFrame com default rate por safra na maturidade especificada.
    """
    if comparison_maturity not in vintage_matrix.columns:
        available = vintage_matrix.columns.tolist()
        nearest = min(available, key=lambda x: abs(x - comparison_maturity))
        logger.warning(
            f"Maturidade {comparison_maturity} não disponível. Usando {nearest}."
        )
        comparison_maturity = nearest

    rates = vintage_matrix[comparison_maturity].dropna().reset_index()
    rates.columns = ["vintage", "default_rate"]
    rates["default_rate_pct"] = rates["default_rate"] * 100

    ref_rate = rates.head(reference_period)["default_rate"].mean()
    rates["vs_reference_pp"] = (rates["default_rate"] - ref_rate) * 100

    logger.info(
        f"Taxa de referência (primeiras {reference_period} safras) em {comparison_maturity}m: "
        f"{ref_rate:.2%}"
    )

    return rates


def simulate_vintage_data(
    n_contracts: int = 10000,
    n_vintages: int = 12,
    max_maturity: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dados sintéticos de safra para demonstração.

    Simula deterioração gradual: safras mais novas têm PD ligeiramente maior.

    Args:
        n_contracts: Total de contratos.
        n_vintages: Número de safras (meses).
        max_maturity: Maturidade máxima em meses.
        seed: Semente aleatória.

    Returns:
        DataFrame com vintage, maturity_month, is_default.
    """
    rng = np.random.default_rng(seed)
    records = []

    base_pd = 0.08
    drift_per_vintage = 0.005  # deterioração de 0.5pp por safra

    contracts_per_vintage = n_contracts // n_vintages

    for v in range(n_vintages):
        vintage_pd = base_pd + v * drift_per_vintage
        maturities_observed = n_vintages - v  # safras mais novas têm menos histórico

        for _ in range(contracts_per_vintage):
            # Mês de default (se ocorrer)
            default_month = (
                int(rng.geometric(p=vintage_pd / max_maturity))
                if rng.random() < vintage_pd
                else None
            )

            for m in range(1, min(maturities_observed, max_maturity) + 1):
                records.append(
                    {
                        "vintage": v + 1,
                        "maturity_month": m,
                        "is_default": 1
                        if (default_month is not None and default_month <= m)
                        else 0,
                    }
                )

    return pd.DataFrame(records)
