"""Detecção e reporte de drift de dados.

Separa dois tipos de deterioração:
1. Drift de população: mudança na distribuição das features (PSI)
2. Drift de modelo: degradação de métricas (AUROC, KS)

Funções principais:
- detect_drift: classifica status de uma série de PSI
- drift_report: relatório completo de drift para um período
- plot_psi_heatmap: mapa de calor de PSI por feature e período
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from src.config import PSI_ATTENTION, PSI_STABLE
from src.monitoring.psi import psi_all_features


def detect_drift(psi_series: pd.Series) -> pd.Series:
    """Classifica o status de drift para uma série de PSI values.

    Args:
        psi_series: pd.Series com PSI por feature (índice = feature, valor = PSI).

    Returns:
        pd.Series com status: 'stable', 'attention' ou 'drift'.
    """

    def _classify(v: float) -> str:
        if v < PSI_STABLE:
            return "stable"
        elif v < PSI_ATTENTION:
            return "attention"
        return "drift"

    return psi_series.map(_classify)


def drift_report(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    period_label: str = "current",
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Gera relatório completo de drift entre duas populações.

    Args:
        ref_df: DataFrame de referência (treinamento).
        curr_df: DataFrame atual.
        period_label: Rótulo do período atual (ex: "2024-Q1").
        save_path: Se fornecido, salva o relatório em CSV.

    Returns:
        DataFrame com feature, psi, status, delta_mean, delta_std.
    """
    psi_df = psi_all_features(ref_df, curr_df)

    numeric_cols = [
        c
        for c in psi_df["feature"].tolist()
        if c in ref_df.columns and c in curr_df.columns
    ]

    delta_stats = []
    for col in numeric_cols:
        ref_mean = ref_df[col].mean()
        curr_mean = curr_df[col].mean()
        ref_std = ref_df[col].std()
        curr_std = curr_df[col].std()
        delta_stats.append(
            {
                "feature": col,
                "ref_mean": ref_mean,
                "curr_mean": curr_mean,
                "delta_mean": curr_mean - ref_mean,
                "ref_std": ref_std,
                "curr_std": curr_std,
            }
        )

    delta_df = pd.DataFrame(delta_stats)
    report = psi_df.merge(delta_df, on="feature", how="left")
    report["period"] = period_label

    n_drift = (report["status"] == "drift").sum()
    n_attention = (report["status"] == "attention").sum()
    n_stable = (report["status"] == "stable").sum()

    logger.info(
        f"Relatório de drift [{period_label}] — "
        f"Drift: {n_drift} | Atenção: {n_attention} | Estável: {n_stable}"
    )

    if n_drift > 0:
        drift_features = report[report["status"] == "drift"]["feature"].tolist()
        logger.warning(f"Features em drift: {drift_features}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(save_path, index=False)
        logger.info(f"Relatório salvo em {save_path}")

    return report


def plot_psi_heatmap(
    reports: list[tuple[str, pd.DataFrame]],
    top_n: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """Mapa de calor de PSI por feature e período.

    Args:
        reports: Lista de (period_label, drift_report_df).
        top_n: Número de features com maior PSI médio a exibir.
        save_path: Se fornecido, salva a figura.

    Returns:
        Figura matplotlib.
    """

    pivot = pd.concat(
        [
            df[["feature", "psi"]].set_index("feature").rename(columns={"psi": label})
            for label, df in reports
        ],
        axis=1,
    )

    # Seleciona top N por PSI médio
    pivot = pivot.loc[pivot.mean(axis=1).nlargest(top_n).index]

    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 1.5), max(6, top_n * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.25)
    plt.colorbar(im, ax=ax, label="PSI")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Linhas de threshold
    for threshold, color, label in [
        (PSI_STABLE, "green", f"Estável (<{PSI_STABLE})"),
        (PSI_ATTENTION, "orange", f"Atenção (<{PSI_ATTENTION})"),
    ]:
        pass  # linhas seriam no colorbar — mantém simples

    ax.set_title("PSI por Feature e Período")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PSI heatmap salvo em {save_path}")

    return fig
