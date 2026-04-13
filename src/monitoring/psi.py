"""Population Stability Index (PSI) por feature.

PSI mede o quão diferente é a distribuição atual de uma feature
em relação à distribuição de referência (treinamento).

Interpretação:
- PSI < 0.10  → estável (sem ação necessária)
- PSI 0.10–0.20 → atenção (investigar causa)
- PSI > 0.20  → drift confirmado (retreino necessário)

Funções principais:
- compute_psi: PSI para uma única feature
- psi_all_features: PSI para todas as features numéricas de dois DataFrames
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PSI_ATTENTION, PSI_STABLE

EPS = 1e-6  # evita log(0)


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Calcula PSI entre distribuição esperada e atual.

    Args:
        expected: Distribuição de referência (ex: dados de treinamento).
        actual: Distribuição atual (ex: dados de produção).
        bins: Número de bins para discretização.

    Returns:
        PSI score (float >= 0).
    """
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)

    # Remove NaNs
    expected_clean = expected_arr[~np.isnan(expected_arr)]
    actual_clean = actual_arr[~np.isnan(actual_arr)]

    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0

    # Usa quantis da distribuição de referência para definir os bins
    breakpoints = np.nanpercentile(expected_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicatas (features constantes)

    if len(breakpoints) < 2:
        return 0.0

    # Conta frequências por bin
    expected_counts = np.histogram(expected_clean, bins=breakpoints)[0]
    actual_counts = np.histogram(actual_clean, bins=breakpoints)[0]

    # Converte para proporções (soma = 1)
    expected_pct = (expected_counts / len(expected_clean)).clip(EPS)
    actual_pct = (actual_counts / len(actual_clean)).clip(EPS)

    # PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def psi_all_features(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    bins: int = 10,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Calcula PSI para todas as features e classifica status.

    Args:
        ref_df: DataFrame de referência (treinamento).
        curr_df: DataFrame atual (produção/validação).
        bins: Número de bins por feature.
        numeric_only: Se True, processa apenas colunas numéricas.

    Returns:
        DataFrame com colunas: feature, psi, status (stable/attention/drift).
    """
    if numeric_only:
        cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in ref_df.columns if c in curr_df.columns]

    records = []
    for col in cols:
        if col not in curr_df.columns:
            continue
        psi_val = compute_psi(ref_df[col], curr_df[col], bins=bins)
        records.append({"feature": col, "psi": psi_val})

    result = (
        pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)
    )
    result["status"] = result["psi"].apply(_classify_psi)

    n_drift = (result["status"] == "drift").sum()
    n_attention = (result["status"] == "attention").sum()
    logger.info(
        f"PSI calculado — {n_drift} features em drift, {n_attention} em atenção"
    )

    return result


def _classify_psi(psi_val: float) -> str:
    """Classifica o PSI conforme thresholds operacionais."""
    if psi_val < PSI_STABLE:
        return "stable"
    elif psi_val < PSI_ATTENTION:
        return "attention"
    else:
        return "drift"
