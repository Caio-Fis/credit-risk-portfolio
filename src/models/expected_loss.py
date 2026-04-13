"""Cálculo de Expected Loss por contrato.

EL = PD × LGD × EAD

- PD: Probabilidade de Default (modelo calibrado)
- LGD: Loss Given Default (regressão Beta)
- EAD: Exposure at Default — valor exposto ao risco (proxy: AMT_CREDIT)

Funções principais:
- compute_el: calcula EL por contrato em R$
- el_summary: sumariza EL da carteira (total, médio, por decil de risco)
- el_by_segment: breakdown por segmento (produto, prazo, etc.)
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_el(
    pd_proba: np.ndarray | pd.Series,
    lgd: np.ndarray | pd.Series,
    ead: np.ndarray | pd.Series,
) -> pd.Series:
    """Calcula Expected Loss por contrato: EL = PD × LGD × EAD.

    Args:
        pd_proba: Probabilidade de default calibrada (0–1).
        lgd: Loss Given Default estimado (0–1).
        ead: Exposure at Default em R$ (valor do crédito).

    Returns:
        pd.Series com EL em R$ por contrato.
    """
    pd_arr = np.asarray(pd_proba, dtype=float)
    lgd_arr = np.asarray(lgd, dtype=float)
    ead_arr = np.asarray(ead, dtype=float)

    if not (len(pd_arr) == len(lgd_arr) == len(ead_arr)):
        raise ValueError("pd_proba, lgd e ead devem ter o mesmo comprimento.")

    el = pd_arr * lgd_arr * ead_arr

    logger.info(
        f"EL calculado — total R$ {el.sum():,.2f} | "
        f"médio R$ {el.mean():,.2f} | "
        f"max R$ {el.max():,.2f}"
    )
    return pd.Series(el, name="expected_loss")


def el_summary(
    df: pd.DataFrame,
    el_col: str = "expected_loss",
    ead_col: str = "AMT_CREDIT",
) -> pd.DataFrame:
    """Sumariza Expected Loss da carteira.

    Args:
        df: DataFrame com colunas de EL e EAD.
        el_col: Nome da coluna de EL.
        ead_col: Nome da coluna de EAD.

    Returns:
        DataFrame com estatísticas da carteira.
    """
    if el_col not in df.columns:
        raise KeyError(f"Coluna '{el_col}' não encontrada. Rode compute_el() primeiro.")

    total_ead = df[ead_col].sum() if ead_col in df.columns else np.nan
    total_el = df[el_col].sum()
    el_rate = total_el / total_ead if total_ead > 0 else np.nan

    summary = pd.DataFrame(
        [
            {
                "total_ead_R$": total_ead,
                "total_el_R$": total_el,
                "el_rate_%": el_rate * 100,
                "el_mean_R$": df[el_col].mean(),
                "el_median_R$": df[el_col].median(),
                "el_p95_R$": df[el_col].quantile(0.95),
                "el_p99_R$": df[el_col].quantile(0.99),
                "n_contracts": len(df),
            }
        ]
    )

    logger.info(f"\n{summary.T.to_string()}")
    return summary


def el_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    el_col: str = "expected_loss",
    ead_col: str = "AMT_CREDIT",
) -> pd.DataFrame:
    """Breakdown de EL por segmento.

    Args:
        df: DataFrame com EL calculado.
        segment_col: Coluna de segmentação (ex: product_type, tenor_months).
        el_col: Coluna de EL.
        ead_col: Coluna de EAD.

    Returns:
        DataFrame com EL agregado por segmento.
    """
    agg = (
        df.groupby(segment_col)
        .agg(
            n_contracts=(el_col, "count"),
            total_ead=(ead_col, "sum"),
            total_el=(el_col, "sum"),
            el_mean=(el_col, "mean"),
        )
        .assign(el_rate_pct=lambda x: x["total_el"] / x["total_ead"] * 100)
        .sort_values("el_rate_pct", ascending=False)
        .reset_index()
    )
    return agg


def add_el_to_df(
    df: pd.DataFrame,
    pd_col: str = "pd_proba",
    lgd_col: str = "lgd_pred",
    ead_col: str = "AMT_CREDIT",
) -> pd.DataFrame:
    """Adiciona coluna expected_loss ao DataFrame.

    Args:
        df: DataFrame com colunas de PD, LGD e EAD.
        pd_col: Coluna de PD predito.
        lgd_col: Coluna de LGD predito.
        ead_col: Coluna de EAD.

    Returns:
        DataFrame com coluna 'expected_loss' adicionada.
    """
    for col in (pd_col, lgd_col, ead_col):
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada no DataFrame.")

    df = df.copy()
    df["expected_loss"] = compute_el(df[pd_col], df[lgd_col], df[ead_col])
    return df
