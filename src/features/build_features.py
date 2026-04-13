"""Construção da feature store batch com janelas temporais 30/90/365 dias.

Funções principais:
- merge_tables: une application_train com tabelas auxiliares do Home Credit
- build_temporal_features: agrega features por janelas temporais
- save_feature_store: persiste a feature store em parquet
- load_feature_store: carrega a feature store já construída

Tabelas grandes (>10M linhas) são agregadas via batch chunked em
batch_aggregations.py para evitar OOM — o pandas nunca materializa as
27M ou 13.6M linhas de uma vez; processa em lotes de 1M linhas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROCESSED_DIR, RAW_DIR, TEMPORAL_WINDOWS
from src.features.batch_aggregations import (
    build_bureau_features_batch,
    build_credit_card_features_batch,
    build_installment_features_batch,
    build_pos_cash_features_batch,
)

HOME_CREDIT_DIR = RAW_DIR / "home_credit"


def _load_raw(filename: str, data_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} não encontrado em {data_dir}. Rode `make data` primeiro."
        )
    logger.debug(f"Carregando {filename}...")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Features de aplicações anteriores (tabela pequena — mantém em pandas)
# ---------------------------------------------------------------------------


def _build_prev_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Conta aplicações e recusas por janela (proxy: DAYS_DECISION)."""
    prev = prev.copy()
    prev["is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    rows = []
    for w in TEMPORAL_WINDOWS:
        mask = prev["DAYS_DECISION"] >= -w
        sub = prev[mask]

        agg = (
            sub.groupby("SK_ID_CURR")
            .agg(
                cnt_applications=("SK_ID_PREV", "nunique"),
                cnt_refused=("is_refused", "sum"),
            )
            .rename(columns=lambda c: f"prev_{c}_{w}d")
            .reset_index()
        )
        rows.append(agg)

    result = rows[0]
    for df in rows[1:]:
        result = result.merge(df, on="SK_ID_CURR", how="outer")

    return result.fillna(0)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def merge_tables(data_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    """Une application_train com features das tabelas auxiliares.

    Tabelas pequenas (<2M linhas) são processadas em pandas.
    Tabelas grandes (bureau_balance 27M, installments 13.6M) são delegadas
    ao Spark via spark_aggregations para evitar OOM.

    Args:
        data_dir: Diretório com os CSVs brutos.

    Returns:
        DataFrame amplo com features de todas as fontes.
    """
    app = _load_raw("application_train.csv", data_dir)
    logger.info(f"application_train: {app.shape}")

    # Pandas: previous_application é pequeno (~887K linhas), sem risco de OOM
    prev = _load_raw("previous_application.csv", data_dir)
    prev_feats = _build_prev_application_features(prev)
    del prev  # libera antes de iniciar o Spark

    # Batch chunked: bureau_balance (27M) + installments (13.6M) em lotes de 1M linhas
    bureau_feats = build_bureau_features_batch(
        bureau_path=data_dir / "bureau.csv",
        bureau_bal_path=data_dir / "bureau_balance.csv",
        windows=TEMPORAL_WINDOWS,
    )
    inst_feats = build_installment_features_batch(
        inst_path=data_dir / "installments_payments.csv",
        windows=TEMPORAL_WINDOWS,
    )
    pos_feats = build_pos_cash_features_batch(
        pos_path=data_dir / "POS_CASH_balance.csv",
        windows=TEMPORAL_WINDOWS,
    )
    cc_feats = build_credit_card_features_batch(
        cc_path=data_dir / "credit_card_balance.csv",
        windows=TEMPORAL_WINDOWS,
    )

    df = (
        app.merge(bureau_feats, on="SK_ID_CURR", how="left")
        .merge(prev_feats, on="SK_ID_CURR", how="left")
        .merge(inst_feats, on="SK_ID_CURR", how="left")
        .merge(pos_feats, on="SK_ID_CURR", how="left")
        .merge(cc_feats, on="SK_ID_CURR", how="left")
    )

    logger.info(f"Tabela unificada: {df.shape}")
    return df


def build_temporal_features(
    df: pd.DataFrame,
    windows: list[int] = TEMPORAL_WINDOWS,
) -> pd.DataFrame:
    """Adiciona features derivadas sobre o DataFrame já unificado.

    Inclui:
    - Razões de crédito (crédito/renda, anuidade/renda)
    - Flags de emprego anômalo
    - Score externo médio

    Args:
        df: DataFrame resultado de merge_tables().
        windows: Janelas temporais a considerar (informativo, não recomputa).

    Returns:
        DataFrame com features adicionais.
    """
    df = df.copy()

    # Razões financeiras
    df["credit_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["credit_annuity_ratio"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    # Idade e tempo de emprego em anos
    df["age_years"] = -df["DAYS_BIRTH"] / 365
    df["employed_years"] = np.where(
        df["DAYS_EMPLOYED"] == 365243,  # código para "não empregado"
        0,
        -df["DAYS_EMPLOYED"] / 365,
    )
    df["employed_to_age_ratio"] = df["employed_years"] / (df["age_years"] + 1)

    # Score externo: agregações e interações
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available = [c for c in ext_cols if c in df.columns]
    if available:
        df["ext_source_mean"] = df[available].mean(axis=1)
        df["ext_source_min"] = df[available].min(axis=1)
        df["ext_source_max"] = df[available].max(axis=1)
        # Produto dos EXT_SOURCE — captura sinal conjunto; top feature em Kaggle Home Credit
        df["ext_source_product"] = df[available].prod(axis=1)
        # Interação pairwise entre os dois scores mais preditivos
        if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
            df["ext_source_2x3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # Taxa de inadimplência no bureau por janela
    for w in windows:
        cnt_col = f"bureau_cnt_credits_{w}d"
        ovd_col = f"bureau_cnt_overdue_{w}d"
        if cnt_col in df.columns and ovd_col in df.columns:
            df[f"bureau_overdue_rate_{w}d"] = (
                df[ovd_col] / (df[cnt_col] + 1)
            )

    # Taxa de recusa em aplicações anteriores por janela
    for w in windows:
        app_col = f"prev_cnt_applications_{w}d"
        ref_col = f"prev_cnt_refused_{w}d"
        if app_col in df.columns and ref_col in df.columns:
            df[f"prev_refusal_rate_{w}d"] = (
                df[ref_col] / (df[app_col] + 1)
            )

    # Inadimplência no círculo social (proxy de rede de risco)
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["social_circle_default_rate"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
        )

    # LTV: razão crédito / bem financiado
    if "AMT_GOODS_PRICE" in df.columns:
        df["credit_goods_ratio"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    # Renda per capita familiar
    if "CNT_FAM_MEMBERS" in df.columns:
        df["income_per_family_member"] = (
            df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
        )

    # Flag explícito de desempregado (365243 é código especial — o flag tem sinal diferente do 0)
    df["flag_unemployed"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)

    logger.info(f"Features construídas: {df.shape[1]} colunas")
    return df


def save_feature_store(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DIR / "feature_store.parquet",
) -> Path:
    """Persiste a feature store em parquet.

    Args:
        df: DataFrame com todas as features.
        output_path: Caminho de saída.

    Returns:
        Path do arquivo salvo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(
        f"Feature store salva: {output_path} ({df.shape[0]:,} × {df.shape[1]})"
    )
    return output_path


def load_feature_store(
    path: Path = PROCESSED_DIR / "feature_store.parquet",
) -> pd.DataFrame:
    """Carrega a feature store já construída.

    Args:
        path: Caminho do arquivo parquet.

    Returns:
        DataFrame com todas as features.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Feature store não encontrada em {path}. Rode `make features`."
        )
    df = pd.read_parquet(path)
    logger.info(f"Feature store carregada: {df.shape[0]:,} × {df.shape[1]}")
    return df


if __name__ == "__main__":
    df_raw = merge_tables()
    df_features = build_temporal_features(df_raw)
    save_feature_store(df_features)
