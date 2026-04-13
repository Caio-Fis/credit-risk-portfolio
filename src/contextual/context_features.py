"""Engenharia de features contextuais — Módulo 3.

Produto, prazo e garantia entram como features — não como filtros.
Isso permite ao modelo aprender as interações cliente × contexto.

Funções principais:
- add_context_features: adiciona features numéricas de contexto
- encode_product: one-hot encoding do produto com significado preservado
- create_interaction_features: cria termos de interação explícitos
"""

import numpy as np
import pandas as pd
from loguru import logger

# Mapeamento de produto para risco relativo de longo prazo
PRODUCT_RISK_ORDER = {
    "antecipacao_recebiveis": 0,
    "capital_de_giro": 1,
    "investimento": 2,
}

# Categorias de prazo
TENOR_BINS = [0, 3, 12, 24, 60]
TENOR_LABELS = ["curtissimo", "curto", "medio", "longo"]


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas do contexto da operação.

    Args:
        df: DataFrame com colunas product_type, tenor_months, has_collateral.

    Returns:
        DataFrame com features contextuais adicionadas.
    """
    df = df.copy()

    # Ordinal de risco do produto (0=menor, 2=maior)
    if "product_type" in df.columns:
        df["product_risk_ordinal"] = (
            df["product_type"].map(PRODUCT_RISK_ORDER).fillna(1)
        )

    # Categorias de prazo
    if "tenor_months" in df.columns:
        df["tenor_category"] = pd.cut(
            df["tenor_months"],
            bins=TENOR_BINS,
            labels=TENOR_LABELS,
            right=True,
        )
        df["log_tenor"] = np.log1p(df["tenor_months"])

    # Flag de operação de longo prazo (> 12 meses)
    if "tenor_months" in df.columns:
        df["is_long_term"] = (df["tenor_months"] > 12).astype(int)

    # Razão EAD/faturamento (proxy de endividamento relativo)
    if "ead" in df.columns and "faturamento_anual" in df.columns:
        df["leverage_ratio"] = df["ead"] / (df["faturamento_anual"] + 1)

    logger.info(f"Context features adicionadas: {df.shape[1]} colunas total")
    return df


def encode_product(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encoding do tipo de produto.

    Args:
        df: DataFrame com coluna product_type.
        drop_first: Se True, remove primeira categoria (evita multicolinearidade).

    Returns:
        DataFrame com colunas product_* adicionadas.
    """
    if "product_type" not in df.columns:
        return df

    dummies = pd.get_dummies(
        df["product_type"], prefix="product", drop_first=drop_first
    )
    df = pd.concat([df, dummies], axis=1)
    logger.debug(f"Product encoded: {dummies.columns.tolist()}")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria termos de interação explícitos cliente × contexto.

    As interações mais importantes:
    - score × log(prazo): clientes fracos sofrem mais em prazos longos
    - score × produto: diferentes produtos penalizam clientes de formas distintas
    - log(prazo) × garantia: garantia atenua risco de prazo

    Args:
        df: DataFrame com features de cliente e contexto.

    Returns:
        DataFrame com interações adicionadas.
    """
    df = df.copy()

    score_col = "score_financeiro"
    prazo_col = "log_tenor"
    garantia_col = "has_collateral"
    produto_risco = "product_risk_ordinal"

    if score_col in df.columns and prazo_col in df.columns:
        df["score_x_prazo"] = df[score_col] * df[prazo_col]

    if score_col in df.columns and produto_risco in df.columns:
        df["score_x_produto"] = df[score_col] * df[produto_risco]

    if prazo_col in df.columns and garantia_col in df.columns:
        df["prazo_x_garantia"] = df[prazo_col] * df[garantia_col]

    if (
        score_col in df.columns
        and produto_risco in df.columns
        and prazo_col in df.columns
    ):
        df["score_x_prazo_x_produto"] = (
            df[score_col] * df[prazo_col] * df[produto_risco]
        )

    interaction_cols = [c for c in df.columns if "_x_" in c]
    logger.debug(f"Interações criadas: {interaction_cols}")
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "default",
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepara a matriz de features e o vetor target para modelagem.

    Args:
        df: DataFrame com features e target.
        target_col: Nome da coluna target.
        id_cols: Colunas a excluir (IDs, targets verdadeiros, etc.).

    Returns:
        Tupla (X, y, feature_names).
    """
    if id_cols is None:
        id_cols = ["client_id", "pd_true", "lgd_true", "el_true", target_col]

    exclude = set(id_cols) | {target_col}

    # Aplica pipeline de features
    df_feat = add_context_features(df)
    df_feat = encode_product(df_feat, drop_first=True)
    df_feat = create_interaction_features(df_feat)

    # Remove colunas não-numéricas e IDs
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in exclude
        and df_feat[c].dtype
        in [np.float64, np.float32, np.int64, np.int32, np.int8, bool]
    ]

    X = df_feat[feature_cols].fillna(df_feat[feature_cols].median())
    y = df_feat[target_col]

    logger.info(f"Feature matrix: {X.shape[0]:,} amostras × {X.shape[1]} features")
    return X, y, feature_cols
