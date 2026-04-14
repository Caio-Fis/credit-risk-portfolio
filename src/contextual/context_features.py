"""Contextual feature engineering — Module 3.

Product, tenor and collateral enter as features — not as filters.
This allows the model to learn client × context interactions.

Main functions:
- add_context_features: adds numerical context features
- encode_product: one-hot encoding of product preserving meaning
- create_interaction_features: creates explicit interaction terms
"""

import numpy as np
import pandas as pd
from loguru import logger

# Product-to-long-term-risk mapping
PRODUCT_RISK_ORDER = {
    "receivables_advance": 0,
    "working_capital": 1,
    "investment": 2,
}

# Tenor categories
TENOR_BINS = [0, 3, 12, 24, 60]
TENOR_LABELS = ["very_short", "short", "medium", "long"]


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features from the operation context.

    Args:
        df: DataFrame with columns product_type, tenor_months, has_collateral.

    Returns:
        DataFrame with contextual features added.
    """
    df = df.copy()

    # Product risk ordinal (0=lower, 2=higher)
    if "product_type" in df.columns:
        df["product_risk_ordinal"] = (
            df["product_type"].map(PRODUCT_RISK_ORDER).fillna(1)
        )

    # Tenor categories
    if "tenor_months" in df.columns:
        df["tenor_category"] = pd.cut(
            df["tenor_months"],
            bins=TENOR_BINS,
            labels=TENOR_LABELS,
            right=True,
        )
        df["log_tenor"] = np.log1p(df["tenor_months"])

    # Long-term operation flag (> 12 months)
    if "tenor_months" in df.columns:
        df["is_long_term"] = (df["tenor_months"] > 12).astype(int)

    # EAD/revenue ratio (relative leverage proxy)
    if "ead" in df.columns and "faturamento_anual" in df.columns:
        df["leverage_ratio"] = df["ead"] / (df["faturamento_anual"] + 1)

    logger.info(f"Context features added: {df.shape[1]} total columns")
    return df


def encode_product(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encoding of product type.

    Args:
        df: DataFrame with column product_type.
        drop_first: If True, drops first category (avoids multicollinearity).

    Returns:
        DataFrame with product_* columns added.
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
    """Creates explicit client × context interaction terms.

    Most important interactions:
    - score × log(tenor): weak clients suffer more at long tenors
    - score × product: different products penalise clients differently
    - log(tenor) × collateral: collateral mitigates tenor risk

    Args:
        df: DataFrame with client and context features.

    Returns:
        DataFrame with interactions added.
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
    logger.debug(f"Interactions created: {interaction_cols}")
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "default",
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepares the feature matrix and target vector for modelling.

    Args:
        df: DataFrame with features and target.
        target_col: Target column name.
        id_cols: Columns to exclude (IDs, true targets, etc.).

    Returns:
        Tuple (X, y, feature_names).
    """
    if id_cols is None:
        id_cols = ["client_id", "pd_true", "lgd_true", "el_true", target_col]

    exclude = set(id_cols) | {target_col}

    # Apply feature pipeline
    df_feat = add_context_features(df)
    df_feat = encode_product(df_feat, drop_first=True)
    df_feat = create_interaction_features(df_feat)

    # Remove non-numeric columns and IDs
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in exclude
        and df_feat[c].dtype
        in [np.float64, np.float32, np.int64, np.int32, np.int8, bool]
    ]

    X = df_feat[feature_cols].fillna(df_feat[feature_cols].median())
    y = df_feat[target_col]

    logger.info(f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    return X, y, feature_cols
