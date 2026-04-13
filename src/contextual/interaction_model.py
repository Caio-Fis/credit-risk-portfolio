"""Modelo contextual: mesmo cliente, produtos e prazos diferentes.

Demonstra quantitativamente por que o score único é insuficiente:
- Score único: ignora produto e prazo
- Score contextual: incorpora produto, prazo e interações

O score único não é errado — é insuficiente.
No modelo contextual, o score único se torna uma feature entre muitas.

Funções principais:
- train_contextual: treina modelo com features contextuais
- compare_single_vs_contextual: tabela comparativa de Expected Loss
- score_by_context: tabela de score por cliente × produto × prazo
"""

import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import PRODUCTS, TENORS_MONTHS
from src.contextual.context_features import get_feature_matrix
from src.contextual.data_generator import DGP_PARAMS


def train_contextual(
    df: pd.DataFrame,
    target_col: str = "default",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[CalibratedClassifierCV, list[str]]:
    """Treina modelo de PD contextual com LightGBM + Platt scaling.

    Args:
        df: Dataset gerado por generate_dataset() com features contextuais.
        target_col: Coluna target (default).
        test_size: Proporção de dados para teste.
        seed: Semente aleatória.

    Returns:
        Tupla (modelo_calibrado, feature_names).
    """
    X, y, feature_names = get_feature_matrix(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model = CalibratedClassifierCV(lgbm, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba_test)
    logger.success(f"Modelo contextual treinado — AUROC: {auroc:.4f}")

    return model, feature_names


def train_single_score(
    df: pd.DataFrame,
    target_col: str = "default",
    seed: int = 42,
) -> CalibratedClassifierCV:
    """Treina modelo baseline sem features contextuais (score único).

    Usa apenas features do cliente, sem produto/prazo/garantia.

    Args:
        df: Dataset com features de cliente.
        target_col: Coluna target.
        seed: Semente aleatória.

    Returns:
        Modelo calibrado (score único).
    """
    # log_tenor só existe após add_context_features() — usa apenas features
    # presentes no dataset bruto gerado por generate_dataset()
    client_features = ["score_financeiro", "idade_empresa_anos"]
    available = [c for c in client_features if c in df.columns]

    X = df[available].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(scaler.fit_transform(X_train), y_train)

    proba_test = lr.predict_proba(scaler.transform(X_test))[:, 1]
    auroc = roc_auc_score(y_test, proba_test)
    logger.info(f"Score único baseline — AUROC: {auroc:.4f} (esperado ser menor)")

    # Wrapper para interface uniforme
    class SingleScoreWrapper:
        def __init__(self, lr, scaler):
            self._lr = lr
            self._scaler = scaler
            self._available = available

        def predict_proba(self, X):
            X_scaled = self._scaler.transform(X[self._available].fillna(0))
            return self._lr.predict_proba(X_scaled)

    return SingleScoreWrapper(lr, scaler)


def score_by_context(
    client_profile: dict,
    model_contextual: CalibratedClassifierCV,
    feature_names: list[str],
    lgd_base: dict = DGP_PARAMS["lgd_base"],
) -> pd.DataFrame:
    """Calcula PD e EL para um cliente em todos os contextos (produto × prazo).

    Args:
        client_profile: Dicionário com features do cliente.
        model_contextual: Modelo contextual treinado.
        feature_names: Lista de features usadas pelo modelo.
        lgd_base: LGD base por produto (do DGP).

    Returns:
        DataFrame com PD, LGD e EL por produto × prazo.
    """
    from src.contextual.context_features import (
        add_context_features,
        create_interaction_features,
        encode_product,
    )

    rows = []
    for produto in PRODUCTS:
        for prazo in TENORS_MONTHS:
            row = {**client_profile, "product_type": produto, "tenor_months": prazo}
            rows.append(row)

    df_grid = pd.DataFrame(rows)
    df_grid = add_context_features(df_grid)
    df_grid = encode_product(df_grid, drop_first=True)
    df_grid = create_interaction_features(df_grid)

    # Garante que todas as features existam
    for col in feature_names:
        if col not in df_grid.columns:
            df_grid[col] = 0

    X = df_grid[feature_names].fillna(0)
    pd_pred = model_contextual.predict_proba(X)[:, 1]

    df_grid["pd_contextual"] = pd_pred
    df_grid["lgd_estimate"] = df_grid["product_type"].map(lgd_base)
    ead = client_profile.get("ead", 100_000)
    df_grid["el_contextual"] = df_grid["pd_contextual"] * df_grid["lgd_estimate"] * ead

    result = df_grid[
        [
            "product_type",
            "tenor_months",
            "pd_contextual",
            "lgd_estimate",
            "el_contextual",
        ]
    ]
    return result.sort_values(["product_type", "tenor_months"])


def compare_single_vs_contextual(
    df: pd.DataFrame,
    n_examples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Demonstração quantitativa: score único vs score contextual.

    Treina ambos os modelos e compara o Expected Loss calculado.
    O argumento central: com score único, a decisão de limite/taxa
    para capital de giro 30d e investimento 48m é a mesma — o que é errado.

    Args:
        df: Dataset sintético gerado por generate_dataset().
        n_examples: Número de clientes a mostrar na comparação.
        seed: Semente aleatória.

    Returns:
        DataFrame com colunas: client_id, product, tenor,
        pd_single, pd_contextual, el_single, el_contextual, el_delta.
    """
    model_ctx, feature_names = train_contextual(df, seed=seed)
    model_single = train_single_score(df, seed=seed)

    from src.contextual.context_features import (
        add_context_features,
        create_interaction_features,
        encode_product,
    )

    df_feat = add_context_features(df.copy())
    df_feat = encode_product(df_feat, drop_first=True)
    df_feat = create_interaction_features(df_feat)

    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0

    X_ctx = df_feat[feature_names].fillna(0)
    df_feat["pd_contextual"] = model_ctx.predict_proba(X_ctx)[:, 1]
    df_feat["pd_single"] = model_single.predict_proba(df_feat)[:, 1]

    df_feat["lgd_estimate"] = df_feat["product_type"].map(DGP_PARAMS["lgd_base"])
    df_feat["el_contextual"] = (
        df_feat["pd_contextual"] * df_feat["lgd_estimate"] * df_feat["ead"]
    )
    df_feat["el_single"] = (
        df_feat["pd_single"] * df_feat["lgd_estimate"] * df_feat["ead"]
    )
    df_feat["el_delta"] = df_feat["el_contextual"] - df_feat["el_single"]
    df_feat["el_delta_pct"] = df_feat["el_delta"] / (df_feat["el_single"] + 1) * 100

    sample_clients = (
        df_feat["client_id"].drop_duplicates().sample(n=n_examples, random_state=seed)
    )
    comparison = df_feat[df_feat["client_id"].isin(sample_clients)][
        [
            "client_id",
            "product_type",
            "tenor_months",
            "pd_single",
            "pd_contextual",
            "el_single",
            "el_contextual",
            "el_delta",
            "el_delta_pct",
        ]
    ].sort_values(["client_id", "product_type", "tenor_months"])

    logger.info(
        f"\nComparação EL médio:\n"
        f"  Score único:     R$ {df_feat['el_single'].mean():,.2f}\n"
        f"  Score contextual: R$ {df_feat['el_contextual'].mean():,.2f}\n"
        f"  Delta médio:     R$ {df_feat['el_delta'].mean():,.2f} ({df_feat['el_delta_pct'].mean():.1f}%)"
    )

    return comparison
