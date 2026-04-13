"""Gerador de dataset sintético com DGP controlado — Módulo 3.

Objetivo: demonstrar que o mesmo CNPJ tem risco radicalmente diferente
dependendo do produto e prazo da operação.

Data Generating Process (DGP) — totalmente documentado e auditável:

PD(cliente, produto, prazo) = sigmoid(
    β_cliente × score_financeiro
    + β_produto × efeito_produto[produto]
    + β_prazo × log(prazo_meses)
    + β_interacao × score_financeiro × log(prazo_meses)
    + ε
)

Efeitos do produto (β_produto):
- capital_de_giro:        0.0  (baseline)
- investimento:          +1.2  (prazo longo → PD estruturalmente maior)
- antecipacao_recebiveis: -0.8 (curto prazo, garantia embutida → PD menor)

Efeito do prazo: +0.4 × log(prazo_meses)
- 1 mês:  log(1) = 0.00 → sem efeito adicional
- 12 meses: log(12) = 2.48 → +0.99 no logit
- 48 meses: log(48) = 3.87 → +1.55 no logit

Interação (cliente × prazo):
- Clientes com score baixo sofrem mais com prazos longos (β_interacao = -0.3)

Funções principais:
- generate_dataset: gera o dataset sintético completo
- dgp_pd: implementa o DGP para cálculo de PD por contrato
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PRODUCTS, TENORS_MONTHS

# ---------------------------------------------------------------------------
# Parâmetros do DGP — documentados e versionados aqui
# ---------------------------------------------------------------------------
DGP_PARAMS = {
    "beta_cliente": -2.0,  # score alto → PD menor
    "beta_prazo": 0.4,  # prazo maior → PD maior
    "beta_interacao": -0.3,  # score baixo penalizado em prazos longos
    "beta_garantia": -0.6,  # garantia real → PD menor
    "product_effects": {
        "capital_de_giro": 0.0,  # baseline
        "investimento": 1.2,  # risco estrutural de longo prazo
        "antecipacao_recebiveis": -0.8,  # garantia embutida (recebíveis)
    },
    "lgd_base": {
        "capital_de_giro": 0.55,
        "investimento": 0.65,  # sem garantia real → LGD maior
        "antecipacao_recebiveis": 0.30,  # recebíveis como colateral
    },
    "noise_std": 0.5,  # variabilidade idiossincrática
}


def dgp_pd(
    score_financeiro: np.ndarray,
    produto: str,
    prazo_meses: int,
    has_collateral: np.ndarray,
    params: dict = DGP_PARAMS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Calcula PD via o DGP controlado.

    Args:
        score_financeiro: Score normalizado [0,1] do cliente (maior = melhor).
        produto: Tipo de produto ('capital_de_giro', 'investimento', 'antecipacao_recebiveis').
        prazo_meses: Prazo em meses.
        has_collateral: Array binário (1 = com garantia real).
        params: Parâmetros do DGP.
        rng: Gerador aleatório numpy.

    Returns:
        Array de PD por contrato em [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    log_prazo = np.log(max(prazo_meses, 1))

    logit_pd = (
        params["beta_cliente"] * score_financeiro
        + params["beta_prazo"] * log_prazo
        + params["beta_interacao"] * score_financeiro * log_prazo
        + params["product_effects"].get(produto, 0.0)
        + params["beta_garantia"] * has_collateral.astype(float)
        + rng.normal(0, params["noise_std"], size=len(score_financeiro))
    )

    pd_values = 1 / (1 + np.exp(-logit_pd))
    return pd_values


def generate_dataset(
    n: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dataset sintético com DGP controlado.

    Cada linha representa um contrato com:
    - Perfil do cliente (score_financeiro, idade, setor)
    - Contexto da operação (produto, prazo, garantia)
    - PD real (gerada pelo DGP) + default observado
    - LGD real (determinístico + ruído)
    - EL = PD × LGD × EAD

    O mesmo cliente aparece com múltiplos produtos/prazos para
    demonstrar que o risco é contextual.

    Args:
        n: Número total de contratos.
        seed: Semente aleatória (reproduzível).

    Returns:
        DataFrame com o dataset sintético completo.
    """
    rng = np.random.default_rng(seed)

    # Número de clientes únicos (cada cliente tem ~3 contratos em contextos diferentes)
    n_clients = n // 3

    # ---------------------------------------------------------------------------
    # Perfil dos clientes (fixo por cliente)
    # ---------------------------------------------------------------------------
    client_ids = np.arange(n_clients)
    score_financeiro = rng.beta(
        a=2, b=3, size=n_clients
    )  # distribuição assimétrica à esquerda
    idade_empresa_anos = rng.integers(1, 30, size=n_clients)
    setor = rng.choice(
        ["comercio", "servicos", "industria", "agronegocio"], size=n_clients
    )
    faturamento_anual = rng.lognormal(mean=12, sigma=1.5, size=n_clients)  # R$

    # ---------------------------------------------------------------------------
    # Contratos: cruza cada cliente com combinações de produto × prazo
    # ---------------------------------------------------------------------------
    records = []
    for client_idx in range(n_clients):
        # Sorteia 3 contextos diferentes para o mesmo cliente
        contexts = rng.choice(len(PRODUCTS) * len(TENORS_MONTHS), size=3, replace=False)
        for ctx_idx in contexts:
            produto = PRODUCTS[ctx_idx % len(PRODUCTS)]
            prazo = TENORS_MONTHS[ctx_idx // len(PRODUCTS)]

            ead = faturamento_anual[client_idx] * rng.uniform(0.05, 0.40)
            has_collateral = rng.random() < (0.6 if produto == "investimento" else 0.2)

            pd_true = dgp_pd(
                score_financeiro=np.array([score_financeiro[client_idx]]),
                produto=produto,
                prazo_meses=prazo,
                has_collateral=np.array([has_collateral]),
                rng=rng,
            )[0]

            default = int(rng.random() < pd_true)

            lgd_base = DGP_PARAMS["lgd_base"][produto]
            lgd_true = float(np.clip(lgd_base + rng.normal(0, 0.10), 0.01, 0.99))

            el = pd_true * lgd_true * ead

            records.append(
                {
                    "client_id": client_ids[client_idx],
                    "score_financeiro": round(score_financeiro[client_idx], 4),
                    "idade_empresa_anos": int(idade_empresa_anos[client_idx]),
                    "setor": setor[client_idx],
                    "faturamento_anual": round(faturamento_anual[client_idx], 2),
                    "product_type": produto,
                    "tenor_months": prazo,
                    "has_collateral": int(has_collateral),
                    "ead": round(ead, 2),
                    "pd_true": round(pd_true, 6),
                    "lgd_true": round(lgd_true, 4),
                    "el_true": round(el, 2),
                    "default": default,
                }
            )

    df = pd.DataFrame(records).reset_index(drop=True)

    logger.success(
        f"Dataset sintético gerado: {len(df):,} contratos × {df.shape[1]} colunas | "
        f"Taxa de default: {df['default'].mean():.2%} | "
        f"PD média: {df['pd_true'].mean():.3f}"
    )
    return df
