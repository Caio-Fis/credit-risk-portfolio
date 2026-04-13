"""Gatilhos comportamentais para early warning.

Complementa a trajetória de score com sinais operacionais que indicam
deterioração antes que o score capture:

- Queda de volume transacional: redução de receita/NF-e emitidas
- Protestos: registro de protesto em cartório
- Atrasos em outras obrigações: DPD crescente em outras operações

No dataset Home Credit, esses sinais são aproximados por:
- Queda no número de parcelas pagas no mês (installments)
- Aumento de DPD em bureau_balance
- Aumento de contratos com status "Bad debt" no bureau

Funções principais:
- flag_volume_drop: queda de volume transacional
- flag_protests: protestos / bad debt no bureau
- aggregate_signals: agrega todos os sinais em score de risco composto
"""

import numpy as np
import pandas as pd
from loguru import logger

# Thresholds de volume (configuráveis)
VOLUME_DROP_PCT = 0.30  # queda de 30% no volume → alerta
PROTEST_FLAG_THRESHOLD = 1  # qualquer protesto → alerta


def flag_volume_drop(
    df: pd.DataFrame,
    entity_col: str = "SK_ID_CURR",
    volume_col: str = "monthly_payment_count",
    date_col: str = "reference_date",
    window_months: int = 3,
    threshold_pct: float = VOLUME_DROP_PCT,
) -> pd.DataFrame:
    """Detecta queda brusca no volume de transações/pagamentos.

    Compara o volume médio dos últimos `window_months` com o período anterior.

    Args:
        df: DataFrame com histórico de volumes por entidade e mês.
        entity_col: Coluna de identificador da entidade.
        volume_col: Coluna de volume mensal (ex: pagamentos realizados).
        date_col: Coluna de data de referência.
        window_months: Janela de comparação em meses.
        threshold_pct: Percentual de queda que dispara o alerta.

    Returns:
        DataFrame com entidades em alerta de queda de volume.
    """
    for col in (entity_col, volume_col, date_col):
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([entity_col, date_col])

    alerts = []
    cutoff = df[date_col].max() - pd.DateOffset(months=window_months)

    for entity_id, group in df.groupby(entity_col):
        recent = group[group[date_col] >= cutoff][volume_col].mean()
        historic = group[group[date_col] < cutoff][volume_col].mean()

        if pd.isna(recent) or pd.isna(historic) or historic == 0:
            continue

        drop_pct = (historic - recent) / historic
        if drop_pct >= threshold_pct:
            alerts.append(
                {
                    entity_col: entity_id,
                    "volume_recent": recent,
                    "volume_hist": historic,
                    "volume_drop_pct": drop_pct,
                    "signal_volume": True,
                }
            )

    result = pd.DataFrame(alerts)
    logger.info(
        f"Volume drop: {len(result)} entidades em alerta (>{threshold_pct:.0%})"
    )
    return result


def flag_protests(
    bureau_df: pd.DataFrame,
    entity_col: str = "SK_ID_CURR",
    status_col: str = "CREDIT_ACTIVE",
    bad_status_values: list[str] | None = None,
) -> pd.DataFrame:
    """Detecta contratos com status de protesto ou bad debt no bureau.

    No Home Credit, usa CREDIT_ACTIVE == 'Bad debt' como proxy de protesto.

    Args:
        bureau_df: DataFrame do bureau com status de créditos ativos.
        entity_col: Coluna de identificador.
        status_col: Coluna de status do crédito no bureau.
        bad_status_values: Valores que indicam protesto/bad debt.

    Returns:
        DataFrame com entidades com protesto, ordenado por contagem.
    """
    if bad_status_values is None:
        bad_status_values = ["Bad debt", "Sold"]

    if status_col not in bureau_df.columns:
        logger.warning(f"Coluna '{status_col}' não encontrada. Sem sinal de protesto.")
        return pd.DataFrame(columns=[entity_col, "protest_count", "signal_protest"])

    bad_df = bureau_df[bureau_df[status_col].isin(bad_status_values)]

    result = (
        bad_df.groupby(entity_col)
        .size()
        .reset_index(name="protest_count")
        .assign(signal_protest=True)
    )

    logger.info(f"Protestos: {len(result)} entidades com bad debt/sold no bureau")
    return result


def aggregate_signals(
    trajectory_alerts: pd.DataFrame,
    volume_alerts: pd.DataFrame | None = None,
    protest_alerts: pd.DataFrame | None = None,
    entity_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Agrega todos os sinais em um score de risco composto por entidade.

    Score composto (0–3):
    - +1 se queda de score > threshold
    - +1 se queda de volume > threshold
    - +1 se protesto/bad debt no bureau

    Args:
        trajectory_alerts: Resultado de flag_score_drop().
        volume_alerts: Resultado de flag_volume_drop() (opcional).
        protest_alerts: Resultado de flag_protests() (opcional).
        entity_col: Coluna de identificador.

    Returns:
        DataFrame com entidades em alerta, score composto e sinais ativos.
    """
    # Base: entidades com queda de score
    base = trajectory_alerts[[entity_col, "score_drop", "alert_level"]].copy()
    base["signal_score"] = True
    base["composite_risk"] = 1

    if (
        volume_alerts is not None
        and not volume_alerts.empty
        and entity_col in volume_alerts.columns
    ):
        vol_subset = volume_alerts[[entity_col, "volume_drop_pct", "signal_volume"]]
        base = base.merge(vol_subset, on=entity_col, how="left")
        base["signal_volume"] = base["signal_volume"].fillna(False)
        base["composite_risk"] += base["signal_volume"].astype(int)
    else:
        base["signal_volume"] = False

    if (
        protest_alerts is not None
        and not protest_alerts.empty
        and entity_col in protest_alerts.columns
    ):
        prot_subset = protest_alerts[[entity_col, "protest_count", "signal_protest"]]
        base = base.merge(prot_subset, on=entity_col, how="left")
        base["signal_protest"] = base["signal_protest"].fillna(False)
        base["composite_risk"] += base["signal_protest"].astype(int)
    else:
        base["signal_protest"] = False

    base["risk_label"] = (
        base["composite_risk"]
        .map(
            {
                1: "moderado",
                2: "alto",
                3: "crítico",
            }
        )
        .fillna("moderado")
    )

    result = base.sort_values(
        ["composite_risk", "score_drop"], ascending=False
    ).reset_index(drop=True)

    logger.warning(
        f"Alerta composto: {len(result)} entidades | "
        f"crítico={(result['composite_risk'] == 3).sum()} | "
        f"alto={(result['composite_risk'] == 2).sum()} | "
        f"moderado={(result['composite_risk'] == 1).sum()}"
    )
    return result


def simulate_behavioral_data(
    n_entities: int = 500,
    n_months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dados sintéticos de comportamento para demonstração.

    Args:
        n_entities: Número de entidades.
        n_months: Número de meses de histórico.
        seed: Semente aleatória.

    Returns:
        DataFrame com entity_id, reference_date, pd_score, monthly_payment_count.
    """
    rng = np.random.default_rng(seed)
    records = []
    base_date = pd.Timestamp("2023-01-01")

    for i in range(n_entities):
        # Score inicial (500–900)
        score_init = rng.integers(500, 900)
        # Trend: 80% estável, 15% queda, 5% melhora
        trend_type = rng.choice(["stable", "drop", "improve"], p=[0.80, 0.15, 0.05])

        # Volume de pagamentos mensal
        vol_base = rng.integers(3, 15)

        for m in range(n_months):
            date = base_date + pd.DateOffset(months=m)

            if trend_type == "drop" and m >= n_months // 2:
                noise = rng.normal(0, 15)
                score = max(100, score_init - (m - n_months // 2) * 20 + noise)
                volume = max(
                    0, vol_base - (m - n_months // 2) * 2 + rng.integers(-1, 2)
                )
            elif trend_type == "improve":
                noise = rng.normal(0, 10)
                score = min(999, score_init + m * 5 + noise)
                volume = min(20, vol_base + m // 3 + rng.integers(0, 2))
            else:
                noise = rng.normal(0, 20)
                score = float(np.clip(score_init + noise, 100, 999))
                volume = int(np.clip(vol_base + rng.integers(-2, 3), 0, 20))

            records.append(
                {
                    "SK_ID_CURR": i,
                    "reference_date": date,
                    "pd_score": round(score, 1),
                    "monthly_payment_count": volume,
                }
            )

    return pd.DataFrame(records)
