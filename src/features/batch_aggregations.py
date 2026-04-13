"""Agregações em batch sobre tabelas grandes usando pandas chunked read.

Substitui as funções pandas que carregavam 27M + 13.6M linhas inteiras na RAM.
A estratégia é processar os CSVs em lotes de 1M linhas, acumular resultados
parciais e combinar no final — o pico de memória cai de ~4 GB para ~200 MB.

Funções principais:
- build_bureau_features_batch: agrega bureau_balance (27M linhas) por janela
- build_installment_features_batch: agrega installments (13.6M linhas) por janela
- build_pos_cash_features_batch: agrega POS_CASH_balance (10M linhas) por janela
- build_credit_card_features_batch: agrega credit_card_balance (3.8M linhas) por janela
"""

from pathlib import Path

import pandas as pd
from loguru import logger

_OVERDUE_STATUS = {"1", "2", "3", "4", "5"}
_DEFAULT_CHUNKSIZE = 1_000_000


def build_bureau_features_batch(
    bureau_path: Path,
    bureau_bal_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Agrega features de bureau por janela temporal em modo batch.

    A bureau.csv (1.7M linhas) é carregada inteira — é pequena o suficiente.
    A bureau_balance.csv (27.3M linhas) é lida em chunks de `chunksize` linhas.

    Estratégia por métrica:
    - cnt_credits (nunique SK_ID_BUREAU): acumula set de bureau_ids vistos por janela;
      a contagem final é o tamanho do set, sem duplicações entre chunks.
    - cnt_overdue (sum is_overdue): soma parcial por chunk é aditiva; basta somar.

    Args:
        bureau_path: Caminho para bureau.csv.
        bureau_bal_path: Caminho para bureau_balance.csv.
        windows: Janelas em dias, ex: [30, 90, 365].
        chunksize: Linhas por chunk ao ler bureau_balance.

    Returns:
        DataFrame pandas com SK_ID_CURR + features por janela.
    """
    if windows is None:
        windows = [30, 90, 365]

    # bureau.csv cabe na RAM: 1.7M linhas × 2 colunas ~ 14 MB
    logger.info("Batch bureau: carregando bureau.csv (1.7M linhas)...")
    bureau = pd.read_csv(bureau_path, usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
    id_map: dict[int, int] = bureau.set_index("SK_ID_BUREAU")["SK_ID_CURR"].to_dict()
    del bureau

    # Acumuladores cross-chunk por janela
    seen_bureaus: dict[int, set] = {w: set() for w in windows}  # → cnt_credits
    overdue_chunks: dict[int, list] = {w: [] for w in windows}  # → cnt_overdue

    n_chunks = 0
    logger.info(
        f"Batch bureau: lendo bureau_balance.csv em chunks de {chunksize:,} linhas..."
    )
    for chunk in pd.read_csv(
        bureau_bal_path,
        chunksize=chunksize,
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    ):
        chunk["SK_ID_CURR"] = chunk["SK_ID_BUREAU"].map(id_map)
        chunk = chunk.dropna(subset=["SK_ID_CURR"])
        chunk["SK_ID_CURR"] = chunk["SK_ID_CURR"].astype(int)
        chunk["is_overdue"] = chunk["STATUS"].isin(_OVERDUE_STATUS).astype(int)

        for w in windows:
            cutoff = -(w // 30)
            sub = chunk[chunk["MONTHS_BALANCE"] >= cutoff]
            if sub.empty:
                continue

            # cnt_credits: rastreia bureaus únicos (não aditivo entre chunks)
            seen_bureaus[w].update(sub["SK_ID_BUREAU"].unique())

            # cnt_overdue: soma parcial (aditivo entre chunks)
            overdue_chunk = sub.groupby("SK_ID_CURR")["is_overdue"].sum()
            overdue_chunks[w].append(overdue_chunk)

        n_chunks += 1

    logger.info(f"Batch bureau: processados {n_chunks} chunks. Combinando resultados...")

    result = None
    for w in windows:
        # cnt_credits: mapeia bureaus vistos → clientes → conta por cliente
        seen_df = pd.DataFrame({"SK_ID_BUREAU": list(seen_bureaus[w])})
        seen_df["SK_ID_CURR"] = seen_df["SK_ID_BUREAU"].map(id_map)
        cnt_credits = (
            seen_df.dropna(subset=["SK_ID_CURR"])
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name=f"bureau_cnt_credits_{w}d")
        )

        # cnt_overdue: soma total entre chunks
        cnt_overdue = (
            pd.concat(overdue_chunks[w])
            .groupby(level=0)
            .sum()
            .reset_index()
            .rename(columns={"is_overdue": f"bureau_cnt_overdue_{w}d"})
        )

        df_w = cnt_credits.merge(cnt_overdue, on="SK_ID_CURR", how="outer")
        result = df_w if result is None else result.merge(df_w, on="SK_ID_CURR", how="outer")

    assert result is not None
    df = result.fillna(0)
    logger.info(f"Bureau features: {df.shape[0]:,} clientes × {df.shape[1]} colunas")
    return df


def build_installment_features_batch(
    inst_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Calcula DPD por janela temporal em modo batch.

    installments_payments.csv (13.6M linhas) é lida em chunks de `chunksize`.

    Estratégia por métrica:
    - dpd_mean: acumula soma e contagem separadamente; combina no final.
    - dpd_max: max parcial é combinável via max.
    - dpd_cnt_positive: contagem parcial é aditiva.

    Args:
        inst_path: Caminho para installments_payments.csv.
        windows: Janelas em dias, ex: [30, 90, 365].
        chunksize: Linhas por chunk ao ler installments_payments.

    Returns:
        DataFrame pandas com SK_ID_CURR + features por janela.
    """
    if windows is None:
        windows = [30, 90, 365]

    # Acumuladores por janela — listas de Series agregadas por chunk
    sum_chunks: dict[int, list] = {w: [] for w in windows}
    count_chunks: dict[int, list] = {w: [] for w in windows}
    max_chunks: dict[int, list] = {w: [] for w in windows}
    pos_chunks: dict[int, list] = {w: [] for w in windows}

    n_chunks = 0
    logger.info(
        f"Batch installments: lendo installments_payments.csv em chunks de {chunksize:,} linhas..."
    )
    for chunk in pd.read_csv(
        inst_path,
        chunksize=chunksize,
        usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"],
    ):
        chunk["DPD"] = (
            chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        ).clip(lower=0)

        for w in windows:
            sub = chunk[chunk["DAYS_INSTALMENT"] >= -w]
            if sub.empty:
                continue

            g = sub.groupby("SK_ID_CURR")["DPD"]
            sum_chunks[w].append(g.sum().rename("dpd_sum"))
            count_chunks[w].append(g.count().rename("dpd_count"))
            max_chunks[w].append(g.max().rename("dpd_max"))
            # dpd_cnt_positive: filtra antes de contar para evitar lambda
            pos_sub = sub[sub["DPD"] > 0]
            if not pos_sub.empty:
                pos_chunks[w].append(
                    pos_sub.groupby("SK_ID_CURR")["DPD"].count().rename("dpd_pos")
                )

        n_chunks += 1

    logger.info(
        f"Batch installments: processados {n_chunks} chunks. Combinando resultados..."
    )

    result = None
    for w in windows:
        dpd_sum = pd.concat(sum_chunks[w]).groupby(level=0).sum()
        dpd_count = pd.concat(count_chunks[w]).groupby(level=0).sum()
        dpd_max = pd.concat(max_chunks[w]).groupby(level=0).max()
        dpd_pos = (
            pd.concat(pos_chunks[w]).groupby(level=0).sum()
            if pos_chunks[w]
            else pd.Series(dtype=float, name="dpd_pos")
        )

        # Alinha índices para operações vetorizadas
        common_idx = dpd_sum.index
        dpd_pos = dpd_pos.reindex(common_idx, fill_value=0)

        df_w = pd.DataFrame(
            {
                "SK_ID_CURR": common_idx,
                f"installments_dpd_mean_{w}d": (dpd_sum / dpd_count).values,
                f"installments_dpd_max_{w}d": dpd_max.values,
                f"installments_dpd_cnt_positive_{w}d": dpd_pos.values,
            }
        )
        result = df_w if result is None else result.merge(df_w, on="SK_ID_CURR", how="outer")

    assert result is not None
    df = result.fillna(0)
    logger.info(f"Installment features: {df.shape[0]:,} clientes × {df.shape[1]} colunas")
    return df


def build_pos_cash_features_batch(
    pos_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Agrega features de POS_CASH_balance por janela temporal em modo batch.

    POS_CASH_balance.csv (10M linhas): snapshots mensais de empréstimos
    parcelados e de caixa. Janelas em meses: 30d≈1m, 90d≈3m, 365d≈12m.

    Features extraídas por janela:
    - pos_dpd_mean: média de SK_DPD por cliente
    - pos_dpd_max: máximo de SK_DPD por cliente
    - pos_cnt_dpd_positive: meses com SK_DPD > 0

    Args:
        pos_path: Caminho para POS_CASH_balance.csv.
        windows: Janelas em dias; convertido para meses internamente.
        chunksize: Linhas por chunk.

    Returns:
        DataFrame com SK_ID_CURR + features por janela.
    """
    if windows is None:
        windows = [30, 90, 365]

    # MONTHS_BALANCE é negativo (ex: -1 = mês passado); janela_meses = -(days//30)
    window_months = {w: -(w // 30) for w in windows}

    sum_chunks: dict[int, list] = {w: [] for w in windows}
    count_chunks: dict[int, list] = {w: [] for w in windows}
    max_chunks: dict[int, list] = {w: [] for w in windows}
    pos_chunks: dict[int, list] = {w: [] for w in windows}

    n_chunks = 0
    logger.info(
        f"Batch POS_CASH: lendo POS_CASH_balance.csv em chunks de {chunksize:,} linhas..."
    )
    for chunk in pd.read_csv(
        pos_path,
        chunksize=chunksize,
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"],
    ):
        for w in windows:
            sub = chunk[chunk["MONTHS_BALANCE"] >= window_months[w]]
            if sub.empty:
                continue

            g = sub.groupby("SK_ID_CURR")["SK_DPD"]
            sum_chunks[w].append(g.sum().rename("dpd_sum"))
            count_chunks[w].append(g.count().rename("dpd_count"))
            max_chunks[w].append(g.max().rename("dpd_max"))
            pos_sub = sub[sub["SK_DPD"] > 0]
            if not pos_sub.empty:
                pos_chunks[w].append(
                    pos_sub.groupby("SK_ID_CURR")["SK_DPD"].count().rename("dpd_pos")
                )

        n_chunks += 1

    logger.info(f"Batch POS_CASH: processados {n_chunks} chunks. Combinando...")

    result = None
    for w in windows:
        if not sum_chunks[w]:
            continue
        dpd_sum = pd.concat(sum_chunks[w]).groupby(level=0).sum()
        dpd_count = pd.concat(count_chunks[w]).groupby(level=0).sum()
        dpd_max = pd.concat(max_chunks[w]).groupby(level=0).max()
        dpd_pos = (
            pd.concat(pos_chunks[w]).groupby(level=0).sum()
            if pos_chunks[w]
            else pd.Series(dtype=float, name="dpd_pos")
        )
        dpd_pos = dpd_pos.reindex(dpd_sum.index, fill_value=0)

        df_w = pd.DataFrame(
            {
                "SK_ID_CURR": dpd_sum.index,
                f"pos_dpd_mean_{w}d": (dpd_sum / dpd_count).values,
                f"pos_dpd_max_{w}d": dpd_max.values,
                f"pos_cnt_dpd_positive_{w}d": dpd_pos.values,
            }
        )
        result = df_w if result is None else result.merge(df_w, on="SK_ID_CURR", how="outer")

    if result is None:
        result = pd.DataFrame(columns=["SK_ID_CURR"])
    df = result.fillna(0)
    logger.info(f"POS_CASH features: {df.shape[0]:,} clientes × {df.shape[1]} colunas")
    return df


def build_credit_card_features_batch(
    cc_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Agrega features de credit_card_balance por janela temporal em modo batch.

    credit_card_balance.csv (3.8M linhas): snapshots mensais de cartão de crédito.

    Features extraídas por janela:
    - cc_utilization_mean: utilização média (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL)
    - cc_dpd_max: máximo de SK_DPD
    - cc_cnt_dpd_positive: meses com SK_DPD > 0

    Args:
        cc_path: Caminho para credit_card_balance.csv.
        windows: Janelas em dias; convertido para meses internamente.
        chunksize: Linhas por chunk.

    Returns:
        DataFrame com SK_ID_CURR + features por janela.
    """
    if windows is None:
        windows = [30, 90, 365]

    window_months = {w: -(w // 30) for w in windows}

    util_sum_chunks: dict[int, list] = {w: [] for w in windows}
    util_count_chunks: dict[int, list] = {w: [] for w in windows}
    max_chunks: dict[int, list] = {w: [] for w in windows}
    pos_chunks: dict[int, list] = {w: [] for w in windows}

    n_chunks = 0
    logger.info(
        f"Batch credit_card: lendo credit_card_balance.csv em chunks de {chunksize:,} linhas..."
    )
    for chunk in pd.read_csv(
        cc_path,
        chunksize=chunksize,
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                 "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"],
    ):
        # Utilização: clipa limite em 1 para evitar divisão por zero
        chunk["utilization"] = chunk["AMT_BALANCE"] / chunk["AMT_CREDIT_LIMIT_ACTUAL"].clip(lower=1)
        chunk["utilization"] = chunk["utilization"].clip(0, 1)

        for w in windows:
            sub = chunk[chunk["MONTHS_BALANCE"] >= window_months[w]]
            if sub.empty:
                continue

            g_util = sub.groupby("SK_ID_CURR")["utilization"]
            util_sum_chunks[w].append(g_util.sum().rename("util_sum"))
            util_count_chunks[w].append(g_util.count().rename("util_count"))

            g_dpd = sub.groupby("SK_ID_CURR")["SK_DPD"]
            max_chunks[w].append(g_dpd.max().rename("dpd_max"))
            pos_sub = sub[sub["SK_DPD"] > 0]
            if not pos_sub.empty:
                pos_chunks[w].append(
                    pos_sub.groupby("SK_ID_CURR")["SK_DPD"].count().rename("dpd_pos")
                )

        n_chunks += 1

    logger.info(f"Batch credit_card: processados {n_chunks} chunks. Combinando...")

    result = None
    for w in windows:
        if not util_sum_chunks[w]:
            continue
        util_sum = pd.concat(util_sum_chunks[w]).groupby(level=0).sum()
        util_count = pd.concat(util_count_chunks[w]).groupby(level=0).sum()
        dpd_max = pd.concat(max_chunks[w]).groupby(level=0).max()
        dpd_pos = (
            pd.concat(pos_chunks[w]).groupby(level=0).sum()
            if pos_chunks[w]
            else pd.Series(dtype=float, name="dpd_pos")
        )
        dpd_pos = dpd_pos.reindex(util_sum.index, fill_value=0)

        df_w = pd.DataFrame(
            {
                "SK_ID_CURR": util_sum.index,
                f"cc_utilization_mean_{w}d": (util_sum / util_count).values,
                f"cc_dpd_max_{w}d": dpd_max.reindex(util_sum.index, fill_value=0).values,
                f"cc_cnt_dpd_positive_{w}d": dpd_pos.values,
            }
        )
        result = df_w if result is None else result.merge(df_w, on="SK_ID_CURR", how="outer")

    if result is None:
        result = pd.DataFrame(columns=["SK_ID_CURR"])
    df = result.fillna(0)
    logger.info(f"Credit card features: {df.shape[0]:,} clientes × {df.shape[1]} colunas")
    return df
