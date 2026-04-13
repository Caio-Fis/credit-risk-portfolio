"""Batch aggregations over large tables using pandas chunked read.

Replaces pandas functions that loaded 27M + 13.6M rows entirely into RAM.
The strategy is to process CSVs in batches of 1M rows, accumulate partial
results, and combine at the end — peak memory drops from ~4 GB to ~200 MB.

Main functions:
- build_bureau_features_batch: aggregates bureau_balance (27M rows) by window
- build_installment_features_batch: aggregates installments (13.6M rows) by window
- build_pos_cash_features_batch: aggregates POS_CASH_balance (10M rows) by window
- build_credit_card_features_batch: aggregates credit_card_balance (3.8M rows) by window
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
    """Aggregates bureau features by time window in batch mode.

    bureau.csv (1.7M rows) is loaded entirely — it is small enough.
    bureau_balance.csv (27.3M rows) is read in chunks of `chunksize` rows.

    Strategy per metric:
    - cnt_credits (nunique SK_ID_BUREAU): accumulates set of bureau_ids seen per window;
      the final count is the set size, without duplicates across chunks.
    - cnt_overdue (sum is_overdue): partial sum per chunk is additive; just sum.

    Args:
        bureau_path: Path to bureau.csv.
        bureau_bal_path: Path to bureau_balance.csv.
        windows: Windows in days, e.g.: [30, 90, 365].
        chunksize: Rows per chunk when reading bureau_balance.

    Returns:
        Pandas DataFrame with SK_ID_CURR + features per window.
    """
    if windows is None:
        windows = [30, 90, 365]

    # bureau.csv fits in RAM: 1.7M rows × 2 columns ~ 14 MB
    logger.info("Batch bureau: loading bureau.csv (1.7M rows)...")
    bureau = pd.read_csv(bureau_path, usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
    id_map: dict[int, int] = bureau.set_index("SK_ID_BUREAU")["SK_ID_CURR"].to_dict()
    del bureau

    # Cross-chunk accumulators per window
    seen_bureaus: dict[int, set] = {w: set() for w in windows}  # → cnt_credits
    overdue_chunks: dict[int, list] = {w: [] for w in windows}  # → cnt_overdue

    n_chunks = 0
    logger.info(
        f"Batch bureau: reading bureau_balance.csv in chunks of {chunksize:,} rows..."
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

            # cnt_credits: tracks unique bureaus (not additive across chunks)
            seen_bureaus[w].update(sub["SK_ID_BUREAU"].unique())

            # cnt_overdue: partial sum (additive across chunks)
            overdue_chunk = sub.groupby("SK_ID_CURR")["is_overdue"].sum()
            overdue_chunks[w].append(overdue_chunk)

        n_chunks += 1

    logger.info(f"Batch bureau: processed {n_chunks} chunks. Combining results...")

    result = None
    for w in windows:
        # cnt_credits: maps seen bureaus → clients → count per client
        seen_df = pd.DataFrame({"SK_ID_BUREAU": list(seen_bureaus[w])})
        seen_df["SK_ID_CURR"] = seen_df["SK_ID_BUREAU"].map(id_map)
        cnt_credits = (
            seen_df.dropna(subset=["SK_ID_CURR"])
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name=f"bureau_cnt_credits_{w}d")
        )

        # cnt_overdue: total sum across chunks
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
    logger.info(f"Bureau features: {df.shape[0]:,} clients × {df.shape[1]} columns")
    return df


def build_installment_features_batch(
    inst_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Calculates DPD by time window in batch mode.

    installments_payments.csv (13.6M rows) is read in chunks of `chunksize`.

    Strategy per metric:
    - dpd_mean: accumulates sum and count separately; combines at the end.
    - dpd_max: partial max is combinable via max.
    - dpd_cnt_positive: partial count is additive.

    Args:
        inst_path: Path to installments_payments.csv.
        windows: Windows in days, e.g.: [30, 90, 365].
        chunksize: Rows per chunk when reading installments_payments.

    Returns:
        Pandas DataFrame with SK_ID_CURR + features per window.
    """
    if windows is None:
        windows = [30, 90, 365]

    # Accumulators per window — lists of per-chunk aggregated Series
    sum_chunks: dict[int, list] = {w: [] for w in windows}
    count_chunks: dict[int, list] = {w: [] for w in windows}
    max_chunks: dict[int, list] = {w: [] for w in windows}
    pos_chunks: dict[int, list] = {w: [] for w in windows}

    n_chunks = 0
    logger.info(
        f"Batch installments: reading installments_payments.csv in chunks of {chunksize:,} rows..."
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
            # dpd_cnt_positive: filter before counting to avoid lambda
            pos_sub = sub[sub["DPD"] > 0]
            if not pos_sub.empty:
                pos_chunks[w].append(
                    pos_sub.groupby("SK_ID_CURR")["DPD"].count().rename("dpd_pos")
                )

        n_chunks += 1

    logger.info(
        f"Batch installments: processed {n_chunks} chunks. Combining results..."
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

        # Align indices for vectorised operations
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
    logger.info(f"Installment features: {df.shape[0]:,} clients × {df.shape[1]} columns")
    return df


def build_pos_cash_features_batch(
    pos_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Aggregates POS_CASH_balance features by time window in batch mode.

    POS_CASH_balance.csv (10M rows): monthly snapshots of instalment
    and cash loans. Windows in months: 30d≈1m, 90d≈3m, 365d≈12m.

    Features extracted per window:
    - pos_dpd_mean: mean SK_DPD per client
    - pos_dpd_max: maximum SK_DPD per client
    - pos_cnt_dpd_positive: months with SK_DPD > 0

    Args:
        pos_path: Path to POS_CASH_balance.csv.
        windows: Windows in days; converted to months internally.
        chunksize: Rows per chunk.

    Returns:
        DataFrame with SK_ID_CURR + features per window.
    """
    if windows is None:
        windows = [30, 90, 365]

    # MONTHS_BALANCE is negative (e.g.: -1 = last month); window_months = -(days//30)
    window_months = {w: -(w // 30) for w in windows}

    sum_chunks: dict[int, list] = {w: [] for w in windows}
    count_chunks: dict[int, list] = {w: [] for w in windows}
    max_chunks: dict[int, list] = {w: [] for w in windows}
    pos_chunks: dict[int, list] = {w: [] for w in windows}

    n_chunks = 0
    logger.info(
        f"Batch POS_CASH: reading POS_CASH_balance.csv in chunks of {chunksize:,} rows..."
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

    logger.info(f"Batch POS_CASH: processed {n_chunks} chunks. Combining...")

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
    logger.info(f"POS_CASH features: {df.shape[0]:,} clients × {df.shape[1]} columns")
    return df


def build_credit_card_features_batch(
    cc_path: Path,
    windows: list[int] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Aggregates credit_card_balance features by time window in batch mode.

    credit_card_balance.csv (3.8M rows): monthly credit card snapshots.

    Features extracted per window:
    - cc_utilization_mean: mean utilisation (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL)
    - cc_dpd_max: maximum SK_DPD
    - cc_cnt_dpd_positive: months with SK_DPD > 0

    Args:
        cc_path: Path to credit_card_balance.csv.
        windows: Windows in days; converted to months internally.
        chunksize: Rows per chunk.

    Returns:
        DataFrame with SK_ID_CURR + features per window.
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
        f"Batch credit_card: reading credit_card_balance.csv in chunks of {chunksize:,} rows..."
    )
    for chunk in pd.read_csv(
        cc_path,
        chunksize=chunksize,
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                 "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"],
    ):
        # Utilisation: clip limit at 1 to avoid division by zero
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

    logger.info(f"Batch credit_card: processed {n_chunks} chunks. Combining...")

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
    logger.info(f"Credit card features: {df.shape[0]:,} clients × {df.shape[1]} columns")
    return df
