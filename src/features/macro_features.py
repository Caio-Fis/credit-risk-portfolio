"""Enriquecimento com séries macroeconômicas do Banco Central do Brasil (SGS API).

Séries utilizadas:
- 432:   Taxa Selic (% a.a.)
- 21082: Inadimplência PJ — total (%)
- 24364: IBC-Br — Índice de Atividade Econômica

Funções principais:
- fetch_bcb_series: baixa uma série temporal da API SGS/BCB
- fetch_all_macro: baixa todas as séries macroeconômicas necessárias
- merge_macro_features: une features macro ao DataFrame principal por data
"""

from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger

from src.config import BCB_IBC_BR, BCB_INADIMPLENCIA_PJ, BCB_SELIC, PROCESSED_DIR

BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
MACRO_CACHE_PATH = PROCESSED_DIR / "macro_features.parquet"

# Período padrão: últimos 10 anos
DEFAULT_START = (datetime.now() - timedelta(days=10 * 365)).strftime("%d/%m/%Y")
DEFAULT_END = datetime.now().strftime("%d/%m/%Y")


def fetch_bcb_series(
    series_id: int,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    timeout: int = 30,
) -> pd.Series:
    """Baixa uma série temporal da API SGS do Banco Central do Brasil.

    Args:
        series_id: Código da série no SGS (ex: 432 para Selic).
        start_date: Data inicial no formato DD/MM/YYYY.
        end_date: Data final no formato DD/MM/YYYY.
        timeout: Timeout HTTP em segundos.

    Returns:
        pd.Series com índice de datas e valores da série.
    """
    url = BCB_SGS_URL.format(series_id=series_id)
    params = {"formato": "json", "dataInicial": start_date, "dataFinal": end_date}

    logger.info(f"Buscando série BCB {series_id} ({start_date} → {end_date})...")
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not data:
        logger.warning(f"Série {series_id} retornou vazia para o período solicitado.")
        return pd.Series(dtype=float, name=f"bcb_{series_id}")

    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.set_index("data").sort_index()

    series = df["valor"].rename(f"bcb_{series_id}")
    logger.success(f"Série {series_id}: {len(series)} observações")
    return series


def fetch_all_macro(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    cache: bool = True,
) -> pd.DataFrame:
    """Baixa e combina todas as séries macroeconômicas.

    Args:
        start_date: Data inicial no formato DD/MM/YYYY.
        end_date: Data final no formato DD/MM/YYYY.
        cache: Se True, usa cache em parquet se disponível.

    Returns:
        DataFrame com colunas: selic_rate, pj_default_rate, ibc_br_yoy
    """
    if cache and MACRO_CACHE_PATH.exists():
        logger.info(f"Carregando macro features do cache: {MACRO_CACHE_PATH}")
        return pd.read_parquet(MACRO_CACHE_PATH)

    series_map = {
        BCB_SELIC: "selic_rate",
        BCB_INADIMPLENCIA_PJ: "pj_default_rate",
        BCB_IBC_BR: "ibc_br",
    }

    frames = {}
    for series_id, col_name in series_map.items():
        try:
            s = fetch_bcb_series(series_id, start_date, end_date)
            frames[col_name] = s
        except Exception as exc:
            logger.warning(
                f"Falha ao buscar série {series_id}: {exc}. Preenchendo com NaN."
            )
            frames[col_name] = pd.Series(dtype=float, name=col_name)

    macro = pd.DataFrame(frames)
    macro.index.name = "date"

    # Garante DatetimeIndex mesmo quando todas as séries falharam
    if not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)

    # Calcula variação anual do IBC-Br
    if "ibc_br" in macro.columns:
        macro["ibc_br_yoy"] = macro["ibc_br"].pct_change(periods=12, fill_method=None) * 100
        macro = macro.drop(columns=["ibc_br"])

    # Resampling mensal apenas se há dados; fallback para série vazia com NaN
    if not macro.empty:
        macro = macro.resample("ME").last().ffill()
    else:
        logger.warning("Nenhuma série macro disponível; features macro serão NaN.")

    if cache:
        MACRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        macro.to_parquet(MACRO_CACHE_PATH)
        logger.success(f"Macro features cacheadas em {MACRO_CACHE_PATH}")

    return macro


def merge_macro_features(
    df: pd.DataFrame,
    date_col: str = "reference_date",
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Une features macroeconômicas ao DataFrame principal por data.

    O Home Credit não tem data absoluta. Se `date_col` não existir,
    as macro features são adicionadas com o valor mais recente disponível
    (snapshot estático — simula decisão de crédito em um ponto fixo).

    Args:
        df: DataFrame principal (feature store).
        date_col: Coluna de data para merge temporal. Se ausente, usa snapshot.
        macro: DataFrame de macro features (se None, chama fetch_all_macro()).

    Returns:
        DataFrame com colunas macro adicionadas.
    """
    if macro is None:
        macro = fetch_all_macro()

    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        macro_reset = macro.reset_index().rename(columns={"date": date_col})
        df = pd.merge_asof(
            df,
            macro_reset.sort_values(date_col),
            on=date_col,
            direction="backward",
        )
        logger.info("Macro features unidas por data (merge_asof).")
    else:
        snapshot = macro.iloc[-1]
        for col, val in snapshot.items():
            df[col] = val
        logger.info(
            f"Macro features adicionadas como snapshot ({macro.index[-1].date()})."
        )

    return df


if __name__ == "__main__":
    macro_df = fetch_all_macro(cache=True)
    logger.info(f"\n{macro_df.tail()}")
