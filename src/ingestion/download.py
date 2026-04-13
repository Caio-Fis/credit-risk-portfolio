"""Ingestão do dataset Home Credit Default Risk via Kaggle API.

Funções principais:
- download_home_credit: baixa e extrai os arquivos do Kaggle
- validate_schema: verifica presença de colunas obrigatórias
- partition_by_date: particiona processed/ por ano de referência
"""

import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import PROCESSED_DIR, RAW_DIR

KAGGLE_COMPETITION = "home-credit-default-risk"

# Arquivos esperados após o download
EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "POS_CASH_balance.csv",
]

# Colunas obrigatórias mínimas por arquivo
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "application_train.csv": ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL"],
    "bureau.csv": ["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT"],
    "bureau_balance.csv": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    "previous_application.csv": ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"],
    "installments_payments.csv": [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
    ],
}


def download_home_credit(output_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Baixa o dataset Home Credit via Kaggle API.

    Requer ~/.kaggle/kaggle.json com credenciais válidas.

    Args:
        output_dir: Diretório de destino dos arquivos brutos.
        force: Se True, baixa novamente mesmo que os arquivos existam.

    Returns:
        Path do diretório com os arquivos extraídos.
    """
    target_dir = output_dir / "home_credit"

    if not force and target_dir.exists() and any(target_dir.glob("*.csv")):
        logger.info(f"Dados já existem em {target_dir}. Use force=True para rebaixar.")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError("Instale kaggle: uv add kaggle")

    logger.info(f"Baixando competição '{KAGGLE_COMPETITION}' para {target_dir}...")
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(target_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download falhou:\n{result.stderr}")

    logger.info("Extraindo arquivos ZIP...")
    for zip_path in target_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        zip_path.unlink()
        logger.debug(f"Extraído: {zip_path.name}")

    logger.success(f"Download concluído. Arquivos em {target_dir}")
    return target_dir


def validate_schema(data_dir: Path = RAW_DIR / "home_credit") -> bool:
    """Verifica que os arquivos esperados existem e contêm as colunas mínimas.

    Args:
        data_dir: Diretório com os CSVs do Home Credit.

    Returns:
        True se tudo válido, levanta ValueError caso contrário.
    """
    errors: list[str] = []

    for filename in EXPECTED_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            errors.append(f"Arquivo ausente: {filename}")
            continue

        if filename in REQUIRED_COLUMNS:
            df_head = pd.read_csv(filepath, nrows=0)
            missing = set(REQUIRED_COLUMNS[filename]) - set(df_head.columns)
            if missing:
                errors.append(f"{filename}: colunas ausentes {missing}")

    if errors:
        for err in errors:
            logger.error(err)
        raise ValueError(f"Validação falhou com {len(errors)} erro(s). Veja logs.")

    logger.success(f"Schema validado — {len(EXPECTED_FILES)} arquivo(s) OK.")
    return True


def load_application_train(data_dir: Path = RAW_DIR / "home_credit") -> pd.DataFrame:
    """Carrega application_train.csv com tipos corretos.

    Args:
        data_dir: Diretório com os CSVs do Home Credit.

    Returns:
        DataFrame com colunas tipadas.
    """
    path = data_dir / "application_train.csv"
    logger.info(f"Carregando {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"Carregado: {df.shape[0]:,} contratos × {df.shape[1]} colunas")
    return df


def partition_by_date(
    df: pd.DataFrame,
    date_col: str = "DAYS_BIRTH",
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Salva o DataFrame particionado em parquet por ano aproximado.

    Home Credit não tem data absoluta — usa DAYS_BIRTH como proxy de ordenação
    para simular particionamento temporal.

    Args:
        df: DataFrame a particionar.
        date_col: Coluna de referência temporal.
        output_dir: Diretório raiz de saída.
    """
    if date_col not in df.columns:
        logger.warning(f"Coluna {date_col} não encontrada. Salvando sem partição.")
        out = output_dir / "application_train.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Salvo em {out}")
        return

    # Converte DAYS_BIRTH em anos negativos → bins de 10 anos
    df = df.copy()
    df["_year_bin"] = pd.cut(df[date_col], bins=5, labels=False)

    for bin_id, group in df.groupby("_year_bin"):
        out_path = output_dir / f"application_train_bin{int(bin_id)}.parquet"
        group.drop(columns=["_year_bin"]).to_parquet(out_path, index=False)
        logger.debug(f"Partição {bin_id}: {len(group):,} registros → {out_path.name}")

    logger.success(f"Particionamento concluído. Arquivos em {output_dir}")


if __name__ == "__main__":
    data_dir = download_home_credit()
    validate_schema(data_dir)
