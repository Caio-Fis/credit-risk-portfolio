"""Home Credit Default Risk dataset ingestion via Kaggle API.

Main functions:
- download_home_credit: downloads and extracts Kaggle files
- validate_schema: checks presence of required columns
- partition_by_date: partitions processed/ by reference year
"""

import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import PROCESSED_DIR, RAW_DIR

KAGGLE_COMPETITION = "home-credit-default-risk"

# Expected files after download
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

# Minimum required columns per file
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
    """Downloads the Home Credit dataset via Kaggle API.

    Requires ~/.kaggle/kaggle.json with valid credentials.

    Args:
        output_dir: Destination directory for raw files.
        force: If True, re-downloads even if files already exist.

    Returns:
        Path to the directory with extracted files.
    """
    target_dir = output_dir / "home_credit"

    if not force and target_dir.exists() and any(target_dir.glob("*.csv")):
        logger.info(f"Data already exists at {target_dir}. Use force=True to re-download.")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError("Install kaggle: uv add kaggle")

    logger.info(f"Downloading competition '{KAGGLE_COMPETITION}' to {target_dir}...")
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
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

    logger.info("Extracting ZIP files...")
    for zip_path in target_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        zip_path.unlink()
        logger.debug(f"Extracted: {zip_path.name}")

    logger.success(f"Download complete. Files at {target_dir}")
    return target_dir


def validate_schema(data_dir: Path = RAW_DIR / "home_credit") -> bool:
    """Checks that expected files exist and contain the minimum required columns.

    Args:
        data_dir: Directory with Home Credit CSV files.

    Returns:
        True if everything is valid, raises ValueError otherwise.
    """
    errors: list[str] = []

    for filename in EXPECTED_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            errors.append(f"Missing file: {filename}")
            continue

        if filename in REQUIRED_COLUMNS:
            df_head = pd.read_csv(filepath, nrows=0)
            missing = set(REQUIRED_COLUMNS[filename]) - set(df_head.columns)
            if missing:
                errors.append(f"{filename}: missing columns {missing}")

    if errors:
        for err in errors:
            logger.error(err)
        raise ValueError(f"Validation failed with {len(errors)} error(s). Check logs.")

    logger.success(f"Schema validated — {len(EXPECTED_FILES)} file(s) OK.")
    return True


def load_application_train(data_dir: Path = RAW_DIR / "home_credit") -> pd.DataFrame:
    """Loads application_train.csv with correct types.

    Args:
        data_dir: Directory with Home Credit CSV files.

    Returns:
        DataFrame with typed columns.
    """
    path = data_dir / "application_train.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded: {df.shape[0]:,} contracts × {df.shape[1]} columns")
    return df


def partition_by_date(
    df: pd.DataFrame,
    date_col: str = "DAYS_BIRTH",
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Saves the DataFrame partitioned into parquet by approximate year.

    Home Credit has no absolute date — uses DAYS_BIRTH as an ordering proxy
    to simulate temporal partitioning.

    Args:
        df: DataFrame to partition.
        date_col: Reference temporal column.
        output_dir: Root output directory.
    """
    if date_col not in df.columns:
        logger.warning(f"Column {date_col} not found. Saving without partition.")
        out = output_dir / "application_train.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Saved to {out}")
        return

    # Converts DAYS_BIRTH (negative years) → 5 bins
    df = df.copy()
    df["_year_bin"] = pd.cut(df[date_col], bins=5, labels=False)

    for bin_id, group in df.groupby("_year_bin"):
        out_path = output_dir / f"application_train_bin{int(bin_id)}.parquet"
        group.drop(columns=["_year_bin"]).to_parquet(out_path, index=False)
        logger.debug(f"Partition {bin_id}: {len(group):,} records → {out_path.name}")

    logger.success(f"Partitioning complete. Files at {output_dir}")


if __name__ == "__main__":
    data_dir = download_home_credit()
    validate_schema(data_dir)
