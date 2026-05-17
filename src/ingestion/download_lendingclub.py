"""LendingClub 2007-2018 dataset ingestion from Zenodo (record 11295916).

Downloads the cleaned ``LC_loans_granting_model_dataset.csv`` (~167 MB,
~1.35M loans, final status only), verifies MD5, and persists as parquet
for fast downstream loading.

The dataset is CC-BY 4.0 and requires no authentication.
"""

import hashlib
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from src.config import (
    LENDINGCLUB_MD5,
    LENDINGCLUB_PARQUET,
    LENDINGCLUB_RAW,
    LENDINGCLUB_URL,
)


def _md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_lendingclub(
    output_path: Path = LENDINGCLUB_RAW,
    force: bool = False,
    verify_checksum: bool = True,
) -> Path:
    """Stream-downloads the LendingClub CSV from Zenodo.

    Idempotent: skips download if the file already exists with a matching MD5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and output_path.exists():
        if verify_checksum and _md5sum(output_path) == LENDINGCLUB_MD5:
            logger.info(f"Already downloaded and checksum OK: {output_path}")
            return output_path
        if not verify_checksum:
            logger.info(f"File exists (skip checksum): {output_path}")
            return output_path
        logger.warning("Existing file has wrong checksum — re-downloading.")

    logger.info(f"Downloading LendingClub dataset → {output_path} (~167 MB)...")
    with requests.get(LENDINGCLUB_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with output_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if total and written % (16 << 20) < (1 << 20):
                    pct = 100 * written / total
                    logger.debug(f"  {written / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)")

    if verify_checksum:
        actual = _md5sum(output_path)
        if actual != LENDINGCLUB_MD5:
            raise RuntimeError(
                f"MD5 mismatch: expected {LENDINGCLUB_MD5}, got {actual}. "
                "The Zenodo record may have been updated — verify upstream."
            )
        logger.success("Checksum verified.")

    logger.success(f"Download complete: {output_path}")
    return output_path


def csv_to_parquet(
    csv_path: Path = LENDINGCLUB_RAW,
    parquet_path: Path = LENDINGCLUB_PARQUET,
    force: bool = False,
) -> Path:
    """Converts the raw CSV to parquet for fast loading downstream."""
    if not force and parquet_path.exists():
        logger.info(f"Parquet already exists: {parquet_path}")
        return parquet_path

    logger.info(f"Reading CSV → {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    logger.success(f"Saved parquet: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")
    return parquet_path


def load_lendingclub(parquet_path: Path = LENDINGCLUB_PARQUET) -> pd.DataFrame:
    """Loads the cached parquet."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run `python -m src.ingestion.download_lendingclub` first."
        )
    return pd.read_parquet(parquet_path)


if __name__ == "__main__":
    csv = download_lendingclub()
    csv_to_parquet(csv)
    df = load_lendingclub()
    logger.info(f"Sample columns: {list(df.columns)[:15]}")
    logger.info(f"Shape: {df.shape}")
