"""Read and write the training table.

Parquet when an engine is available, CSV otherwise. Parquet is preferred — it
keeps dtypes and timezones, and is roughly 5x smaller — but a missing pyarrow
should degrade the format, not stop the pipeline. Reviewers should not have to
debug a storage engine to see the model train.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PARQUET = "orders.parquet"
CSV = "orders.csv"


def _has_parquet_engine() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def write_table(df: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    if _has_parquet_engine():
        path = directory / PARQUET
        df.to_parquet(path, index=False)
        # Drop a stale CSV so `read_table` cannot pick up an older copy.
        (directory / CSV).unlink(missing_ok=True)
        return path

    log.warning(
        "No parquet engine (pyarrow/fastparquet) — writing CSV instead. "
        "Install pyarrow for smaller files and preserved dtypes."
    )
    path = directory / CSV
    df.to_csv(path, index=False)
    return path


def find_table(directory: Path) -> Path | None:
    """Newest of whichever format is present."""
    candidates = [p for p in (directory / PARQUET, directory / CSV) if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # CSV loses the timestamp type, and the whole pipeline depends on
        # created_at being sortable as a datetime.
        df = pd.read_csv(path, parse_dates=["created_at"])

    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df
