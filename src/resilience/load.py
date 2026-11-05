"""Read the tidy raw indicator tables written by :mod:`resilience.fetch`."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from resilience.config import CORE_TABLES, RAW_DIR, TABLES, TableSpec
from resilience.validate import validate_raw

log = logging.getLogger(__name__)


def load_indicator(spec: TableSpec, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `resilience fetch` to download it.")
    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"AreaCode": "string", "Area": "string", "Year": "int64"},
    )
    df[spec.value_col] = pd.to_numeric(df[spec.value_col], errors="coerce")
    validate_raw(df, spec.value_col, spec.key, spec.min_value, spec.max_value)
    log.info(
        "Loaded %s: %d rows, years %d-%d, %d missing values",
        spec.key,
        len(df),
        df["Year"].min(),
        df["Year"].max(),
        int(df[spec.value_col].isna().sum()),
    )
    return df


def load_all(raw_dir: Path = RAW_DIR, keys: Iterable[str] = CORE_TABLES) -> dict[str, pd.DataFrame]:
    return {key: load_indicator(TABLES[key], raw_dir) for key in keys}
