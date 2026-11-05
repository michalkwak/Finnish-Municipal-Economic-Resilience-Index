"""Schema and value checks run at each pipeline stage."""

from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a table does not meet the pipeline's expectations."""


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}; got {list(df.columns)}")


def require_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dupes = df[df.duplicated(keys, keep=False)]
    if not dupes.empty:
        sample = dupes[keys].drop_duplicates().head(5).to_dict("records")
        raise DataValidationError(f"{name}: {len(dupes)} rows share a key {keys}, e.g. {sample}")


def require_range(
    df: pd.DataFrame,
    column: str,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    values = df[column].dropna()
    bad = pd.Series(False, index=values.index)
    if min_value is not None:
        bad |= values < min_value
    if max_value is not None:
        bad |= values > max_value
    if bad.any():
        raise DataValidationError(
            f"{name}: {int(bad.sum())} values of {column!r} outside "
            f"[{min_value}, {max_value}], e.g. {values[bad].head(5).tolist()}"
        )


def validate_raw(
    df: pd.DataFrame,
    value_col: str,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """Checks for a tidy raw indicator table (``AreaCode, Area, Year, value``)."""
    require_columns(df, ["AreaCode", "Area", "Year", value_col], name)
    if df.empty:
        raise DataValidationError(f"{name}: table is empty")
    if df["AreaCode"].isna().any():
        raise DataValidationError(f"{name}: AreaCode has missing values")
    require_unique(df, ["AreaCode", "Year"], name)
    require_range(df, "Year", name, 1900, 2100)
    require_range(df, value_col, name, min_value, max_value)


def validate_index(df: pd.DataFrame) -> None:
    """Checks for the final index table."""
    name = "index"
    score_cols = [
        "income_score",
        "demographic_score",
        "employment_score",
        "EconomicResilienceIndex",
    ]
    require_columns(
        df,
        ["MunicipalityCode", "Municipality", "Year", *score_cols, "EconomicResilienceIndex_100"],
        name,
    )
    require_unique(df, ["MunicipalityCode", "Year"], name)
    for col in score_cols:
        if df[col].isna().any():
            raise DataValidationError(f"{name}: {col} has {int(df[col].isna().sum())} NaNs")
        require_range(df, col, name, 0.0, 1.0)
    require_range(df, "EconomicResilienceIndex_100", name, 0.0, 100.0)
