"""Filter to municipalities and join the indicators, keeping an audit of dropped rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce

import pandas as pd

from resilience.config import KEY_COLS, MUNICIPALITY_CODE_PREFIX

log = logging.getLogger(__name__)


def municipalities_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep municipality rows (``KU*`` codes) and rename to the pipeline's key names."""
    out = df[df["AreaCode"].str.startswith(MUNICIPALITY_CODE_PREFIX, na=False)]
    return out.rename(columns={"AreaCode": "MunicipalityCode", "Area": "Municipality"})


@dataclass
class MergeResult:
    merged: pd.DataFrame
    """Rows with every indicator present, restricted to the common year span."""
    dropped: pd.DataFrame
    """Municipality-years inside the common span that were dropped, with the reason."""
    common_years: tuple[int, int]
    name_conflicts: pd.DataFrame
    """Codes whose municipality name differs between source tables."""


def merge_indicators(frames: dict[str, pd.DataFrame], value_cols: dict[str, str]) -> MergeResult:
    """Outer-join indicators on (MunicipalityCode, Year) and report what an inner join loses.

    ``frames`` maps table key -> municipality-level frame; ``value_cols`` maps table key
    -> its indicator column.
    """
    if not frames:
        raise ValueError("No indicator frames to merge")

    spans = {}
    for key, df in frames.items():
        years = df.loc[df[value_cols[key]].notna(), "Year"]
        spans[key] = (int(years.min()), int(years.max()))
    start = max(s for s, _ in spans.values())
    end = min(e for _, e in spans.values())
    if start > end:
        raise ValueError(f"Indicators share no years: {spans}")
    for key, (s, e) in spans.items():
        if (s, e) != (start, end):
            log.info("%s covers %d-%d; trimmed to common span %d-%d", key, s, e, start, end)

    # Name per code from each table, to detect inconsistent naming between sources.
    names = pd.concat(
        [df[["MunicipalityCode", "Municipality"]].assign(table=key) for key, df in frames.items()]
    ).drop_duplicates()
    conflicting_codes = names.groupby("MunicipalityCode")["Municipality"].nunique()
    conflicting_codes = conflicting_codes[conflicting_codes > 1].index
    name_conflicts = names[names["MunicipalityCode"].isin(conflicting_codes)].sort_values(
        ["MunicipalityCode", "table"]
    )
    if not name_conflicts.empty:
        log.warning(
            "%d municipality codes have different names across tables", len(conflicting_codes)
        )

    in_span = {
        key: df.loc[df["Year"].between(start, end), [*KEY_COLS, value_cols[key]]]
        for key, df in frames.items()
    }
    outer = reduce(
        lambda left, right: left.merge(right, on=KEY_COLS, how="outer"), in_span.values()
    )

    # Canonical name: first table that has the code (tables are in TABLES order).
    canonical = names.drop_duplicates("MunicipalityCode")[["MunicipalityCode", "Municipality"]]
    outer = outer.merge(canonical, on="MunicipalityCode", how="left")

    cols = [value_cols[k] for k in frames]
    missing = outer[cols].isna()
    complete = ~missing.any(axis=1)

    dropped = outer.loc[~complete, ["MunicipalityCode", "Municipality", "Year"]].copy()
    dropped["missing"] = missing.loc[~complete].apply(
        lambda row: ", ".join(c for c in cols if row[c]), axis=1
    )
    dropped = dropped.sort_values(KEY_COLS).reset_index(drop=True)

    merged = outer.loc[complete, ["MunicipalityCode", "Municipality", "Year", *cols]]
    merged = merged.sort_values(KEY_COLS).reset_index(drop=True)

    log.info(
        "Merged %d municipality-years (%d municipalities, %d-%d); dropped %d incomplete rows",
        len(merged),
        merged["MunicipalityCode"].nunique(),
        start,
        end,
        len(dropped),
    )
    if not dropped.empty:
        log.warning(
            "Dropped rows by missing indicator: %s", dropped["missing"].value_counts().to_dict()
        )

    return MergeResult(merged, dropped, (start, end), name_conflicts.reset_index(drop=True))
