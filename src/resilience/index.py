"""Economic Resilience Index: per-year min-max scaling and a weighted sum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from resilience.config import Weights


def minmax(series: pd.Series) -> pd.Series:
    """Scale to [0, 1]. A constant series maps to 0.5; NaNs stay NaN."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=series.index, dtype=float)
    return (s - lo) / (hi - lo)


def compute_index(df: pd.DataFrame, weights: Weights | None = None) -> pd.DataFrame:
    """Add component scores and the index to a merged indicator table.

    Every component is min-max scaled within each year, so the index ranks a
    municipality against the others in the same year (0 = worst, 100 = best).
    """
    weights = weights or Weights()
    out = df.copy()
    out["LogPopulation"] = np.log1p(out["Population"])
    by_year = out.groupby("Year")

    out["income_score"] = by_year["GrossIncome"].transform(minmax)
    out["pop_score"] = by_year["LogPopulation"].transform(minmax)
    out["employment_score"] = 1 - by_year["UnemploymentRate"].transform(minmax)

    out["EconomicResilienceIndex"] = (
        weights.employment * out["employment_score"]
        + weights.income * out["income_score"]
        + weights.population * out["pop_score"]
    )
    out["EconomicResilienceIndex_100"] = (out["EconomicResilienceIndex"] * 100).round(1)
    return out


def to_wide(index_df: pd.DataFrame) -> pd.DataFrame:
    """Municipalities as rows, years as columns, index (0-100) as values."""
    return index_df.pivot_table(
        index=["MunicipalityCode", "Municipality"],
        columns="Year",
        values="EconomicResilienceIndex_100",
    )
