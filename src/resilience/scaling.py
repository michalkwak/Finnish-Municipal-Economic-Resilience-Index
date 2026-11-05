"""Normalisation methods. Each maps a series to [0, 1], higher = higher raw value.

A constant (or single-value) series maps to 0.5 and NaNs stay NaN for every method.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from resilience.config import Scaling, Scope

ROBUST_Z_CLIP = 3.0
IQR_TO_SIGMA = 1.349  # IQR of a standard normal distribution


def _constant(series: pd.Series) -> pd.Series:
    return pd.Series(np.where(series.notna(), 0.5, np.nan), index=series.index, dtype=float)


def minmax(series: pd.Series) -> pd.Series:
    """(x - min) / (max - min)."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return _constant(s)
    return (s - lo) / (hi - lo)


def percentile_rank(series: pd.Series) -> pd.Series:
    """(rank - 1) / (n - 1) with average ranks for ties: lowest 0, highest 1."""
    s = pd.to_numeric(series, errors="coerce")
    n = s.notna().sum()
    if n <= 1 or s.min() == s.max():
        return _constant(s)
    return (s.rank(method="average") - 1) / (n - 1)


def robust_z(series: pd.Series, clip: float = ROBUST_Z_CLIP) -> pd.Series:
    """Robust z-score (median / IQR), clipped to +-clip and mapped linearly to [0, 1].

    Unlike min-max, one extreme municipality cannot compress everybody else: values
    beyond `clip` robust standard deviations simply saturate at 0 or 1.
    """
    s = pd.to_numeric(series, errors="coerce")
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    sigma = (q3 - q1) / IQR_TO_SIGMA
    if pd.isna(sigma) or sigma == 0:
        return _constant(s)
    z = ((s - med) / sigma).clip(-clip, clip)
    return (z + clip) / (2 * clip)


SCALERS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "minmax": minmax,
    "rank": percentile_rank,
    "robust_z": robust_z,
}


def scale(
    df: pd.DataFrame,
    column: str,
    method: Scaling = "minmax",
    scope: Scope = "year",
    higher_is_better: bool = True,
) -> pd.Series:
    """Score a column of a panel (needs a ``Year`` column) on [0, 1], 1 = best."""
    try:
        scaler = SCALERS[method]
    except KeyError:
        raise ValueError(f"Unknown scaling {method!r}; choose from {list(SCALERS)}") from None
    if scope == "year":
        scores = df.groupby("Year")[column].transform(scaler)
    elif scope == "pooled":
        scores = scaler(df[column])
    else:
        raise ValueError(f"Unknown scope {scope!r}; choose 'year' or 'pooled'")
    return scores if higher_is_better else 1 - scores
