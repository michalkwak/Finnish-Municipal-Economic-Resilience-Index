"""Economic Resilience Index: scale three components and take a weighted sum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from resilience.config import IndexSpec, Weights
from resilience.features import GROWTH_YEARS
from resilience.scaling import minmax, scale

__all__ = ["COMPONENT_SCORES", "compute_index", "minmax", "required_columns", "to_wide"]

# Score columns in the same order as Weights.as_tuple().
COMPONENT_SCORES = ["employment_score", "income_score", "demographic_score"]

# demographic option -> (input column, higher is better)
DEMOGRAPHIC_INPUTS: dict[str, tuple[str, bool]] = {
    "log_population": ("LogPopulation", True),
    "growth_5y": (f"PopGrowth{GROWTH_YEARS}y", True),
    "dependency_ratio": ("DependencyRatio", False),
}


def required_columns(spec: IndexSpec) -> list[str]:
    """Indicator columns ``compute_index`` needs for this spec."""
    income = "RealGrossIncome" if spec.real_income else "GrossIncome"
    demographic = DEMOGRAPHIC_INPUTS[spec.demographic][0]
    if demographic == "LogPopulation":
        demographic = "Population"
    return ["UnemploymentRate", income, demographic]


def compute_index(df: pd.DataFrame, spec: IndexSpec | None = None) -> pd.DataFrame:
    """Add component scores and the index (0-1 and 0-100) to a merged indicator table.

    With the default spec every component is min-max scaled within each year, so the
    index ranks a municipality against the others in the same year (0 = worst, 100 = best).
    """
    spec = spec or IndexSpec()
    missing = [c for c in required_columns(spec) if c not in df.columns]
    if missing:
        raise KeyError(f"Variant {spec.name!r} needs columns {missing}")

    out = df.copy()
    if "Population" in out.columns:
        out["LogPopulation"] = np.log1p(out["Population"])

    income_col = "RealGrossIncome" if spec.real_income else "GrossIncome"
    demo_col, demo_higher_better = DEMOGRAPHIC_INPUTS[spec.demographic]

    def score(col: str, higher_is_better: bool = True) -> pd.Series:
        return scale(out, col, spec.scaling, spec.scope, higher_is_better)

    out["employment_score"] = score("UnemploymentRate", higher_is_better=False)
    out["income_score"] = score(income_col)
    out["demographic_score"] = score(demo_col, demo_higher_better)

    w = spec.weights
    out["EconomicResilienceIndex"] = (
        w.employment * out["employment_score"]
        + w.income * out["income_score"]
        + w.demographic * out["demographic_score"]
    )
    out["EconomicResilienceIndex_100"] = (out["EconomicResilienceIndex"] * 100).round(1)
    return out


def index_from_scores(scores: np.ndarray, weights: Weights | np.ndarray) -> np.ndarray:
    """Weighted sum of a (n, 3) score matrix; ``weights`` may be (3,) or (3, k)."""
    w = np.asarray(weights.as_tuple() if isinstance(weights, Weights) else weights)
    return scores @ w


def to_wide(index_df: pd.DataFrame) -> pd.DataFrame:
    """Municipalities as rows, years as columns, index (0-100) as values."""
    return index_df.pivot_table(
        index=["MunicipalityCode", "Municipality"],
        columns="Year",
        values="EconomicResilienceIndex_100",
    )
