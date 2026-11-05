"""Paths, index variants, shocks and the Statistics Finland table definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
VARIANTS_DIR = RESULTS_DIR / "variants"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

PXWEB_BASE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin"

# Statistics Finland area codes: "KU" = municipality (kunta). Other prefixes are
# aggregates (SSS whole country, MK region, SK sub-region, HV wellbeing area).
MUNICIPALITY_CODE_PREFIX = "KU"
NATIONAL_CODE = "SSS"

KEY_COLS = ["MunicipalityCode", "Year"]


# --------------------------------------------------------------------------- weights


@dataclass(frozen=True)
class Weights:
    """Weights of the three components. Must be non-negative and sum to 1."""

    employment: float = 0.45
    income: float = 0.35
    demographic: float = 0.20

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(v < 0 for v in values):
            raise ValueError(f"Weights must be non-negative, got {self}")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError(f"Weights must sum to 1, got {sum(values):.6f} ({self})")

    def as_tuple(self) -> tuple[float, float, float]:
        """In component order: employment, income, demographic."""
        return (self.employment, self.income, self.demographic)


# --------------------------------------------------------------------------- index variants

Scaling = Literal["minmax", "rank", "robust_z"]
Scope = Literal["year", "pooled"]
Demographic = Literal["log_population", "growth_5y", "dependency_ratio"]


@dataclass(frozen=True)
class IndexSpec:
    """How to turn the indicators into an index. ``IndexSpec()`` is the original (baseline) method.

    scaling   minmax   - (x - min) / (max - min); sensitive to outliers
              rank     - percentile rank; ignores how far apart municipalities are
              robust_z - (x - median) / (IQR / 1.349), clipped to +-3, mapped to [0, 1]
    scope     year     - scale within each year: a ranking against that year's peers
              pooled   - scale over all years at once: scores are comparable over time
    demographic  which indicator fills the third component
    real_income  deflate gross income to constant euros of the latest year with the CPI
    """

    name: str = "baseline"
    scaling: Scaling = "minmax"
    scope: Scope = "year"
    demographic: Demographic = "log_population"
    real_income: bool = False
    weights: Weights = field(default_factory=Weights)
    description: str = ""


VARIANTS: dict[str, IndexSpec] = {
    spec.name: spec
    for spec in (
        IndexSpec(
            name="baseline",
            description="v1 method: per-year min-max, nominal income, log population.",
        ),
        IndexSpec(
            name="real_pooled",
            scope="pooled",
            real_income=True,
            description="Min-max over all years with CPI-deflated income: scores comparable "
            "over time.",
        ),
        IndexSpec(
            name="rank",
            scaling="rank",
            description="Per-year percentile ranks: immune to outliers such as Kauniainen.",
        ),
        IndexSpec(
            name="dependency",
            demographic="dependency_ratio",
            description="Economic dependency ratio (non-employed per employed) replaces "
            "log population.",
        ),
        IndexSpec(
            name="growth",
            demographic="growth_5y",
            description="5-year population growth replaces log population.",
        ),
        IndexSpec(
            name="v2",
            scaling="robust_z",
            scope="pooled",
            demographic="dependency_ratio",
            real_income=True,
            description="Proposed: robust z-scores pooled over years, real income, "
            "dependency ratio.",
        ),
    )
}
DEFAULT_VARIANT = "v2"  # written to results/; the others go to results/variants/<name>/


# --------------------------------------------------------------------------- shocks


@dataclass(frozen=True)
class Shock:
    """A national employment shock: peak -> trough (resistance), trough -> end (recovery).

    Years refer to employment on 31 December, chosen from the national series of
    employed persons (table 115b).
    """

    key: str
    label: str
    peak: int
    trough: int
    recovery_end: int

    def __post_init__(self) -> None:
        if not self.peak < self.trough < self.recovery_end:
            raise ValueError(f"{self.key}: need peak < trough < recovery_end")


SHOCKS: tuple[Shock, ...] = (
    Shock("depression_1990s", "1990s depression", 1989, 1993, 2000),
    Shock("financial_crisis", "2008 financial crisis", 2008, 2009, 2011),
    Shock("stagnation_2010s", "2010s stagnation", 2011, 2015, 2019),
    Shock("covid", "COVID-19", 2019, 2020, 2022),
)

# Municipalities with fewer employed people than this at the shock peak get a
# `small` flag: their percentage changes are noisy.
SMALL_MUNICIPALITY_EMPLOYED = 1000


# --------------------------------------------------------------------------- source tables


@dataclass(frozen=True)
class TableSpec:
    """One Statistics Finland PxWeb table reduced to a single indicator."""

    key: str  # file stem in data/raw
    value_col: str  # indicator column name in the pipeline
    table_path: str  # path below PXWEB_BASE_URL
    selections: dict[str, list[str]] = field(default_factory=dict)  # fixed single-value dims
    area_dim: str | None = "alue_23_20250101"  # None: national-only table
    time_dim: str = "timeperiod_y"
    min_value: float | None = None
    max_value: float | None = None

    @property
    def url(self) -> str:
        return f"{PXWEB_BASE_URL}/{self.table_path}"

    @property
    def filename(self) -> str:
        return f"{self.key}.csv"


_TOTALS_115B = {
    "sukupuoli_9_20180101": ["SSS"],  # sex: Total
    "ikaryhma_10_20180101": ["SSS"],  # age: Total
    "contentscode": ["tyokay-vaesto"],  # Population 31 Dec
}

TABLES: dict[str, TableSpec] = {
    # --- core indicators of the index
    "gross_income": TableSpec(
        key="gross_income",
        value_col="GrossIncome",
        table_path="tjt/118w.px",  # Income of household-dwelling units by region
        selections={"contentscode": ["bruttotu"]},  # 6. GROSS INCOME, mean (EUR)
        min_value=0,
    ),
    "population": TableSpec(
        key="population",
        value_col="Population",
        table_path="tyokay/115b.px",  # Population by area, main type of activity, sex, age
        selections={"paaasial_toimin_3_20180101": ["SSS"], **_TOTALS_115B},  # activity: Total
        min_value=0,
    ),
    "unemployment_rate": TableSpec(
        key="unemployment_rate",
        value_col="UnemploymentRate",
        table_path="tyokay/115x.px",  # Key indicators of employment statistics
        selections={"contentscode": ["tyokay-tyottomyysaste"]},  # Unemployment rate 18-64, %
        min_value=0,
        max_value=100,
    ),
    # --- auxiliary tables used by variants and analyses
    "dependency_ratio": TableSpec(
        key="dependency_ratio",
        value_col="DependencyRatio",
        table_path="tyokay/115x.px",
        # Economic dependency ratio: unemployed + outside labour force per 100 employed
        selections={"contentscode": ["taloudellinenhuoltosuhde"]},
        min_value=0,
    ),
    "employed": TableSpec(
        key="employed",
        value_col="Employed",
        table_path="tyokay/115b.px",
        selections={"paaasial_toimin_3_20180101": ["11"], **_TOTALS_115B},  # activity: Employed
        min_value=0,
    ),
    "cpi": TableSpec(
        key="cpi",
        value_col="CPI",
        table_path="khi/11xt.px",  # Consumer Price Indices, overall index, yearly data
        selections={"indeksisarja_1_20160101": ["0_1995"], "contentscode": ["Pisteluku"]},
        area_dim=None,
        min_value=0,
    ),
}

CORE_TABLES = ("gross_income", "population", "unemployment_rate")
