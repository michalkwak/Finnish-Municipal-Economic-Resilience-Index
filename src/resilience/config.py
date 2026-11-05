"""Paths, index weights and the Statistics Finland table definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

PXWEB_BASE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin"

# Statistics Finland area codes: "KU" = municipality (kunta). Other prefixes are
# aggregates (SSS whole country, MK region, SK sub-region, HV wellbeing area).
MUNICIPALITY_CODE_PREFIX = "KU"

KEY_COLS = ["MunicipalityCode", "Year"]


@dataclass(frozen=True)
class Weights:
    """Weights of the three components. Must be non-negative and sum to 1."""

    employment: float = 0.45
    income: float = 0.35
    population: float = 0.20

    def __post_init__(self) -> None:
        values = (self.employment, self.income, self.population)
        if any(v < 0 for v in values):
            raise ValueError(f"Weights must be non-negative, got {self}")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError(f"Weights must sum to 1, got {sum(values):.6f} ({self})")


@dataclass(frozen=True)
class TableSpec:
    """One Statistics Finland PxWeb table reduced to a single indicator."""

    key: str  # file stem in data/raw
    value_col: str  # indicator column name in the pipeline
    table_path: str  # path below PXWEB_BASE_URL
    selections: dict[str, list[str]] = field(default_factory=dict)  # fixed single-value dims
    area_dim: str = "alue_23_20250101"
    time_dim: str = "timeperiod_y"
    min_value: float | None = None
    max_value: float | None = None

    @property
    def url(self) -> str:
        return f"{PXWEB_BASE_URL}/{self.table_path}"

    @property
    def filename(self) -> str:
        return f"{self.key}.csv"


TABLES: dict[str, TableSpec] = {
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
        selections={
            "paaasial_toimin_3_20180101": ["SSS"],  # main type of activity: Total
            "sukupuoli_9_20180101": ["SSS"],  # sex: Total
            "ikaryhma_10_20180101": ["SSS"],  # age: Total
            "contentscode": ["tyokay-vaesto"],  # Population 31 Dec
        },
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
}
