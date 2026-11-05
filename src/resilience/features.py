"""Derived indicators: real income and population growth."""

from __future__ import annotations

import pandas as pd

from resilience.config import KEY_COLS, NATIONAL_CODE

GROWTH_YEARS = 5


def add_real_income(
    df: pd.DataFrame, cpi: pd.DataFrame, base_year: int | None = None
) -> pd.DataFrame:
    """Add ``RealGrossIncome``: gross income in constant euros of ``base_year``.

    ``base_year`` defaults to the latest year in ``df``. ``cpi`` is the national CPI
    table (``AreaCode, Area, Year, CPI``).
    """
    national = cpi.loc[cpi["AreaCode"] == NATIONAL_CODE].set_index("Year")["CPI"].dropna()
    base_year = int(df["Year"].max()) if base_year is None else base_year
    missing = sorted(set(df["Year"]) - set(national.index))
    if missing or base_year not in national.index:
        raise ValueError(f"CPI missing for years {missing or [base_year]}")
    out = df.copy()
    out["RealGrossIncome"] = out["GrossIncome"] * national[base_year] / out["Year"].map(national)
    out.attrs["real_income_base_year"] = base_year
    return out


def population_growth(population: pd.DataFrame, years: int = GROWTH_YEARS) -> pd.DataFrame:
    """Population change over the previous ``years`` years, as a fraction.

    Takes a municipality-level population frame (``MunicipalityCode, Municipality, Year,
    Population``) and returns the same keys with ``PopGrowth{years}y``. Joins on
    ``Year - years`` explicitly, so gaps in the series give NaN instead of a wrong lag.
    """
    col = f"PopGrowth{years}y"
    lagged = population[["MunicipalityCode", "Year", "Population"]].copy()
    lagged["Year"] += years
    out = population.merge(lagged, on=KEY_COLS, how="left", suffixes=("", "_lag"))
    out[col] = out["Population"] / out["Population_lag"] - 1
    return out[["MunicipalityCode", "Municipality", "Year", col]]
