import numpy as np
import pandas as pd
import pytest

from resilience.features import add_real_income, population_growth


def _cpi(values: dict[int, float]):
    return pd.DataFrame(
        {
            "AreaCode": "SSS",
            "Area": "WHOLE COUNTRY",
            "Year": list(values),
            "CPI": list(values.values()),
        }
    )


def test_real_income_in_latest_year_euros():
    df = pd.DataFrame({"Year": [2020, 2024], "GrossIncome": [50000.0, 60000.0]})
    out = add_real_income(df, _cpi({2020: 100.0, 2024: 120.0}))
    assert out["RealGrossIncome"].tolist() == [pytest.approx(60000.0), pytest.approx(60000.0)]
    assert out.attrs["real_income_base_year"] == 2024


def test_real_income_custom_base_year():
    df = pd.DataFrame({"Year": [2020, 2024], "GrossIncome": [50000.0, 60000.0]})
    out = add_real_income(df, _cpi({2020: 100.0, 2024: 120.0}), base_year=2020)
    assert out["RealGrossIncome"].tolist() == [pytest.approx(50000.0), pytest.approx(50000.0)]


def test_real_income_requires_cpi_for_every_year():
    df = pd.DataFrame({"Year": [2019, 2024], "GrossIncome": [1.0, 1.0]})
    with pytest.raises(ValueError, match="2019"):
        add_real_income(df, _cpi({2020: 100.0, 2024: 120.0}))


def test_population_growth_uses_exact_lag_year():
    pop = pd.DataFrame(
        {
            "MunicipalityCode": ["KU1"] * 4,
            "Municipality": ["X"] * 4,
            "Year": [2014, 2015, 2019, 2020],  # 2016-2018 missing
            "Population": [100.0, 200.0, 110.0, 220.0],
        }
    )
    out = population_growth(pop, years=5).set_index("Year")["PopGrowth5y"]
    assert out[2019] == pytest.approx(0.10)
    assert out[2020] == pytest.approx(0.10)
    assert np.isnan(out[2014]) and np.isnan(out[2015])


def test_population_growth_does_not_mix_municipalities():
    pop = pd.DataFrame(
        {
            "MunicipalityCode": ["KU1", "KU2", "KU1", "KU2"],
            "Municipality": ["A", "B", "A", "B"],
            "Year": [2015, 2015, 2020, 2020],
            "Population": [100.0, 1000.0, 150.0, 900.0],
        }
    )
    out = population_growth(pop).set_index(["MunicipalityCode", "Year"])["PopGrowth5y"]
    assert out[("KU1", 2020)] == pytest.approx(0.5)
    assert out[("KU2", 2020)] == pytest.approx(-0.1)
