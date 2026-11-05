import pandas as pd
import pytest

from resilience.config import Shock
from resilience.shocks import check_shock, martin_index, shock_resilience, shock_summary

SHOCK = Shock("test", "Test shock", 2000, 2001, 2003)


def _employed(rows):
    """rows: {code: (name, {year: employed})}"""
    return pd.DataFrame(
        [(c, n, y, v) for c, (n, series) in rows.items() for y, v in series.items()],
        columns=["AreaCode", "Area", "Year", "Employed"],
    )


def _example():
    return _employed(
        {
            # nation: -10% in the shock, +10% in recovery
            "SSS": ("WHOLE COUNTRY", {2000: 1000, 2001: 900, 2003: 990}),
            # lost nothing, then grew 20%: resistance +1, recovery +1
            "KU1": ("Steady", {2000: 2000, 2001: 2000, 2003: 2400}),
            # lost 20%, then grew 10%: resistance -1, recovery 0
            "KU2": ("Fragile", {2000: 1500, 2001: 1200, 2003: 1320}),
            # aggregate region must be ignored
            "MK01": ("Region", {2000: 3500, 2001: 3200, 2003: 3720}),
        }
    )


def test_martin_index_values():
    r = shock_resilience(_example(), [SHOCK]).set_index("Municipality")
    assert set(r.index) == {"Steady", "Fragile"}
    assert r.loc["Steady", "Resistance"] == pytest.approx(1.0)
    assert r.loc["Steady", "Recovery"] == pytest.approx(1.0)
    assert r.loc["Fragile", "Resistance"] == pytest.approx(-1.0)
    assert r.loc["Fragile", "Recovery"] == pytest.approx(0.0)
    assert r.loc["Steady", "Type"] == "Resistant, strong recovery"
    assert r.loc["Fragile", "Type"] == "Vulnerable, weak recovery"
    assert r["NationalChangeShock"].iloc[0] == pytest.approx(-0.1)


def test_small_flag():
    r = shock_resilience(_example(), [SHOCK]).set_index("Municipality")
    assert not r.loc["Steady", "Small"]  # 2000 employed >= 1000


def test_shock_must_be_a_decline():
    nat = pd.Series({2000: 900.0, 2001: 1000.0, 2003: 1100.0})
    with pytest.raises(ValueError, match="did not fall"):
        check_shock(nat, SHOCK)


def test_shock_years_must_exist():
    nat = pd.Series({2000: 1000.0, 2001: 900.0})
    with pytest.raises(ValueError, match="2003"):
        check_shock(nat, SHOCK)


def test_needs_national_row():
    df = _example()
    with pytest.raises(ValueError, match="national"):
        shock_resilience(df[df["AreaCode"] != "SSS"], [SHOCK])


def test_martin_index_zero_national_change_is_nan():
    assert martin_index(pd.Series([0.1, -0.1]), 0.0).isna().all()


def test_shock_definition_order():
    with pytest.raises(ValueError, match="peak < trough"):
        Shock("bad", "Bad", 2005, 2003, 2010)


def test_summary_counts_and_wide_columns():
    second = Shock("second", "Second", 2001, 2003, 2004)
    df = _employed(
        {
            "SSS": ("WHOLE COUNTRY", {2000: 1000, 2001: 900, 2003: 850, 2004: 900}),
            "KU1": ("A", {2000: 2000, 2001: 2000, 2003: 2000, 2004: 2200}),
        }
    )
    s = shock_summary(shock_resilience(df, [SHOCK, second]))
    row = s.iloc[0]
    assert row["ShocksMeasured"] == 2
    assert row["TimesResistant"] == 2
    assert {"Resistance_test", "Recovery_second"} <= set(s.columns)
