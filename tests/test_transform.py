import pandas as pd
import pytest

from resilience.transform import merge_indicators, municipalities_only

VALUE_COLS = {"a": "A", "b": "B"}


def _frame(value_col, rows):
    return pd.DataFrame(rows, columns=["MunicipalityCode", "Municipality", "Year", value_col])


def test_municipalities_only_drops_aggregates_and_renames():
    raw = pd.DataFrame(
        {
            "AreaCode": pd.array(["SSS", "MK01", "SK011", "HV01", "KU091"], dtype="string"),
            "Area": ["WHOLE COUNTRY", "Uusimaa", "Helsinki sub-region", "Wellbeing", "Helsinki"],
            "Year": [2024] * 5,
            "X": [1, 2, 3, 4, 5],
        }
    )
    out = municipalities_only(raw)
    assert out["MunicipalityCode"].tolist() == ["KU091"]
    assert {"MunicipalityCode", "Municipality"} <= set(out.columns)


def test_merge_trims_to_common_years():
    a = _frame("A", [("KU1", "X", y, 1.0) for y in (2020, 2021, 2022)])
    b = _frame("B", [("KU1", "X", y, 2.0) for y in (2021, 2022, 2023)])
    res = merge_indicators({"a": a, "b": b}, VALUE_COLS)
    assert res.common_years == (2021, 2022)
    assert res.merged["Year"].tolist() == [2021, 2022]
    assert res.dropped.empty  # out-of-span years are trimmed, not "dropped"


def test_merge_reports_missing_values_and_missing_rows():
    a = _frame("A", [("KU1", "X", 2024, 1.0), ("KU2", "Y", 2024, None), ("KU3", "Z", 2024, 3.0)])
    b = _frame("B", [("KU1", "X", 2024, 1.0), ("KU2", "Y", 2024, 2.0)])  # KU3 absent
    res = merge_indicators({"a": a, "b": b}, VALUE_COLS)
    assert res.merged["MunicipalityCode"].tolist() == ["KU1"]
    reasons = dict(zip(res.dropped["MunicipalityCode"], res.dropped["missing"], strict=True))
    assert reasons == {"KU2": "A", "KU3": "B"}
    assert res.dropped.loc[res.dropped.MunicipalityCode == "KU3", "Municipality"].item() == "Z"


def test_merge_flags_name_conflicts():
    a = _frame("A", [("KU1", "Maarianhamina", 2024, 1.0)])
    b = _frame("B", [("KU1", "Mariehamn", 2024, 2.0)])
    res = merge_indicators({"a": a, "b": b}, VALUE_COLS)
    assert set(res.name_conflicts["Municipality"]) == {"Maarianhamina", "Mariehamn"}
    assert res.merged["Municipality"].item() == "Maarianhamina"  # first table wins


def test_merge_rejects_disjoint_years():
    a = _frame("A", [("KU1", "X", 2000, 1.0)])
    b = _frame("B", [("KU1", "X", 2010, 1.0)])
    with pytest.raises(ValueError, match="share no years"):
        merge_indicators({"a": a, "b": b}, VALUE_COLS)
