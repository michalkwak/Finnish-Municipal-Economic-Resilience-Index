import pandas as pd
import pytest

from resilience.validate import DataValidationError, validate_index, validate_raw


def _raw(**overrides):
    data = {"AreaCode": ["KU1", "KU2"], "Area": ["X", "Y"], "Year": [2024, 2024], "V": [1.0, 2.0]}
    data.update(overrides)
    return pd.DataFrame(data)


def test_valid_raw_passes():
    validate_raw(_raw(), "V", "t", min_value=0, max_value=100)


def test_missing_column():
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_raw(_raw().drop(columns="Area"), "V", "t")


def test_duplicate_keys():
    with pytest.raises(DataValidationError, match="share a key"):
        validate_raw(_raw(AreaCode=["KU1", "KU1"]), "V", "t")


def test_out_of_range():
    with pytest.raises(DataValidationError, match="outside"):
        validate_raw(_raw(V=[50.0, 101.0]), "V", "t", min_value=0, max_value=100)


def test_nan_values_are_allowed_in_raw():
    validate_raw(_raw(V=[1.0, None]), "V", "t", min_value=0)


def test_empty_table():
    with pytest.raises(DataValidationError, match="empty"):
        validate_raw(_raw().iloc[0:0], "V", "t")


def test_missing_area_code():
    with pytest.raises(DataValidationError, match="AreaCode"):
        validate_raw(_raw(AreaCode=["KU1", None]), "V", "t")


def _index(**overrides):
    data = {
        "MunicipalityCode": ["KU1"],
        "Municipality": ["X"],
        "Year": [2024],
        "income_score": [0.5],
        "pop_score": [0.5],
        "employment_score": [0.5],
        "EconomicResilienceIndex": [0.5],
        "EconomicResilienceIndex_100": [50.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_valid_index_passes():
    validate_index(_index())


def test_index_rejects_nan_scores():
    with pytest.raises(DataValidationError, match="NaN"):
        validate_index(_index(income_score=[None]))


def test_index_rejects_score_above_one():
    with pytest.raises(DataValidationError, match="outside"):
        validate_index(_index(pop_score=[1.2]))
