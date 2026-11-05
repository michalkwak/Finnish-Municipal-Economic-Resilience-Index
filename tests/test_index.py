import numpy as np
import pandas as pd
import pytest

from resilience.config import Weights
from resilience.index import compute_index, minmax, to_wide


class TestMinmax:
    def test_scales_to_unit_interval(self):
        out = minmax(pd.Series([10.0, 20.0, 30.0]))
        assert out.tolist() == [0.0, 0.5, 1.0]

    def test_constant_series_maps_to_half(self):
        assert minmax(pd.Series([5.0, 5.0, 5.0])).tolist() == [0.5, 0.5, 0.5]

    def test_single_value_maps_to_half(self):
        assert minmax(pd.Series([42.0])).tolist() == [0.5]

    def test_all_nan_stays_nan(self):
        assert minmax(pd.Series([np.nan, np.nan])).isna().all()

    def test_nan_is_preserved_and_ignored_for_bounds(self):
        out = minmax(pd.Series([0.0, np.nan, 10.0]))
        assert out.iloc[0] == 0.0
        assert np.isnan(out.iloc[1])
        assert out.iloc[2] == 1.0

    def test_keeps_index(self):
        s = pd.Series([1.0, 2.0], index=["a", "b"])
        assert list(minmax(s).index) == ["a", "b"]


class TestWeights:
    def test_defaults_sum_to_one(self):
        w = Weights()
        assert w.employment + w.income + w.population == pytest.approx(1.0)

    def test_rejects_bad_sum(self):
        with pytest.raises(ValueError, match="sum to 1"):
            Weights(0.5, 0.5, 0.5)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            Weights(1.2, -0.1, -0.1)


def _merged():
    return pd.DataFrame(
        {
            "MunicipalityCode": ["KU1", "KU2", "KU3"] * 2,
            "Municipality": ["Best", "Mid", "Worst"] * 2,
            "Year": [2023] * 3 + [2024] * 3,
            "GrossIncome": [70000, 50000, 30000, 80000, 60000, 40000],
            "Population": [500000, 50000, 500, 510000, 49000, 480],
            "UnemploymentRate": [5.0, 10.0, 20.0, 4.0, 9.0, 25.0],
        }
    )


class TestComputeIndex:
    def test_best_and_worst_get_extremes(self):
        out = compute_index(_merged())
        assert out.loc[out.Municipality == "Best", "EconomicResilienceIndex_100"].eq(100).all()
        assert out.loc[out.Municipality == "Worst", "EconomicResilienceIndex_100"].eq(0).all()

    def test_employment_score_inverts_unemployment(self):
        out = compute_index(_merged())
        lowest_unemp = out.loc[out.groupby("Year")["UnemploymentRate"].idxmin()]
        assert (lowest_unemp["employment_score"] == 1.0).all()

    def test_scaling_is_per_year(self):
        df = _merged()
        df.loc[df.Year == 2024, "GrossIncome"] *= 10  # level shift in one year only
        a = compute_index(_merged())
        b = compute_index(df)
        pd.testing.assert_series_equal(a["income_score"], b["income_score"])

    def test_weights_are_applied(self):
        out = compute_index(_merged(), Weights(1.0, 0.0, 0.0))
        pd.testing.assert_series_equal(
            out["EconomicResilienceIndex"], out["employment_score"], check_names=False
        )

    def test_does_not_mutate_input(self):
        df = _merged()
        compute_index(df)
        assert "income_score" not in df.columns

    def test_to_wide_shape(self):
        wide = to_wide(compute_index(_merged()))
        assert wide.shape == (3, 2)
        assert list(wide.columns) == [2023, 2024]
