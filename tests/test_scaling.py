import numpy as np
import pandas as pd
import pytest

from resilience.scaling import minmax, percentile_rank, robust_z, scale


@pytest.mark.parametrize("fn", [minmax, percentile_rank, robust_z])
class TestCommonContract:
    def test_range_and_order(self, fn):
        s = pd.Series([3.0, 1.0, 2.0, 10.0, 5.0])
        out = fn(s)
        assert out.between(0, 1).all()
        assert out.rank().tolist() == s.rank().tolist()  # monotone

    def test_constant_maps_to_half(self, fn):
        assert fn(pd.Series([4.0, 4.0, 4.0])).tolist() == [0.5, 0.5, 0.5]

    def test_nan_preserved(self, fn):
        out = fn(pd.Series([1.0, np.nan, 3.0, 2.0]))
        assert np.isnan(out.iloc[1])
        assert out.drop(index=1).notna().all()


def test_percentile_rank_is_evenly_spaced_and_ignores_distance():
    assert percentile_rank(pd.Series([1.0, 2.0, 1000.0])).tolist() == [0.0, 0.5, 1.0]


def test_percentile_rank_ties_share_average():
    out = percentile_rank(pd.Series([1.0, 1.0, 5.0]))
    assert out.tolist() == [0.25, 0.25, 1.0]


def test_robust_z_resists_outliers():
    base = pd.Series(np.arange(1.0, 21.0))
    with_outlier = pd.concat([base, pd.Series([1e6])], ignore_index=True)
    # Under min-max, one outlier squashes everyone else towards 0...
    assert minmax(with_outlier).iloc[:20].max() < 0.001
    # ...under robust z, the others keep their spread and the outlier saturates.
    rz = robust_z(with_outlier)
    assert rz.iloc[:20].max() - rz.iloc[:20].min() > 0.3
    assert rz.iloc[-1] == 1.0


def test_robust_z_median_maps_to_half():
    assert robust_z(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])).iloc[2] == 0.5


def _panel():
    return pd.DataFrame({"Year": [2023, 2023, 2024, 2024], "X": [1.0, 3.0, 11.0, 13.0]})


def test_scale_per_year_vs_pooled():
    df = _panel()
    assert scale(df, "X", scope="year").tolist() == [0.0, 1.0, 0.0, 1.0]
    assert scale(df, "X", scope="pooled").tolist() == [
        0.0,
        pytest.approx(1 / 6),
        pytest.approx(5 / 6),
        1.0,
    ]


def test_scale_lower_is_better():
    assert scale(_panel(), "X", higher_is_better=False).tolist() == [1.0, 0.0, 1.0, 0.0]


def test_scale_rejects_unknown_options():
    with pytest.raises(ValueError, match="scaling"):
        scale(_panel(), "X", method="zscore")
    with pytest.raises(ValueError, match="scope"):
        scale(_panel(), "X", scope="decade")
