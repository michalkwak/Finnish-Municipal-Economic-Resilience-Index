import numpy as np
import pandas as pd
import pytest

from resilience.config import IndexSpec, Weights
from resilience.index import COMPONENT_SCORES
from resilience.sensitivity import pca_weights, ranks_desc, spearman, weight_sensitivity


def test_ranks_desc_1d_and_2d():
    assert ranks_desc(np.array([0.2, 0.9, 0.5])).tolist() == [3, 1, 2]
    two = ranks_desc(np.array([[0.2, 3.0], [0.9, 1.0], [0.5, 2.0]]))
    assert two[:, 0].tolist() == [3, 1, 2]
    assert two[:, 1].tolist() == [1, 3, 2]


def test_spearman():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_pca_weights_sum_to_one_and_drop_noise_component():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=500)
    scores = np.column_stack(
        [signal + rng.normal(0, 0.1, 500), signal + rng.normal(0, 0.1, 500), rng.normal(size=500)]
    )
    w = pca_weights(scores)
    assert sum(w.as_tuple()) == pytest.approx(1.0)
    assert w.demographic < 0.05  # uncorrelated third column gets ~no weight
    assert w.employment == pytest.approx(w.income, abs=0.05)


def _index_frame(n=50, seed=1):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(size=(n, 3))
    df = pd.DataFrame(scores, columns=COMPONENT_SCORES)
    df["MunicipalityCode"] = [f"KU{i:03d}" for i in range(n)]
    df["Municipality"] = [f"M{i}" for i in range(n)]
    df["Year"] = 2024
    df["EconomicResilienceIndex"] = scores @ np.array(Weights().as_tuple())
    return df


def test_weight_sensitivity_table_and_summary():
    res = weight_sensitivity(_index_frame(), IndexSpec(), n_draws=300, seed=3)
    t = res.table
    assert len(t) == 50
    assert t["BaseRank"].tolist() == list(range(1, 51))  # sorted by base rank
    # Base rank matches ranking the stored index directly.
    df = _index_frame()
    expected = ranks_desc(df["EconomicResilienceIndex"].to_numpy())
    got = t.set_index("MunicipalityCode").loc[df["MunicipalityCode"], "BaseRank"].to_numpy()
    assert (got == expected).all()
    # Local draws stay close to the chosen weights, global draws roam further.
    assert (t["LocalRange90"] <= t["GlobalRange90"] + 1e-9).mean() > 0.9
    assert res.summary["median_spearman_local"] > res.summary["median_spearman_global"]
    assert set(res.summary["pca_weights"]) == set(COMPONENT_SCORES)


def test_weight_sensitivity_is_reproducible():
    a = weight_sensitivity(_index_frame(), n_draws=200, seed=7).table
    b = weight_sensitivity(_index_frame(), n_draws=200, seed=7).table
    pd.testing.assert_frame_equal(a, b)


def test_weight_sensitivity_unknown_year():
    with pytest.raises(ValueError, match="1990"):
        weight_sensitivity(_index_frame(), year=1990)
