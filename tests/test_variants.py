"""Index variants and the analysis run, on the synthetic panel and the real data."""

import json

import pandas as pd
import pytest

from resilience.analysis import compare_variants, run_analysis
from resilience.cli import main
from resilience.config import VARIANTS, IndexSpec
from resilience.index import compute_index, required_columns
from resilience.pipeline import build_index


@pytest.mark.parametrize("name", list(VARIANTS))
def test_every_variant_builds_on_panel(panel_raw_dir, name):
    idx, merge = build_index(panel_raw_dir, VARIANTS[name])
    assert idx["MunicipalityCode"].nunique() == 6
    assert idx["EconomicResilienceIndex_100"].between(0, 100).all()
    assert merge.common_years == (1995, 2024)


def test_real_income_variant_uses_deflated_income(panel_raw_dir):
    idx, _ = build_index(panel_raw_dir, VARIANTS["real_pooled"])
    latest = idx[idx["Year"] == 2024]
    assert (latest["RealGrossIncome"] == latest["GrossIncome"]).all()  # base year
    early = idx[idx["Year"] == 1995]
    assert (early["RealGrossIncome"] > early["GrossIncome"]).all()


def test_pooled_scores_are_comparable_over_time(panel_raw_dir):
    """Per-year scaling hides a level shift; pooled scaling shows it."""
    per_year, _ = build_index(panel_raw_dir, IndexSpec(real_income=True))
    pooled, _ = build_index(panel_raw_dir, VARIANTS["real_pooled"])
    for df in (per_year, pooled):
        df.attrs.clear()
    yearly_mean = pooled.groupby("Year")["income_score"].mean()
    assert yearly_mean[2024] > yearly_mean[1995]  # real income grew in the panel
    per_year_mean = per_year.groupby("Year")["income_score"].mean()
    assert per_year_mean.std() < yearly_mean.std()


def test_dependency_variant_inverts_ratio(panel_raw_dir):
    idx, _ = build_index(panel_raw_dir, VARIANTS["dependency"])
    y = idx[idx["Year"] == 2024]
    best = y.loc[y["DependencyRatio"].idxmin()]
    assert best["demographic_score"] == 1.0


def test_compute_index_reports_missing_columns():
    with pytest.raises(KeyError, match="DependencyRatio"):
        compute_index(pd.DataFrame({"Year": [2024]}), VARIANTS["dependency"])
    assert "RealGrossIncome" in required_columns(VARIANTS["v2"])


def test_compare_variants_rank_table():
    a = pd.DataFrame(
        {
            "MunicipalityCode": ["KU1", "KU2", "KU3"],
            "Municipality": ["A", "B", "C"],
            "Year": 2024,
            "EconomicResilienceIndex": [0.9, 0.5, 0.1],
        }
    )
    b = a.assign(EconomicResilienceIndex=[0.1, 0.5, 0.9])
    ranks, corr = compare_variants({"a": a, "b": b}, 2024)
    assert ranks["Rank_a"].tolist() == [1, 2, 3]
    assert ranks["Rank_b"].tolist() == [3, 2, 1]
    assert corr.loc["a", "b"] == pytest.approx(-1.0)
    assert ranks["MaxRankShift"].tolist() == [2, 0, 2]


def test_run_analysis_on_panel(panel_raw_dir, tmp_path):
    res = run_analysis(
        panel_raw_dir,
        analysis_dir=tmp_path / "analysis",
        figures_dir=tmp_path / "figures",
        n_draws=100,
        variants_root=tmp_path / "variants",
    )
    assert set(res.indexes) == set(VARIANTS)
    assert set(res.shocks["Shock"]) == {
        "depression_1990s",
        "financial_crisis",
        "stagnation_2010s",
        "covid",
    }
    # Smallville (beta 0.4) swings least, Testby (beta 1.6) most.
    mean_res = res.shocks.groupby("Municipality")["Resistance"].mean()
    assert mean_res["Smallville"] > 0 > mean_res["Testby"]
    summary = json.loads((tmp_path / "analysis" / "analysis_summary.json").read_text("utf-8"))
    assert summary["year"] == 2024
    assert set(summary["weight_sensitivity"]) == {"baseline", "v2"}
    for fig in (
        "shock_resilience.png",
        "variant_rank_correlation.png",
        "weight_sensitivity_v2.png",
    ):
        assert (tmp_path / "figures" / fig).stat().st_size > 1000
    for name in VARIANTS:
        assert (
            tmp_path / "variants" / name / "finland_economic_resilience_index_long.csv"
        ).exists()


def test_cli_build_variant_and_weight_override(panel_raw_dir, tmp_path):
    out = tmp_path / "v2"
    args = ["build", "--variant", "v2", "--raw-dir", str(panel_raw_dir), "--results-dir", str(out)]
    assert (
        main([*args, "--w-employment", "0.45", "--w-income", "0.45", "--w-demographic", "0.1"]) == 0
    )
    summary = json.loads((out / "build_summary.json").read_text("utf-8"))
    assert summary["variant"]["name"] == "v2"
    assert summary["variant"]["weights"] == {"employment": 0.45, "income": 0.45, "demographic": 0.1}
    assert summary["real_income_base_year"] == 2024


def test_cli_variants_lists_all(capsys):
    assert main(["variants"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in VARIANTS)


def test_analysis_on_real_data(real_raw_dir, tmp_path):
    res = run_analysis(
        real_raw_dir,
        analysis_dir=tmp_path / "a",
        figures_dir=tmp_path / "f",
        n_draws=200,
        write_variants=False,
    )
    # Variants disagree on details but agree broadly.
    assert (res.variant_corr.to_numpy() > 0.85).all()
    # The four shocks are real national declines (check_shock would raise otherwise),
    # and every municipality gets a score for each.
    assert res.shocks.groupby("Shock")["MunicipalityCode"].nunique().min() > 300
