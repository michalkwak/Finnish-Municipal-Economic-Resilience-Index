import json

import pandas as pd
import pytest

from resilience.cli import main
from resilience.pipeline import DROPPED_FILE, LONG_FILE, SUMMARY_FILE, WIDE_FILE, build


def test_build_on_tiny_data(tiny_raw_dir, tmp_path):
    out_dir = tmp_path / "results"
    result = build(tiny_raw_dir, out_dir)

    idx = result.index
    # Aggregates (SSS, MK01) excluded; Sottunga dropped for missing unemployment.
    assert set(idx["Municipality"]) == {"Helsinki", "Tampere"}
    assert sorted(idx["Year"].unique()) == [2023, 2024]  # population's 2022 trimmed
    assert set(result.dropped["Municipality"]) == {"Sottunga"}
    assert (result.dropped["missing"] == "UnemploymentRate").all()

    for name in (LONG_FILE, WIDE_FILE, DROPPED_FILE, SUMMARY_FILE):
        assert (out_dir / name).exists()
    summary = json.loads((out_dir / SUMMARY_FILE).read_text("utf-8"))
    assert summary["municipalities"] == 2
    assert summary["dropped_rows"] == 2


def test_missing_raw_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="resilience fetch"):
        build(tmp_path, tmp_path / "out")


def test_full_pipeline_on_real_data(real_raw_dir, tmp_path):
    result = build(real_raw_dir, tmp_path)
    idx = result.index
    assert idx["MunicipalityCode"].nunique() > 300
    assert idx["MunicipalityCode"].str.startswith("KU").all()
    assert idx["Year"].min() == 1995
    assert idx["EconomicResilienceIndex_100"].between(0, 100).all()
    # Every year has a best (100) and worst (0) municipality only if one leads on all
    # components, so just check that the index spreads across the scale each year.
    spread = idx.groupby("Year")["EconomicResilienceIndex_100"].agg(lambda s: s.max() - s.min())
    assert (spread > 30).all()
    assert {"Helsinki", "Tampere", "Turku", "Oulu"} <= set(idx["Municipality"])


def test_cli_build_and_plot(tiny_raw_dir, tmp_path, capsys):
    out_dir = tmp_path / "results"
    assert main(["build", "--raw-dir", str(tiny_raw_dir), "--results-dir", str(out_dir)]) == 0
    png = tmp_path / "fig.png"
    assert (
        main(["plot", "Helsinki", "Tampere", "--results-dir", str(out_dir), "--out", str(png)]) == 0
    )
    assert png.stat().st_size > 1000
    assert "Saved" in capsys.readouterr().out


def test_cli_rejects_bad_weights(tiny_raw_dir, tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        main(
            [
                "build",
                "--raw-dir",
                str(tiny_raw_dir),
                "--results-dir",
                str(tmp_path),
                "--w-employment",
                "0.9",
            ]
        )


def test_plot_unknown_municipality(tiny_raw_dir, tmp_path):
    from resilience.plots import plot_municipalities

    idx = pd.DataFrame(
        {"Municipality": ["Helsinki"], "Year": [2024], "EconomicResilienceIndex_100": [50.0]}
    )
    with pytest.raises(ValueError, match="Atlantis"):
        plot_municipalities(idx, ["Atlantis"], tmp_path / "x.png")
