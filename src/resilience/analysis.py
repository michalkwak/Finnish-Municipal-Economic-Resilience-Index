"""`resilience analyze`: variants, weight sensitivity and shock resilience in one run."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

from resilience.config import (
    ANALYSIS_DIR,
    FIGURES_DIR,
    RAW_DIR,
    SHOCKS,
    TABLES,
    VARIANTS,
    IndexSpec,
)
from resilience.load import load_indicator
from resilience.pipeline import build, build_index, default_results_dir, write_csv, write_json
from resilience.plots import (
    HIGHLIGHT,
    plot_municipalities,
    plot_rank_uncertainty,
    plot_shock_quadrants,
    plot_variant_correlation,
)
from resilience.sensitivity import ranks_desc, weight_sensitivity
from resilience.shocks import shock_resilience, shock_summary

log = logging.getLogger(__name__)

SENSITIVITY_VARIANTS = ("baseline", "v2")


@dataclass
class AnalysisResult:
    indexes: dict[str, pd.DataFrame]
    variant_ranks: pd.DataFrame
    variant_corr: pd.DataFrame
    sensitivity: dict[str, Any]
    shocks: pd.DataFrame
    outputs: list[Path] = field(default_factory=list)


def compare_variants(
    indexes: dict[str, pd.DataFrame], year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranks of each municipality under every variant in ``year`` + Spearman matrix.

    Only municipalities present in every variant are compared.
    """
    per_variant = []
    for name, df in indexes.items():
        sub = df.loc[
            df["Year"] == year, ["MunicipalityCode", "Municipality", "EconomicResilienceIndex"]
        ].copy()
        sub[f"Rank_{name}"] = ranks_desc(sub["EconomicResilienceIndex"].to_numpy())
        per_variant.append(sub.drop(columns="EconomicResilienceIndex"))
    ranks = reduce(
        lambda a, b: a.merge(b, on=["MunicipalityCode", "Municipality"], how="inner"), per_variant
    )
    rank_cols = [f"Rank_{n}" for n in indexes]
    # Re-rank within the common set so ranks are 1..n in every column.
    for col in rank_cols:
        ranks[col] = ranks[col].rank(method="first").astype(int)
    corr = ranks[rank_cols].corr(method="pearson")  # Pearson on ranks = Spearman
    corr.index = corr.columns = list(indexes)
    first = rank_cols[0]
    ranks["MaxRankShift"] = ranks[rank_cols].max(axis=1) - ranks[rank_cols].min(axis=1)
    return ranks.sort_values(first).reset_index(drop=True), corr.round(3)


def run_analysis(
    raw_dir: Path = RAW_DIR,
    analysis_dir: Path = ANALYSIS_DIR,
    figures_dir: Path = FIGURES_DIR,
    year: int | None = None,
    n_draws: int = 2000,
    seed: int = 0,
    write_variants: bool = True,
    variants_root: Path | None = None,
) -> AnalysisResult:
    """Build every variant, then compare them, test weight sensitivity and score shocks."""
    outputs: list[Path] = []

    # 1. Variants -------------------------------------------------------------------
    indexes: dict[str, pd.DataFrame] = {}
    for name, spec in VARIANTS.items():
        if write_variants:
            out_dir = (variants_root / name) if variants_root else default_results_dir(spec)
            result = build(raw_dir, out_dir, spec)
            outputs += result.outputs
            indexes[name] = result.index
        else:
            indexes[name] = build_index(raw_dir, spec)[0]

    year = year or int(min(df["Year"].max() for df in indexes.values()))
    ranks, corr = compare_variants(indexes, year)
    outputs += [
        write_csv(ranks, analysis_dir / f"variant_ranks_{year}.csv"),
        write_csv(corr, analysis_dir / f"variant_rank_correlation_{year}.csv", index=True),
        plot_variant_correlation(corr, figures_dir / "variant_rank_correlation.png", year),
        plot_municipalities(
            indexes["baseline"],
            HIGHLIGHT,
            figures_dir / "resilience_baseline_helsinki_tampere_turku_oulu.png",
            subtitle="relative to all municipalities that year",
            title="Economic Resilience Index over time (original baseline method)",
        ),
    ]

    # 2. Weight sensitivity -----------------------------------------------------------
    sensitivity: dict[str, Any] = {}
    for name in SENSITIVITY_VARIANTS:
        spec: IndexSpec = VARIANTS[name]
        res = weight_sensitivity(indexes[name], spec, year, n_draws, seed)
        sensitivity[name] = res.summary
        outputs += [
            write_csv(res.table, analysis_dir / f"weight_sensitivity_{name}_{year}.csv"),
            plot_rank_uncertainty(
                res.table,
                figures_dir / f"weight_sensitivity_{name}.png",
                f"How much do {year} ranks depend on the weights? ({name})",
            ),
        ]

    # 3. Shock resilience ------------------------------------------------------------
    employed = load_indicator(TABLES["employed"], raw_dir)
    shocks = shock_resilience(employed, SHOCKS)
    summary_by_muni = shock_summary(shocks)
    outputs += [
        write_csv(shocks, analysis_dir / "shock_resilience.csv"),
        write_csv(summary_by_muni, analysis_dir / "shock_resilience_summary.csv"),
        plot_shock_quadrants(shocks, figures_dir / "shock_resilience.png"),
    ]

    # 4. Summary ---------------------------------------------------------------------
    big = summary_by_muni[summary_by_muni["EmployedLatestPeak"] >= 1000]
    summary = {
        "year": year,
        "variants": {n: s.description for n, s in VARIANTS.items()},
        "variant_rank_correlation": corr.to_dict(),
        "largest_rank_shifts": ranks.nlargest(10, "MaxRankShift")[
            ["Municipality", *[c for c in ranks.columns if c.startswith("Rank_")], "MaxRankShift"]
        ].to_dict("records"),
        "weight_sensitivity": sensitivity,
        "shocks": [
            {
                "key": s.key,
                "label": s.label,
                "peak": s.peak,
                "trough": s.trough,
                "recovery_end": s.recovery_end,
                "national_change_shock": round(
                    float(shocks.loc[shocks.Shock == s.key, "NationalChangeShock"].iloc[0]), 4
                ),
                "national_change_recovery": round(
                    float(shocks.loc[shocks.Shock == s.key, "NationalChangeRecovery"].iloc[0]), 4
                ),
                "type_counts": shocks.loc[shocks.Shock == s.key, "Type"].value_counts().to_dict(),
            }
            for s in SHOCKS
        ],
        "most_resistant_across_shocks": big.head(10)[
            ["Municipality", "TimesResistant", "MeanResistance", "MeanRecovery"]
        ]
        .round(3)
        .to_dict("records"),
        "least_resistant_across_shocks": big.tail(10)[
            ["Municipality", "TimesResistant", "MeanResistance", "MeanRecovery"]
        ]
        .round(3)
        .to_dict("records"),
    }
    outputs.append(write_json(summary, analysis_dir / "analysis_summary.json"))
    for p in outputs:
        log.debug("Wrote %s", p)
    return AnalysisResult(indexes, ranks, corr, sensitivity, shocks, outputs)
