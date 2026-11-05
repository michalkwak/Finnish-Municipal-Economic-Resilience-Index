"""End-to-end build: raw tables -> merged indicators -> index -> result files."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from resilience.config import (
    CORE_TABLES,
    DEFAULT_VARIANT,
    RAW_DIR,
    RESULTS_DIR,
    TABLES,
    VARIANTS,
    VARIANTS_DIR,
    IndexSpec,
)
from resilience.features import add_real_income, population_growth
from resilience.index import compute_index, to_wide
from resilience.load import load_all, load_indicator
from resilience.transform import MergeResult, merge_indicators, municipalities_only
from resilience.validate import validate_index

log = logging.getLogger(__name__)

LONG_FILE = "finland_economic_resilience_index_long.csv"
WIDE_FILE = "finland_economic_resilience_index_wide.csv"
DROPPED_FILE = "merge_audit_dropped_rows.csv"
SUMMARY_FILE = "build_summary.json"


@dataclass
class BuildResult:
    index: pd.DataFrame
    dropped: pd.DataFrame
    outputs: list[Path]


#: Significant digits for floats in CSV output. Full repr (17 digits) exposes
#: last-ULP differences in numpy's transcendental functions (e.g. ``log1p``)
#: between platforms, which breaks byte-for-byte reproducibility in CI.
CSV_FLOAT_FORMAT = "%.12g"


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=index,
        encoding="utf-8",
        lineterminator="\n",
        float_format=CSV_FLOAT_FORMAT,
    )
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n")
    return path


def default_results_dir(spec: IndexSpec) -> Path:
    return RESULTS_DIR if spec.name == DEFAULT_VARIANT else VARIANTS_DIR / spec.name


def merged_indicators(raw_dir: Path, spec: IndexSpec) -> tuple[pd.DataFrame, MergeResult]:
    """Load the tables a variant needs and merge them on municipality-year."""
    raw = load_all(raw_dir, CORE_TABLES)
    frames = {key: municipalities_only(df) for key, df in raw.items()}
    value_cols = {key: TABLES[key].value_col for key in frames}

    if spec.demographic == "dependency_ratio":
        frames["dependency_ratio"] = municipalities_only(
            load_indicator(TABLES["dependency_ratio"], raw_dir)
        )
        value_cols["dependency_ratio"] = TABLES["dependency_ratio"].value_col
    elif spec.demographic == "growth_5y":
        growth = population_growth(frames["population"])
        frames["population_growth"] = growth
        value_cols["population_growth"] = growth.columns[-1]

    merge = merge_indicators(frames, value_cols)
    merged = merge.merged
    if spec.real_income:
        merged = add_real_income(merged, load_indicator(TABLES["cpi"], raw_dir))
    return merged, merge


def build_index(
    raw_dir: Path = RAW_DIR, spec: IndexSpec | None = None
) -> tuple[pd.DataFrame, MergeResult]:
    """Run the pipeline in memory. Returns (index frame, merge result)."""
    spec = spec or VARIANTS[DEFAULT_VARIANT]
    merged, merge = merged_indicators(raw_dir, spec)
    index_df = compute_index(merged, spec)
    validate_index(index_df)
    return index_df, merge


def spec_record(spec: IndexSpec) -> dict[str, Any]:
    record = dataclasses.asdict(spec)
    record["weights"] = dataclasses.asdict(spec.weights)
    return record


def build(
    raw_dir: Path = RAW_DIR,
    results_dir: Path | None = None,
    spec: IndexSpec | None = None,
) -> BuildResult:
    """Run the pipeline and write the result files.

    ``results_dir`` defaults to ``results/`` for the default variant (v2) and
    ``results/variants/<name>/`` for other variants.
    """
    spec = spec or VARIANTS[DEFAULT_VARIANT]
    results_dir = results_dir or default_results_dir(spec)
    index_df, merge = build_index(raw_dir, spec)

    sources_path = raw_dir / "sources.json"
    sources = json.loads(sources_path.read_text("utf-8")) if sources_path.exists() else {}
    summary = {
        "variant": spec_record(spec),
        "years": list(merge.common_years),
        "municipalities": int(index_df["MunicipalityCode"].nunique()),
        "rows": len(index_df),
        "dropped_rows": len(merge.dropped),
        "dropped_by_missing_indicator": merge.dropped["missing"].value_counts().to_dict(),
        "name_conflicts": merge.name_conflicts.to_dict("records"),
        "real_income_base_year": index_df.attrs.get("real_income_base_year"),
        "sources": {
            k: {
                "table": v.get("table"),
                "publisher_updated": v.get("publisher_updated"),
                "fetched_at": v.get("fetched_at"),
            }
            for k, v in sources.items()
        },
    }

    outputs = [
        write_csv(index_df, results_dir / LONG_FILE),
        write_csv(to_wide(index_df), results_dir / WIDE_FILE, index=True),
        write_csv(merge.dropped, results_dir / DROPPED_FILE),
        write_json(summary, results_dir / SUMMARY_FILE),
    ]
    for p in outputs:
        log.info("Wrote %s", p)
    return BuildResult(index_df, merge.dropped, outputs)
