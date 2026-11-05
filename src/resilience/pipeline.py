"""End-to-end build: raw tables -> merged indicators -> index -> result files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from resilience.config import RAW_DIR, RESULTS_DIR, TABLES, Weights
from resilience.index import compute_index, to_wide
from resilience.load import load_all
from resilience.transform import merge_indicators, municipalities_only
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


def build_index(raw_dir: Path = RAW_DIR, weights: Weights | None = None):
    """Run the pipeline in memory. Returns (index frame, merge result)."""
    raw = load_all(raw_dir)
    frames = {key: municipalities_only(df) for key, df in raw.items()}
    value_cols = {key: TABLES[key].value_col for key in frames}
    merge = merge_indicators(frames, value_cols)
    index_df = compute_index(merge.merged, weights)
    validate_index(index_df)
    return index_df, merge


def build(
    raw_dir: Path = RAW_DIR, results_dir: Path = RESULTS_DIR, weights: Weights | None = None
) -> BuildResult:
    """Run the pipeline and write the result files to ``results_dir``."""
    weights = weights or Weights()
    index_df, merge = build_index(raw_dir, weights)

    results_dir.mkdir(parents=True, exist_ok=True)
    long_path = results_dir / LONG_FILE
    wide_path = results_dir / WIDE_FILE
    dropped_path = results_dir / DROPPED_FILE
    summary_path = results_dir / SUMMARY_FILE

    index_df.to_csv(long_path, index=False, encoding="utf-8", lineterminator="\n")
    to_wide(index_df).to_csv(wide_path, encoding="utf-8", lineterminator="\n")
    merge.dropped.to_csv(dropped_path, index=False, encoding="utf-8", lineterminator="\n")

    sources_path = raw_dir / "sources.json"
    sources = json.loads(sources_path.read_text("utf-8")) if sources_path.exists() else {}
    summary = {
        "weights": {
            "employment": weights.employment,
            "income": weights.income,
            "population": weights.population,
        },
        "years": list(merge.common_years),
        "municipalities": int(index_df["MunicipalityCode"].nunique()),
        "rows": len(index_df),
        "dropped_rows": len(merge.dropped),
        "dropped_by_missing_indicator": merge.dropped["missing"].value_counts().to_dict(),
        "name_conflicts": merge.name_conflicts.to_dict("records"),
        "sources": {
            k: {
                "table": v.get("table"),
                "publisher_updated": v.get("publisher_updated"),
                "fetched_at": v.get("fetched_at"),
            }
            for k, v in sources.items()
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
    )

    outputs = [long_path, wide_path, dropped_path, summary_path]
    for p in outputs:
        log.info("Wrote %s", p)
    return BuildResult(index_df, merge.dropped, outputs)
