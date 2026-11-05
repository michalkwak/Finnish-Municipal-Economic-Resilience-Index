"""Command line interface: ``resilience fetch | build | analyze | plot | variants``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from resilience.config import (
    ANALYSIS_DIR,
    DEFAULT_VARIANT,
    FIGURES_DIR,
    RAW_DIR,
    TABLES,
    VARIANTS,
    Weights,
)
from resilience.fetch import fetch_all
from resilience.pipeline import LONG_FILE, build, default_results_dir
from resilience.plots import HIGHLIGHT, plot_municipalities


def _slug(names: Sequence[str]) -> str:
    return "_".join(re.sub(r"\W+", "-", n.lower()).strip("-") for n in names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resilience", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download raw tables from Statistics Finland")
    p_fetch.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p_fetch.add_argument(
        "--table", choices=list(TABLES), action="append", help="only this table (repeatable)"
    )

    p_build = sub.add_parser("build", help="compute one index variant from the raw tables")
    p_build.add_argument("--variant", choices=list(VARIANTS), default=DEFAULT_VARIANT)
    p_build.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p_build.add_argument(
        "--results-dir",
        type=Path,
        help=f"default: results/ ({DEFAULT_VARIANT}) or results/variants/<name>",
    )
    p_build.add_argument("--w-employment", type=float, help="override the variant's weight")
    p_build.add_argument("--w-income", type=float)
    p_build.add_argument("--w-demographic", type=float)

    p_analyze = sub.add_parser(
        "analyze", help="build all variants; weight sensitivity; shock resilience"
    )
    p_analyze.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p_analyze.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    p_analyze.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p_analyze.add_argument("--variants-dir", type=Path, help="default: results/variants/")
    p_analyze.add_argument("--year", type=int, help="year to rank (default: latest)")
    p_analyze.add_argument("--draws", type=int, default=2000, help="random weight draws")
    p_analyze.add_argument("--seed", type=int, default=0)

    sub.add_parser("variants", help="list the index variants")

    p_plot = sub.add_parser("plot", help="plot the index for some municipalities")
    p_plot.add_argument("municipalities", nargs="*", default=HIGHLIGHT)
    p_plot.add_argument("--variant", choices=list(VARIANTS), default=DEFAULT_VARIANT)
    p_plot.add_argument("--results-dir", type=Path, help="where the variant's results are")
    p_plot.add_argument("--out", type=Path, help=f"image path (default: {FIGURES_DIR}/...)")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if args.command == "fetch":
        fetch_all(args.raw_dir, args.table)

    elif args.command == "build":
        spec = VARIANTS[args.variant]
        overrides = (args.w_employment, args.w_income, args.w_demographic)
        if any(v is not None for v in overrides):
            base = spec.weights.as_tuple()
            weights = Weights(
                *(o if o is not None else b for o, b in zip(overrides, base, strict=True))
            )
            spec = dataclasses.replace(spec, weights=weights)
        result = build(args.raw_dir, args.results_dir, spec)
        idx = result.index
        print(
            f"Built {spec.name}: {idx['MunicipalityCode'].nunique()} municipalities, "
            f"{idx['Year'].min()}-{idx['Year'].max()}, {len(idx)} rows "
            f"({len(result.dropped)} incomplete rows dropped, see merge audit)"
        )

    elif args.command == "analyze":
        from resilience.analysis import run_analysis

        res = run_analysis(
            args.raw_dir,
            args.analysis_dir,
            args.figures_dir,
            args.year,
            args.draws,
            args.seed,
            variants_root=args.variants_dir,
        )
        print(f"Analysis written to {args.analysis_dir} ({len(res.outputs)} files)")
        print("Rank correlation with baseline:", json.dumps(res.variant_corr["baseline"].to_dict()))

    elif args.command == "variants":
        for name, spec in VARIANTS.items():
            print(f"{name:12s} {spec.description}")

    elif args.command == "plot":
        spec = VARIANTS[args.variant]
        long_path = (args.results_dir or default_results_dir(spec)) / LONG_FILE
        if not long_path.exists():
            parser.error(f"{long_path} not found; run `resilience build --variant {spec.name}`")
        index_df = pd.read_csv(long_path, encoding="utf-8")
        suffix = "" if spec.name == DEFAULT_VARIANT else f"{spec.name}_"
        out = args.out or FIGURES_DIR / f"resilience_{suffix}{_slug(args.municipalities)}.png"
        subtitle = (
            "comparable across years"
            if spec.scope == "pooled"
            else "relative to all municipalities that year"
        )
        saved = plot_municipalities(index_df, args.municipalities, out, subtitle=subtitle)
        print(f"Saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
