"""Command line interface: ``resilience fetch | build | plot``."""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from resilience.config import FIGURES_DIR, RAW_DIR, RESULTS_DIR, TABLES, Weights
from resilience.fetch import fetch_all
from resilience.pipeline import LONG_FILE, build
from resilience.plots import plot_municipalities

DEFAULT_MUNICIPALITIES = ["Helsinki", "Tampere", "Turku", "Oulu"]


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

    p_build = sub.add_parser("build", help="compute the index from the raw tables")
    p_build.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p_build.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    defaults = Weights()
    p_build.add_argument("--w-employment", type=float, default=defaults.employment)
    p_build.add_argument("--w-income", type=float, default=defaults.income)
    p_build.add_argument("--w-population", type=float, default=defaults.population)

    p_plot = sub.add_parser("plot", help="plot the index for some municipalities")
    p_plot.add_argument("municipalities", nargs="*", default=DEFAULT_MUNICIPALITIES)
    p_plot.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p_plot.add_argument("--out", type=Path, help=f"image path (default: {FIGURES_DIR}/...)")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if args.command == "fetch":
        fetch_all(args.raw_dir, args.table)
    elif args.command == "build":
        weights = Weights(args.w_employment, args.w_income, args.w_population)
        result = build(args.raw_dir, args.results_dir, weights)
        idx = result.index
        print(
            f"Built index: {idx['MunicipalityCode'].nunique()} municipalities, "
            f"{idx['Year'].min()}-{idx['Year'].max()}, {len(idx)} rows "
            f"({len(result.dropped)} incomplete rows dropped, see merge audit)"
        )
    elif args.command == "plot":
        long_path = args.results_dir / LONG_FILE
        if not long_path.exists():
            parser.error(f"{long_path} not found; run `resilience build` first")
        index_df = pd.read_csv(long_path, encoding="utf-8")
        out = args.out or FIGURES_DIR / f"resilience_{_slug(args.municipalities)}.png"
        print(f"Saved {plot_municipalities(index_df, args.municipalities, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
