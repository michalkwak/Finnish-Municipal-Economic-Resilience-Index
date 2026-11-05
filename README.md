# Finnish Municipal Economic Resilience Index

A Python data pipeline that merges three Statistics Finland open datasets (gross income,
population and unemployment rate) for all 308 Finnish municipalities from 1995 to 2024,
then computes a composite Economic Resilience Index (0–100) for each municipality and year.

![Index for the four largest cities](results/figures/resilience_helsinki_tampere_turku_oulu.png)

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # create .venv with locked dependencies
uv run resilience build        # data/raw -> results/
uv run resilience plot Helsinki Tampere Turku Oulu
```

The raw data is committed, so `build` works offline. To refresh it from the API:

```bash
uv run resilience fetch        # Statistics Finland PxWeb API -> data/raw/
uv run resilience build
```

Try different weights (they must sum to 1):

```bash
uv run resilience build --w-employment 0.4 --w-income 0.4 --w-population 0.2 --results-dir results/alt
```

## How the index works

Each year, three indicators are min-max scaled to 0–1 across all municipalities:

| Component | Indicator | Weight |
| --- | --- | --- |
| Employment score | 1 − scaled unemployment rate (18–64 year olds) | 45% |
| Income score | scaled mean gross income of household-dwelling units | 35% |
| Population score | scaled log(1 + population) | 20% |

The weighted sum is multiplied by 100. Scaling is done **within each year**, so the index
says how a municipality ranks against the others in that year. It does not measure absolute
change over time: a municipality's score can rise in a year when its income fell, as long as
it fell less than elsewhere.

## Data

| File in `data/raw/` | StatFin table | Variable |
| --- | --- | --- |
| `gross_income.csv` | [118w](https://pxdata.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__tjt/statfin_tjt_pxt_118w.px/) Income of household-dwelling units by region | 6. Gross income, mean (EUR) |
| `population.csv` | [115b](https://pxdata.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__tyokay/statfin_tyokay_pxt_115b.px/) Population by area, main type of activity, sex, age | Population 31 Dec (total) |
| `unemployment_rate.csv` | [115x](https://pxdata.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__tyokay/statfin_tyokay_pxt_115x.px/) Key indicators of employment statistics | Unemployment rate, 18–64, % |

Raw files are tidy (`AreaCode, Area, Year, <value>`). Municipalities are joined on their
official code (`KU091` = Helsinki), not their name. Statistics Finland reports every year on
the current (2025) municipal division, so past mergers are already harmonised.
`data/raw/sources.json` records when each table was fetched and last updated by the publisher.

Source: Statistics Finland, StatFin database, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Outputs (`results/`)

| File | Contents |
| --- | --- |
| `finland_economic_resilience_index_long.csv` | One row per municipality-year: inputs, component scores, index |
| `finland_economic_resilience_index_wide.csv` | Municipalities as rows, years as columns, index (0–100) as values |
| `merge_audit_dropped_rows.csv` | Municipality-years dropped because an indicator was missing, and which one |
| `build_summary.json` | Weights, coverage, dropped-row counts, data provenance |
| `figures/` | Charts from `resilience plot` |

Currently 13 rows are dropped, all for Sottunga (population ~90), where Statistics Finland
suppresses the unemployment rate in some years.

## Project layout

```text
src/resilience/
  config.py     paths, weights, StatFin table definitions
  fetch.py      PxWeb API client (json-stat2 -> tidy CSV)
  load.py       read and validate raw tables
  transform.py  filter to municipalities, merge, audit dropped rows
  index.py      min-max scaling and the weighted index
  validate.py   schema / range / uniqueness checks
  pipeline.py   end-to-end build, writes results/
  plots.py      figures (headless)
  cli.py        `resilience fetch | build | plot`
tests/          pytest suite
archive/        the original hackathon script, its inputs and its results
```

## Development

```bash
uv run pytest                    # all tests (includes live API tests)
uv run pytest -m "not network"   # offline only, as in CI
uv run ruff check . && uv run ruff format .
uv run pre-commit install        # lint and format on every commit
```

CI (GitHub Actions) runs lint and the offline tests on Python 3.11 and 3.13, and checks that
`results/` can be rebuilt byte-for-byte from the committed raw data.
