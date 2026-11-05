# v0 – original hackathon version (archived)

Snapshot of the project before the 2026 restructure. Kept for reference only and not
maintained; the live code is in `src/resilience/`.

| File | What it is |
| --- | --- |
| `main.py` | Original single-file script. It no longer runs as is: it reads from a hardcoded local path (`C:\Users\kanci\OneDrive\Pulpit\hackathon`) and expects an older layout of the unemployment export. |
| `inputs/*.csv` | Manual exports from the StatFin web interface (Latin-1, two title rows, years as columns). Files were renamed from `gross income.csv` and `unemployment rates.csv`. |
| `finland_economic_resilience_index_long.csv` / `_wide.csv` | Results produced by `main.py`. |

## How v1 compares

The restructured pipeline fetches the same three tables from the Statistics Finland
API (with official municipality codes) instead of using manual exports. Its `baseline`
variant (`results/variants/baseline/`) reproduces these results exactly: all 9,227
municipality-year rows match on gross income, population, unemployment rate and the
index value (checked 2026-11-05). Since v0.4 the default index is `v2`, which uses a
different method; see the main README.
