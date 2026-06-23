# Finnish Municipal Economic Resilience Index

A Python data pipeline that merges three Finnish open datasets - gross income, population, 
and unemployment rate - across all Finnish municipalities and years, then computes a 
composite Economic Resilience Index (0-100) for each municipality per year.

## How the index works

Each year, three indicators are normalised to 0–1 using min-max scaling:
- **Employment score** (1 - normalised unemployment rate) - weight 45%
- **Income score** (normalised gross income) - weight 35%
- **Population score** (normalised log population) - weight 20%

The weighted sum is scaled to 0–100.

## Output

- `finland_economic_resilience_index_long.csv` - one row per municipality per year
- `finland_economic_resilience_index_wide.csv` - municipalities as rows, years as columns

## Requirements

pandas, numpy, matplotlib
