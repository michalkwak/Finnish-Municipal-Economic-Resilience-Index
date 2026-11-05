import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

print("Current working directory:", os.getcwd())

# Loading data
def load_data(file_path):
    # files are comma-separated and have a title row + blank row, then header row (row 3)
    return pd.read_csv(
        file_path,
        encoding="latin1",
        sep=",",
        engine="python",
        skiprows=2,
        header=0
    )
BASE_DIR = r"C:\Users\kanci\OneDrive\Pulpit\hackathon"

gross_income_data = load_data(f"{BASE_DIR}\\gross_income.csv")
population_data = load_data(f"{BASE_DIR}\\population.csv")
unemployment_data = load_data(f"{BASE_DIR}\\unemployment_rate.csv")

# column cleanup
for df in (gross_income_data, population_data, unemployment_data):
    df.columns = df.columns.astype(str).str.strip().str.replace("\xa0", " ", regex=False)

print("Loaded samples:")
print("\nGross income head:\n", gross_income_data.head())
print("\nPopulation head:\n", population_data.head())
print("\nUnemployment head:\n", unemployment_data.head())


# wide -> long with year extraction
YEAR_RE = re.compile(r"(19|20)\d{2}")

def wide_to_long_extract_year(df, id_col, value_name, extra_keep_cols=None, filter_total_col=None):
    """
    Converts a wide table to long:
    - id_col is municipality column (e.g. 'Region' or 'Area')
    - value columns contain a 4-digit year in their name; we extract it
    - optional: keep extra columns or filter rows (e.g. Main type of activity == 'Total')
    """
    df = df.copy()

    # optional filtering (used for population: keep only Total)
    if filter_total_col is not None:
        col, wanted = filter_total_col
        df = df[df[col].astype(str).str.strip() == wanted]

    keep_cols = [id_col]
    if extra_keep_cols:
        keep_cols += extra_keep_cols

    # all columns that contain a year
    year_cols = [c for c in df.columns if YEAR_RE.search(str(c))]
    if not year_cols:
        raise ValueError(f"No year columns detected in {value_name}. Columns start: {list(df.columns)[:15]}")

    df = df[keep_cols + year_cols]

    long_df = df.melt(
        id_vars=keep_cols,
        var_name="YearRaw",
        value_name=value_name
    )

    # axtract numeric year from column names
    long_df["Year"] = long_df["YearRaw"].astype(str).str.extract(r"((?:19|20)\d{2})", expand=False)
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")

    # standardize municipality column name
    long_df = long_df.rename(columns={id_col: "Municipality"})

    # convert values to numeric 
    # treat '.' as missing
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")

    # drop bad rows
    long_df = long_df.dropna(subset=["Municipality", "Year", value_name])
    long_df["Municipality"] = long_df["Municipality"].astype(str).str.strip()
    long_df["Year"] = long_df["Year"].astype(int)

    return long_df.drop(columns=["YearRaw"])


# Build long tables

# Gross income file:
# header shows first column is "Region"
if "Region" not in gross_income_data.columns:
    raise KeyError(f"Expected 'Region' in gross_income.csv, got: {list(gross_income_data.columns)[:10]}")

gross_income_long = wide_to_long_extract_year(
    gross_income_data,
    id_col="Region",
    value_name="GrossIncome"
)

# Population file:
# header shows columns: Area, Main type of activity, then year columns
if "Area" not in population_data.columns or "Main type of activity" not in population_data.columns:
    raise KeyError(f"Expected 'Area' and 'Main type of activity' in population.csv. Got: {list(population_data.columns)[:10]}")

population_long = wide_to_long_extract_year(
    population_data,
    id_col="Area",
    value_name="Population",
    filter_total_col=("Main type of activity", "Total")
)

# Unemployment file:
# columns: Year, Information, then municipalities as columns
if "Year" not in unemployment_data.columns:
    raise KeyError(f"Expected 'Year' in unemployment_rate.csv, got: {list(unemployment_data.columns)[:10]}")

# keep only unemployment rate rows
unemp = unemployment_data.copy()
unemp = unemp[unemp["Information"].astype(str).str.contains("Unemployment rate", na=False)]

# melt municipalities columns
muni_cols = [c for c in unemp.columns if c not in ["Year", "Information"]]
unemployment_long = unemp.melt(
    id_vars=["Year"],
    value_vars=muni_cols,
    var_name="Municipality",
    value_name="UnemploymentRate"
)

unemployment_long["Year"] = pd.to_numeric(unemployment_long["Year"], errors="coerce").astype("Int64")
unemployment_long["UnemploymentRate"] = pd.to_numeric(unemployment_long["UnemploymentRate"], errors="coerce")
unemployment_long["Municipality"] = unemployment_long["Municipality"].astype(str).str.strip()
unemployment_long = unemployment_long.dropna(subset=["Year", "UnemploymentRate"])
unemployment_long["Year"] = unemployment_long["Year"].astype(int)

print("\nLong format checks:")
print(gross_income_long.head())
print(population_long.head())
print(unemployment_long.head())


# merge on Municipality + Year

merged = gross_income_long.merge(population_long, on=["Municipality", "Year"], how="inner")
merged = merged.merge(unemployment_long, on=["Municipality", "Year"], how="inner")

# drop national aggregate if you want only municipalities
merged = merged[merged["Municipality"].str.upper() != "WHOLE COUNTRY"]

print("\nMerged sample:\n", merged.head())
print("Merged rows:", len(merged), "| years:", merged["Year"].nunique(), "| municipalities:", merged["Municipality"].nunique())


# Economic Resilience Index (per-year normalization)

def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=series.index)
    return (s - mn) / (mx - mn)

merged["LogPopulation"] = np.log1p(merged["Population"])

merged["income_score"] = merged.groupby("Year")["GrossIncome"].transform(minmax)
merged["pop_score"] = merged.groupby("Year")["LogPopulation"].transform(minmax)

unemp_norm = merged.groupby("Year")["UnemploymentRate"].transform(minmax)
merged["employment_score"] = 1 - unemp_norm

merged["EconomicResilienceIndex"] = (
    0.45 * merged["employment_score"] +
    0.35 * merged["income_score"] +
    0.20 * merged["pop_score"]
)

merged["EconomicResilienceIndex_100"] = (merged["EconomicResilienceIndex"] * 100).round(1)

print("\nIndex sample:\n", merged[["Municipality", "Year", "EconomicResilienceIndex_100"]].head())


# save outputs
merged.to_csv("finland_economic_resilience_index_long.csv", index=False)

merged.pivot_table(
    index="Municipality",
    columns="Year",
    values="EconomicResilienceIndex_100"
).to_csv("finland_economic_resilience_index_wide.csv")

print("\nSaved:")
print("- finland_economic_resilience_index_long.csv")
print("- finland_economic_resilience_index_wide.csv")


def plot_city_resilience(df, city):
    city_df = df[df["Municipality"] == city].sort_values("Year")

    plt.figure(figsize=(8, 4))
    plt.plot(city_df["Year"], city_df["EconomicResilienceIndex_100"])
    plt.title(f"Economic Resilience Index – {city}")
    plt.xlabel("Year")
    plt.ylabel("Resilience Index (0–100)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_city_resilience(merged, "Helsinki")



def plot_multiple_cities(df, cities):
    plt.figure(figsize=(10, 5))

    for city in cities:
        city_df = df[df["Municipality"] == city].sort_values("Year")
        plt.plot(city_df["Year"], city_df["EconomicResilienceIndex_100"], label=city)

    plt.title("Economic Resilience Index Over Time")
    plt.xlabel("Year")
    plt.ylabel("Resilience Index (0–100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_multiple_cities(
    merged,
    ["Helsinki", "Tampere", "Turku", "Oulu"]
)
