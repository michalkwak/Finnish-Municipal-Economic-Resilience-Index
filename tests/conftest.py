from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilience.config import RAW_DIR, TABLES


def _raw(key: str, rows: list[tuple[str, str, int, float | None]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["AreaCode", "Area", "Year", TABLES[key].value_col])
    df["AreaCode"] = df["AreaCode"].astype("string")
    df["Area"] = df["Area"].astype("string")
    return df


@pytest.fixture
def tiny_raw_dir(tmp_path):
    """Three municipalities + aggregates over 2 years, written in the raw CSV format."""
    areas = [
        ("SSS", "WHOLE COUNTRY"),
        ("MK01", "Uusimaa"),
        ("KU091", "Helsinki"),
        ("KU837", "Tampere"),
        ("KU766", "Sottunga"),
    ]
    income = {"KU091": 70000, "KU837": 60000, "KU766": 40000, "SSS": 63000, "MK01": 68000}
    pop = {"KU091": 660000, "KU837": 250000, "KU766": 90, "SSS": 5600000, "MK01": 1700000}
    unemp = {"KU091": 9.0, "KU837": 11.0, "KU766": None, "SSS": 10.0, "MK01": 8.5}

    frames = {
        "gross_income": _raw(
            "gross_income",
            [(c, n, y, income[c] + 1000 * (y - 2023)) for c, n in areas for y in (2023, 2024)],
        ),
        "population": _raw(
            "population", [(c, n, y, pop[c]) for c, n in areas for y in (2022, 2023, 2024)]
        ),
        "unemployment_rate": _raw(
            "unemployment_rate", [(c, n, y, unemp[c]) for c, n in areas for y in (2023, 2024)]
        ),
    }
    for key, df in frames.items():
        df.to_csv(tmp_path / TABLES[key].filename, index=False, encoding="utf-8")
    return tmp_path


# National employed persons 1987-2024 (table 115b), used to give the synthetic panel
# realistic shocks.
NATIONAL_EMPLOYED = [
    2319527, 2353195, 2373747, 2332282, 2169147, 2014430, 1877721, 1917051, 1932752, 1957144,
    2037997, 2132704, 2173885, 2228557, 2235317, 2242303, 2245780, 2262359, 2265211, 2313788,
    2369066, 2377181, 2289975, 2325679, 2354422, 2339904, 2301751, 2274076, 2256459, 2275679,
    2327730, 2373668, 2373526, 2284673, 2377126, 2423548, 2417365, 2423433,
]  # fmt: skip
PANEL_YEARS = list(range(1987, 2025))


@pytest.fixture
def panel_raw_dir(tmp_path):
    """Six synthetic municipalities, 1987-2024, every table the pipeline and analysis use."""
    rng = np.random.default_rng(42)
    munis = [
        ("KU091", "Helsinki"),
        ("KU837", "Tampere"),
        ("KU853", "Turku"),
        ("KU564", "Oulu"),
        ("KU001", "Testby"),
        ("KU002", "Smallville"),
    ]
    shares = np.array([0.30, 0.25, 0.2, 0.15, 0.07, 0.03])
    nat = np.array(NATIONAL_EMPLOYED, dtype=float)
    # Each municipality follows the nation with its own sensitivity to swings.
    beta = np.array([0.6, 1.0, 1.3, 0.9, 1.6, 0.4])
    swing = nat / nat.mean() - 1
    employed = shares[:, None] * nat.mean() * (1 + beta[:, None] * swing[None, :])
    employed *= 1 + rng.normal(0, 0.003, employed.shape)
    employed[:, :] *= nat / employed.sum(axis=0)  # municipalities sum to the nation

    rows: dict[str, list] = {k: [] for k in TABLES}
    for i, (code, name) in enumerate(munis):
        for j, y in enumerate(PANEL_YEARS):
            pop = employed[i, j] * 2.2 * (1 + 0.004 * (j - 10) * (i - 2))
            rows["population"].append((code, name, y, round(pop)))
            rows["employed"].append((code, name, y, round(employed[i, j])))
            rows["unemployment_rate"].append(
                (code, name, y, round(8 + 6 * (beta[i] - 1) - 20 * swing[j] + i * 0.3, 1))
            )
            rows["dependency_ratio"].append((code, name, y, round(110 + 15 * beta[i] - i, 0)))
            if y >= 1995:
                income = 40000 + 6000 * (5 - i) + 1500 * (y - 1995) + rng.normal(0, 300)
                rows["gross_income"].append((code, name, y, round(income)))
    for j, y in enumerate(PANEL_YEARS):
        rows["employed"].append(("SSS", "WHOLE COUNTRY", y, NATIONAL_EMPLOYED[j]))
        if y >= 1995:
            rows["cpi"].append(("SSS", "WHOLE COUNTRY", y, round(100 * 1.018 ** (y - 1995), 1)))

    for key, r in rows.items():
        _raw(key, r).to_csv(tmp_path / TABLES[key].filename, index=False, encoding="utf-8")
    return tmp_path


@pytest.fixture
def real_raw_dir():
    if not all((RAW_DIR / spec.filename).exists() for spec in TABLES.values()):
        pytest.skip("data/raw not populated; run `resilience fetch`")
    return RAW_DIR
