from __future__ import annotations

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


@pytest.fixture
def real_raw_dir():
    if not all((RAW_DIR / spec.filename).exists() for spec in TABLES.values()):
        pytest.skip("data/raw not populated; run `resilience fetch`")
    return RAW_DIR
