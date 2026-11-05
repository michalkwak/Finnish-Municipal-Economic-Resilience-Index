"""Shock-based resilience: resistance to and recovery from national employment shocks.

Uses the sensitivity index of Martin, Sunley, Gardiner & Tyler (2016), "How regions
react to recessions", Regional Studies 50(4):

    resistance_i = (dE_i / E_i - dE_n / E_n) / |dE_n / E_n|    over peak -> trough
    recovery_i   = same formula                                over trough -> recovery end

where E is the number of employed persons and n the whole country. 0 means the
municipality moved exactly with the country; +1 means it lost nothing while the
country lost (resistance) or grew twice as fast as the country (recovery); -1 means
it fell twice as hard as the country.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from resilience.config import (
    MUNICIPALITY_CODE_PREFIX,
    NATIONAL_CODE,
    SHOCKS,
    SMALL_MUNICIPALITY_EMPLOYED,
    Shock,
)

QUADRANTS = {
    (True, True): "Resistant, strong recovery",
    (True, False): "Resistant, weak recovery",
    (False, True): "Vulnerable, strong recovery",
    (False, False): "Vulnerable, weak recovery",
}


def _relative_change(start: pd.Series | float, end: pd.Series | float):
    return end / start - 1


def martin_index(local_change: pd.Series, national_change: float) -> pd.Series:
    if not np.isfinite(national_change) or abs(national_change) < 1e-12:
        return pd.Series(np.nan, index=local_change.index)
    return (local_change - national_change) / abs(national_change)


def check_shock(national: pd.Series, shock: Shock) -> None:
    """The shock definition must match the data: employment falls from peak to trough."""
    missing = [y for y in (shock.peak, shock.trough, shock.recovery_end) if y not in national]
    if missing:
        raise ValueError(f"{shock.key}: no national employment data for {missing}")
    if not national[shock.trough] < national[shock.peak]:
        raise ValueError(
            f"{shock.key}: national employment did not fall from {shock.peak} "
            f"({national[shock.peak]:,.0f}) to {shock.trough} ({national[shock.trough]:,.0f})"
        )


def shock_resilience(employed: pd.DataFrame, shocks: Iterable[Shock] = SHOCKS) -> pd.DataFrame:
    """Resistance and recovery per municipality and shock.

    ``employed`` is the raw employed-persons table (``AreaCode, Area, Year, Employed``)
    including the national ``SSS`` row.
    """
    wide = employed.pivot_table(index="AreaCode", columns="Year", values="Employed")
    names = employed.drop_duplicates("AreaCode").set_index("AreaCode")["Area"]
    if NATIONAL_CODE not in wide.index:
        raise ValueError("Employed table has no national (SSS) row")
    national = wide.loc[NATIONAL_CODE]
    munis = wide[wide.index.str.startswith(MUNICIPALITY_CODE_PREFIX)]

    frames = []
    for shock in shocks:
        check_shock(national, shock)
        e_peak, e_trough, e_end = (munis[y] for y in (shock.peak, shock.trough, shock.recovery_end))
        n_res = _relative_change(national[shock.peak], national[shock.trough])
        n_rec = _relative_change(national[shock.trough], national[shock.recovery_end])
        local_res = _relative_change(e_peak, e_trough)
        local_rec = _relative_change(e_trough, e_end)

        f = pd.DataFrame(
            {
                "MunicipalityCode": munis.index,
                "Municipality": names.reindex(munis.index).to_numpy(),
                "Shock": shock.key,
                "ShockLabel": shock.label,
                "PeakYear": shock.peak,
                "TroughYear": shock.trough,
                "RecoveryEndYear": shock.recovery_end,
                "EmployedPeak": e_peak.to_numpy(),
                "EmployedTrough": e_trough.to_numpy(),
                "EmployedRecoveryEnd": e_end.to_numpy(),
                "ChangeShock": local_res.to_numpy(),
                "ChangeRecovery": local_rec.to_numpy(),
                "NationalChangeShock": n_res,
                "NationalChangeRecovery": n_rec,
                "Resistance": martin_index(local_res, n_res).to_numpy(),
                "Recovery": martin_index(local_rec, n_rec).to_numpy(),
            }
        )
        f["Small"] = f["EmployedPeak"] < SMALL_MUNICIPALITY_EMPLOYED
        valid = f[["Resistance", "Recovery"]].notna().all(axis=1)
        f["Type"] = pd.Series(pd.NA, index=f.index, dtype="string")
        f.loc[valid, "Type"] = [
            QUADRANTS[(bool(a > 0), bool(b > 0))]
            for a, b in zip(f.loc[valid, "Resistance"], f.loc[valid, "Recovery"], strict=True)
        ]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def shock_summary(resilience: pd.DataFrame) -> pd.DataFrame:
    """One row per municipality: averages across shocks and how often it was resistant."""
    g = resilience.groupby(["MunicipalityCode", "Municipality"])
    out = pd.DataFrame(
        {
            "ShocksMeasured": g["Resistance"].count(),
            "TimesResistant": g["Resistance"].agg(lambda s: int((s > 0).sum())),
            "TimesStrongRecovery": g["Recovery"].agg(lambda s: int((s > 0).sum())),
            "MeanResistance": g["Resistance"].mean(),
            "MeanRecovery": g["Recovery"].mean(),
            "EmployedLatestPeak": g["EmployedPeak"].last(),
        }
    ).reset_index()
    for shock, sub in resilience.groupby("Shock", sort=False):
        out = out.merge(
            sub[["MunicipalityCode", "Resistance", "Recovery"]].rename(
                columns={"Resistance": f"Resistance_{shock}", "Recovery": f"Recovery_{shock}"}
            ),
            on="MunicipalityCode",
            how="left",
        )
    return out.sort_values("MeanResistance", ascending=False).reset_index(drop=True)
