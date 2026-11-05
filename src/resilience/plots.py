"""Figures. Uses matplotlib's object API so it works headless (CI, servers)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def plot_municipalities(
    index_df: pd.DataFrame, municipalities: Sequence[str], out_path: Path
) -> Path:
    """Line chart of the index over time for the named municipalities, saved as an image."""
    known = set(index_df["Municipality"])
    unknown = [m for m in municipalities if m not in known]
    if unknown:
        raise ValueError(f"Unknown municipalities: {unknown}")

    fig = Figure(figsize=(10, 5), layout="constrained")
    ax = fig.subplots()
    for name in municipalities:
        rows = index_df[index_df["Municipality"] == name].sort_values("Year")
        ax.plot(rows["Year"], rows["EconomicResilienceIndex_100"], label=name, linewidth=2)

    title = (
        f"Economic Resilience Index – {municipalities[0]}"
        if len(municipalities) == 1
        else "Economic Resilience Index over time"
    )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Index (0–100, relative to all municipalities that year)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    if len(municipalities) > 1:
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return out_path
