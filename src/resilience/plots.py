"""Figures. Uses matplotlib's object API so it works headless (CI, servers).

Colours follow one fixed palette: categorical slots in a fixed order for named
municipalities, neutral grey for context marks, one blue ramp for magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
CONTEXT = "#b9b8b2"  # neutral grey for non-highlighted marks
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

HIGHLIGHT = ["Helsinki", "Tampere", "Turku", "Oulu"]


def _figure(
    figsize: tuple[float, float], ncols: int = 1, nrows: int = 1
) -> tuple[Figure, np.ndarray]:
    fig = Figure(figsize=figsize, layout="constrained", facecolor=SURFACE)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CONTEXT)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
        ax.title.set_color(INK)
    return fig, axes


LABEL_BOX = {"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85}
LEADER = {"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 3}


def _spread(values: Sequence[float], min_gap: float) -> np.ndarray:
    """Nudge label positions apart so neighbours are at least ``min_gap`` apart.

    Keeps the original order and the group's centre roughly in place.
    """
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    pos = v[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    pos -= (pos.mean() - v[order].mean()) if len(pos) else 0  # re-centre the group
    for i in range(1, len(pos)):  # re-centring can't create overlaps, but keep it exact
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _save(fig: Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    return out_path


def plot_municipalities(
    index_df: pd.DataFrame,
    municipalities: Sequence[str],
    out_path: Path,
    subtitle: str = "relative to all municipalities that year",
    title: str | None = None,
) -> Path:
    """Line chart of the index over time for the named municipalities, saved as an image."""
    known = set(index_df["Municipality"])
    unknown = [m for m in municipalities if m not in known]
    if unknown:
        raise ValueError(f"Unknown municipalities: {unknown}")
    if len(municipalities) > len(CATEGORICAL):
        raise ValueError(f"At most {len(CATEGORICAL)} municipalities per chart")

    fig, axes = _figure((10, 5))
    ax = axes[0, 0]
    ends = []
    for name, color in zip(municipalities, CATEGORICAL, strict=False):
        rows = index_df[index_df["Municipality"] == name].sort_values("Year")
        ax.plot(
            rows["Year"], rows["EconomicResilienceIndex_100"], color=color, linewidth=2, label=name
        )
        last = rows.iloc[-1]
        ends.append((name, last["Year"], last["EconomicResilienceIndex_100"]))
    if len(municipalities) > 1:  # direct labels at the line ends, nudged apart
        label_y = _spread([y for _, _, y in ends], min_gap=3.5)
        for (name, x, y), ly in zip(ends, label_y, strict=True):
            ax.annotate(
                name,
                (x, y),
                xytext=(x + 0.6, ly),
                textcoords="data",
                va="center",
                fontsize=9,
                color=INK_SECONDARY,
                arrowprops=LEADER if abs(ly - y) > 1 else None,
            )

    title = title or (
        f"Economic Resilience Index – {municipalities[0]}"
        if len(municipalities) == 1
        else "Economic Resilience Index over time"
    )
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index (0–100, {subtitle})")
    ax.set_ylim(0, 100)
    if len(municipalities) > 1:
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left", ncols=4)
        ax.margins(x=0.08)
    return _save(fig, out_path)


def plot_rank_uncertainty(table: pd.DataFrame, out_path: Path, title: str) -> Path:
    """Each municipality's base rank (dot) and 90% rank interval under random weights."""
    t = table.sort_values("BaseRank")
    fig, axes = _figure((11, 5))
    ax = axes[0, 0]
    x = t["BaseRank"].to_numpy()
    ax.vlines(
        x,
        t["GlobalRankP05"],
        t["GlobalRankP95"],
        color=GRID,
        linewidth=2,
        label="90% interval, any weights",
    )
    ax.vlines(
        x,
        t["LocalRankP05"],
        t["LocalRankP95"],
        color="#86b6ef",
        linewidth=2,
        label="90% interval, weights near the chosen ones",
    )
    ax.scatter(x, x, s=8, color=CATEGORICAL[0], zorder=3, label="rank with chosen weights")
    n = len(t)
    marks = [
        (name, int(t.loc[t["Municipality"] == name, "BaseRank"].iloc[0]))
        for name in HIGHLIGHT
        if (t["Municipality"] == name).any()
    ]
    # Labels sit below the diagonal, spread apart horizontally, with leaders.
    label_x = _spread([r for _, r in marks], min_gap=n * 0.07)
    for (name, r), lx in zip(marks, label_x, strict=True):
        ax.annotate(
            name,
            (r, r),
            xytext=(lx, min(r + n * 0.22, n * 0.95)),
            textcoords="data",
            ha="center",
            fontsize=8,
            color=INK,
            bbox=LABEL_BOX,
            arrowprops=LEADER,
            zorder=5,
        )
    ax.set_xlim(0, n + 1)
    ax.set_ylim(n + 1, 0)  # rank 1 at the top
    ax.set_xlabel("Rank with chosen weights (1 = most resilient)")
    ax.set_ylabel("Rank under alternative weights")
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower left", fontsize=9)
    return _save(fig, out_path)


def plot_shock_quadrants(resilience: pd.DataFrame, out_path: Path) -> Path:
    """Resistance vs recovery per shock: one panel each, big cities labelled."""
    shocks = list(dict.fromkeys(resilience["Shock"]))
    ncols = 2
    nrows = int(np.ceil(len(shocks) / ncols))
    fig, axes = _figure((11, 4.6 * nrows), ncols=ncols, nrows=nrows)
    # Common, outlier-robust axis limits so panels are comparable.
    vals = resilience.loc[~resilience["Small"], ["Resistance", "Recovery"]].to_numpy().ravel()
    lim = float(np.nanpercentile(np.abs(vals), 98)) * 1.1 or 1.0

    for ax, shock in zip(axes.flat, shocks, strict=False):
        sub = resilience[resilience["Shock"] == shock]
        big = sub[~sub["Small"]]
        small = sub[sub["Small"]]
        ax.scatter(
            small["Resistance"].clip(-lim, lim),
            small["Recovery"].clip(-lim, lim),
            s=10,
            facecolors="none",
            edgecolors=CONTEXT,
            linewidths=0.8,
            label="< 1,000 employed (noisy)",
        )
        ax.scatter(
            big["Resistance"].clip(-lim, lim),
            big["Recovery"].clip(-lim, lim),
            s=12,
            color=CONTEXT,
            label="other municipalities",
        )
        marks = []
        for name, color in zip(HIGHLIGHT, CATEGORICAL, strict=False):
            row = sub[sub["Municipality"] == name]
            if row.empty:
                continue
            xr, yr = row["Resistance"].iloc[0], row["Recovery"].iloc[0]
            ax.scatter(
                [xr],
                [yr],
                s=60,
                color=color,
                edgecolors=SURFACE,
                linewidths=2,
                zorder=4,
                label=name,
            )
            marks.append((name, xr, yr))
        # Labels to the right of the highlighted points, spread apart vertically.
        label_y = _spread([y for _, _, y in marks], min_gap=lim * 0.13)
        for (name, xr, yr), ly in zip(marks, label_y, strict=True):
            ax.annotate(
                name,
                (xr, yr),
                xytext=(xr + lim * 0.12, ly),
                textcoords="data",
                va="center",
                fontsize=8,
                color=INK,
                bbox=LABEL_BOX,
                arrowprops=LEADER,
                zorder=5,
            )
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        label = sub["ShockLabel"].iloc[0]
        years = (
            f"{sub['PeakYear'].iloc[0]}–{sub['TroughYear'].iloc[0]}–"
            f"{sub['RecoveryEndYear'].iloc[0]}"
        )
        ax.set_title(f"{label} ({years})", loc="left", fontsize=11)
        ax.set_xlabel("Resistance (0 = fell with the country)")
        ax.set_ylabel("Recovery (0 = grew with the country)")
        for tx, ty, text in (
            (0.98, 0.98, "resistant, strong recovery"),
            (0.02, 0.02, "vulnerable, weak recovery"),
        ):
            ax.text(
                tx,
                ty,
                text,
                transform=ax.transAxes,
                ha="right" if tx > 0.5 else "left",
                va="top" if ty > 0.5 else "bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )
    for ax in list(axes.flat)[len(shocks) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=6,
        frameon=False,
        labelcolor=INK_SECONDARY,
        fontsize=9,
    )
    fig.suptitle(
        "Employment resistance and recovery in national shocks",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    return _save(fig, out_path)


def plot_variant_correlation(corr: pd.DataFrame, out_path: Path, year: int) -> Path:
    """Heatmap of Spearman rank correlations between index variants."""
    fig, axes = _figure((7, 6))
    ax = axes[0, 0]
    ax.grid(False)
    values = corr.to_numpy()
    lo = float(np.floor(values.min() * 10) / 10)
    im = ax.imshow(values, cmap=BLUE_RAMP, vmin=lo, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = values[i, j]
            dark = (v - lo) / (1 - lo) > 0.4 if lo < 1 else True
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=SURFACE if dark else INK,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.set_title(
        f"Rank agreement between index variants ({year}, Spearman)", loc="left", fontsize=11
    )
    return _save(fig, out_path)
