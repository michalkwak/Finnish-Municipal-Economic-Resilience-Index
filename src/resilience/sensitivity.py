"""How much do municipality rankings depend on the choice of weights?

Follows the uncertainty-analysis step of the OECD/JRC *Handbook on Constructing
Composite Indicators* (2008): re-weight the component scores many times and look
at the spread of each municipality's rank.

* local  - weights drawn from a Dirichlet centred on the chosen weights
           (small, plausible disagreements about the weights)
* global - weights drawn uniformly from the whole simplex (any weighting at all)
* equal  - 1/3 each
* pca    - squared loadings of the first principal component of the scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from resilience.config import IndexSpec, Weights
from resilience.index import COMPONENT_SCORES, index_from_scores

LOCAL_CONCENTRATION = 30.0  # Dirichlet alpha = concentration * base weights


def ranks_desc(values: np.ndarray) -> np.ndarray:
    """Rank along axis 0, 1 = highest value. Works on (n,) or (n, k) arrays."""
    order = np.argsort(-values, axis=0, kind="stable")
    ranks = np.empty_like(order)
    if values.ndim == 1:
        ranks[order] = np.arange(1, len(values) + 1)
    else:
        np.put_along_axis(ranks, order, np.arange(1, values.shape[0] + 1)[:, None], axis=0)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length vectors (no SciPy needed)."""
    return float(np.corrcoef(ranks_desc(a), ranks_desc(b))[0, 1])


def pca_weights(scores: np.ndarray) -> Weights:
    """Weights from the squared loadings of the first principal component.

    Columns are standardised first. Squared loadings of a unit eigenvector sum to 1,
    so they can be used directly as weights.
    """
    std = scores.std(axis=0)
    z = (scores - scores.mean(axis=0)) / np.where(std == 0, 1, std)
    eigvals, eigvecs = np.linalg.eigh(np.cov(z, rowvar=False))
    w = eigvecs[:, np.argmax(eigvals)] ** 2
    w = w / w.sum()
    return Weights(*(float(x) for x in w))


@dataclass
class SensitivityResult:
    table: pd.DataFrame
    """One row per municipality: ranks under each weighting scheme and their spread."""
    summary: dict[str, Any]


def weight_sensitivity(
    index_df: pd.DataFrame,
    spec: IndexSpec | None = None,
    year: int | None = None,
    n_draws: int = 2000,
    seed: int = 0,
) -> SensitivityResult:
    """Rank stability of one year's index under alternative weights."""
    spec = spec or IndexSpec()
    year = int(index_df["Year"].max()) if year is None else year
    frame = index_df.loc[index_df["Year"] == year].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No index rows for {year}")
    scores = frame[COMPONENT_SCORES].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    base_w = spec.weights
    pca_w = pca_weights(index_df[COMPONENT_SCORES].to_numpy(dtype=float))
    base = index_from_scores(scores, base_w)
    base_rank = ranks_desc(base)

    def draws(alpha: np.ndarray) -> np.ndarray:
        weights = rng.dirichlet(alpha, size=n_draws)  # (k, 3)
        return ranks_desc(index_from_scores(scores, weights.T))  # (n, k)

    local = draws(LOCAL_CONCENTRATION * np.asarray(base_w.as_tuple()))
    glob = draws(np.ones(3))

    table = frame[["MunicipalityCode", "Municipality"]].copy()
    table["BaseRank"] = base_rank
    table["EqualWeightsRank"] = ranks_desc(index_from_scores(scores, Weights(1 / 3, 1 / 3, 1 / 3)))
    table["PcaWeightsRank"] = ranks_desc(index_from_scores(scores, pca_w))
    for name, r in (("Local", local), ("Global", glob)):
        p05, p50, p95 = np.percentile(r, [5, 50, 95], axis=1)
        table[f"{name}MedianRank"] = p50
        table[f"{name}RankP05"] = p05
        table[f"{name}RankP95"] = p95
        table[f"{name}Range90"] = p95 - p05
    table["ShareTop10Local"] = (local <= 10).mean(axis=1)
    table = table.sort_values("BaseRank").reset_index(drop=True)

    def rank_corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
        # Pearson correlation of ranks = Spearman correlation.
        return round(float(np.corrcoef(a, b)[0, 1]), 3)

    def median_rank_corr(r: np.ndarray) -> float:
        return round(float(np.median([rank_corr(base_rank, r[:, j]) for j in range(n_draws)])), 3)

    summary = {
        "variant": spec.name,
        "year": year,
        "municipalities": len(frame),
        "draws": n_draws,
        "seed": seed,
        "base_weights": dict(zip(COMPONENT_SCORES, base_w.as_tuple(), strict=True)),
        "pca_weights": {
            k: round(v, 3) for k, v in zip(COMPONENT_SCORES, pca_w.as_tuple(), strict=True)
        },
        "spearman_base_vs_equal": rank_corr(table["BaseRank"], table["EqualWeightsRank"]),
        "spearman_base_vs_pca": rank_corr(table["BaseRank"], table["PcaWeightsRank"]),
        "median_spearman_local": median_rank_corr(local),
        "median_spearman_global": median_rank_corr(glob),
        "median_range90_local": float(table["LocalRange90"].median()),
        "median_range90_global": float(table["GlobalRange90"].median()),
        "share_range90_global_over_50": round(float((table["GlobalRange90"] > 50).mean()), 3),
        "robust_top10_local": table.loc[table["ShareTop10Local"] >= 0.9, "Municipality"].tolist(),
    }
    return SensitivityResult(table, summary)
