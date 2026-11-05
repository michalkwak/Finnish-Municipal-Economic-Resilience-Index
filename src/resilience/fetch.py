"""Download indicator tables from the Statistics Finland PxWeb API.

Each table is reduced to one indicator and written to ``data/raw/<key>.csv`` in a
tidy format: ``AreaCode, Area, Year, <value_col>``. A ``sources.json`` file next to
the CSVs records where and when each table was fetched.
"""

from __future__ import annotations

import itertools
import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from resilience.config import RAW_DIR, TABLES, TableSpec

log = logging.getLogger(__name__)

USER_AGENT = "finnish-resilience-index/0.2 (python-urllib)"
TIMEOUT_S = 60


def build_query(spec: TableSpec) -> dict[str, Any]:
    """PxWeb v1 query: all areas and years, one value for every other dimension."""
    query = [
        {"code": spec.area_dim, "selection": {"filter": "all", "values": ["*"]}},
        {"code": spec.time_dim, "selection": {"filter": "all", "values": ["*"]}},
    ]
    for code, values in spec.selections.items():
        query.append({"code": code, "selection": {"filter": "item", "values": values}})
    return {"query": query, "response": {"format": "json-stat2"}}


def parse_jsonstat2(payload: dict[str, Any], spec: TableSpec) -> pd.DataFrame:
    """Turn a json-stat2 dataset into ``AreaCode, Area, Year, <value_col>``.

    Every dimension other than area and time must have exactly one category,
    otherwise the table was not reduced to a single indicator.
    """
    dim_ids: list[str] = payload["id"]
    sizes: list[int] = payload["size"]
    dims = payload["dimension"]

    for dim_id, size in zip(dim_ids, sizes, strict=True):
        if dim_id not in (spec.area_dim, spec.time_dim) and size != 1:
            raise ValueError(
                f"{spec.key}: dimension {dim_id!r} has {size} categories; "
                "add it to TableSpec.selections so the table has one value per area-year"
            )
    for required in (spec.area_dim, spec.time_dim):
        if required not in dim_ids:
            raise ValueError(f"{spec.key}: response is missing dimension {required!r}")

    def ordered_codes(dim_id: str) -> list[str]:
        index = dims[dim_id]["category"]["index"]
        if isinstance(index, list):
            return index
        return sorted(index, key=index.__getitem__)

    codes_per_dim = [ordered_codes(d) for d in dim_ids]
    values = payload["value"]
    if isinstance(values, dict):  # sparse form: {"<flat index>": value}
        dense = [None] * len(list(itertools.product(*codes_per_dim)))
        for k, v in values.items():
            dense[int(k)] = v
        values = dense

    area_pos, time_pos = dim_ids.index(spec.area_dim), dim_ids.index(spec.time_dim)
    area_labels = dims[spec.area_dim]["category"].get("label", {})

    rows = []
    # json-stat2 values are row-major over `id` order, matching itertools.product.
    for combo, value in zip(itertools.product(*codes_per_dim), values, strict=True):
        area_code = combo[area_pos]
        rows.append(
            {
                "AreaCode": area_code,
                "Area": area_labels.get(area_code, area_code),
                "Year": int(combo[time_pos]),
                spec.value_col: value,
            }
        )

    df = pd.DataFrame(rows)
    df[spec.value_col] = pd.to_numeric(df[spec.value_col], errors="coerce")
    return df


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        # PxWeb prefixes some responses with a UTF-8 BOM.
        return json.loads(response.read().decode("utf-8-sig"))


def fetch_table(spec: TableSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one table. Returns the tidy frame and its provenance record."""
    log.info("Fetching %s from %s", spec.key, spec.url)
    payload = _post_json(spec.url, build_query(spec))
    df = parse_jsonstat2(payload, spec)
    source = {
        "table": payload.get("label"),
        "url": spec.url,
        "query": build_query(spec),
        "publisher_updated": payload.get("updated"),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(df),
        "source": payload.get("source"),
    }
    return df, source


def fetch_all(raw_dir: Path = RAW_DIR, keys: list[str] | None = None) -> list[Path]:
    """Fetch the configured tables into ``raw_dir`` and update ``sources.json``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources_path = raw_dir / "sources.json"
    sources = json.loads(sources_path.read_text("utf-8")) if sources_path.exists() else {}

    written = []
    for key in keys or list(TABLES):
        spec = TABLES[key]
        df, source = fetch_table(spec)
        path = raw_dir / spec.filename
        df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        sources[key] = source
        written.append(path)
        log.info("Wrote %s (%d rows)", path, len(df))

    sources_path.write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
    )
    return written
