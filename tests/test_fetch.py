import pytest

from resilience.config import TABLES, TableSpec
from resilience.fetch import build_query, fetch_table, parse_jsonstat2

SPEC = TableSpec(key="t", value_col="V", table_path="x/y.px", selections={"contentscode": ["c1"]})


def _payload(values, contents=("c1",)):
    return {
        "id": ["alue_23_20250101", "timeperiod_y", "contentscode"],
        "size": [2, 2, len(contents)],
        "dimension": {
            "alue_23_20250101": {
                "category": {
                    "index": {"SSS": 0, "KU020": 1},
                    "label": {"SSS": "WHOLE COUNTRY", "KU020": "Akaa"},
                }
            },
            "timeperiod_y": {"category": {"index": {"2023": 0, "2024": 1}}},
            "contentscode": {"category": {"index": {c: i for i, c in enumerate(contents)}}},
        },
        "value": values,
    }


def test_build_query_selects_all_areas_and_years():
    q = build_query(SPEC)
    by_code = {item["code"]: item["selection"] for item in q["query"]}
    assert by_code["alue_23_20250101"] == {"filter": "all", "values": ["*"]}
    assert by_code["timeperiod_y"] == {"filter": "all", "values": ["*"]}
    assert by_code["contentscode"] == {"filter": "item", "values": ["c1"]}
    assert q["response"]["format"] == "json-stat2"


def test_parse_row_major_order_and_labels():
    df = parse_jsonstat2(_payload([1, 2, 3, None]), SPEC)
    assert df.to_dict("records") == [
        {"AreaCode": "SSS", "Area": "WHOLE COUNTRY", "Year": 2023, "V": 1.0},
        {"AreaCode": "SSS", "Area": "WHOLE COUNTRY", "Year": 2024, "V": 2.0},
        {"AreaCode": "KU020", "Area": "Akaa", "Year": 2023, "V": 3.0},
        {
            "AreaCode": "KU020",
            "Area": "Akaa",
            "Year": 2024,
            "V": pytest.approx(float("nan"), nan_ok=True),
        },
    ]


def test_parse_sparse_values():
    df = parse_jsonstat2(_payload({"0": 1, "3": 4}), SPEC)
    assert df["V"].isna().tolist() == [False, True, True, False]


def test_parse_rejects_unreduced_dimension():
    with pytest.raises(ValueError, match="contentscode"):
        parse_jsonstat2(_payload([1] * 8, contents=("c1", "c2")), SPEC)


def test_parse_national_table_without_area_dim():
    spec = TableSpec(
        key="cpi",
        value_col="CPI",
        table_path="khi/11xt.px",
        selections={"contentscode": ["p"]},
        area_dim=None,
    )
    payload = {
        "id": ["timeperiod_y", "contentscode"],
        "size": [2, 1],
        "dimension": {
            "timeperiod_y": {"category": {"index": {"2023": 0, "2024": 1}}},
            "contentscode": {"category": {"index": {"p": 0}}},
        },
        "value": [160.0, 167.7],
    }
    df = parse_jsonstat2(payload, spec)
    assert df["AreaCode"].tolist() == ["SSS", "SSS"]
    assert df["CPI"].tolist() == [160.0, 167.7]
    codes = [q["code"] for q in build_query(spec)["query"]]
    assert codes == ["timeperiod_y", "contentscode"]


def test_every_configured_table_fixes_its_extra_dims():
    for spec in TABLES.values():
        assert spec.selections, spec.key
        assert all(len(v) == 1 for v in spec.selections.values()), spec.key


@pytest.mark.network
@pytest.mark.parametrize("key", list(TABLES))
def test_live_api_returns_expected_areas(key):
    spec = TABLES[key]
    df, source = fetch_table(spec)
    if spec.area_dim is None:
        assert (df["AreaCode"] == "SSS").all()
        assert df[spec.value_col].notna().sum() > 25
    else:
        assert df["AreaCode"].str.startswith("KU").sum() > 250
    assert source["rows"] == len(df)
