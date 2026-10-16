import importlib
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_update_db_modules_import_without_creating_an_engine(monkeypatch):
    import core.db

    def fail_engine():
        raise AssertionError("an Update DB module created an engine during import")

    monkeypatch.setattr(core.db, "get_engine", fail_engine)
    for name in (
        "features.update_db.data",
        "features.update_db.validation",
        "features.update_db.writes",
        "features.update_db.display",
        "features.update_db.page",
    ):
        importlib.reload(importlib.import_module(name))


def test_fund_nav_validation_is_pure_dataframe_work():
    from features.update_db.validation import validate_fund_navs

    clean, summary = validate_fund_navs(
        pd.DataFrame(
            {"Fund": ["Example Fund"], "Date": ["2024-01-31"], "NAV": [123.45]}
        )
    )

    assert summary == {"rows": 1, "unique_fund_dates": 1}
    assert clean.loc[0, "fund_name"] == "Example Fund"


def test_company_financial_validators_parse_uploaded_periods_without_database_access():
    from features.update_db.validation import (
        validate_annual_book_value,
        validate_quarterly_pat,
        validate_quarterly_sales,
    )

    quarterly_upload = pd.DataFrame(
        {
            "ISIN": ["ine000000001", "INE000000001"],
            "Quarter end": [202503, 202506],
            "Value": [100.0, 125.0],
        }
    )
    for validate, value_column in (
        (validate_quarterly_sales, "sales"),
        (validate_quarterly_pat, "pat"),
    ):
        clean, summary = validate(quarterly_upload)
        assert summary["rows_clean"] == 2
        assert clean["isin"].tolist() == ["INE000000001", "INE000000001"]
        assert clean["period_end"].dt.strftime("%Y-%m-%d").tolist() == [
            "2025-03-31", "2025-06-30"
        ]
        assert clean["fiscal_quarter"].tolist() == [1, 2]
        assert clean[value_column].tolist() == [100.0, 125.0]

    annual_upload = pd.DataFrame(
        {"ISIN": ["INE000000001"], "Year end": [202503], "Book value": [500.0]}
    )
    clean, summary = validate_annual_book_value(annual_upload)
    assert summary["rows_clean"] == 1
    assert clean.loc[0, "year_end"] == pd.Timestamp("2025-03-31")
    assert clean.loc[0, "book_value"] == 500.0


def test_writer_only_mutates_when_explicitly_called(monkeypatch):
    from features.update_db import writes

    calls = []

    class Conn:
        def execute(self, *args, **kwargs):
            calls.append((args, kwargs))

    class Engine:
        def begin(self):
            class Context:
                def __enter__(self):
                    return Conn()

                def __exit__(self, *args):
                    return False

            return Context()

    monkeypatch.setattr(writes, "get_engine", lambda: Engine())
    assert calls == []
    writes.upload_stock_master(
        pd.DataFrame(
            [{"isin": "INE000000001", "company_name": "Example", "industry": "Other", "is_financial": False}]
        )
    )
    assert calls


def test_app_routes_only_to_extracted_update_db_page():
    source = open("app13.py", encoding="utf-8").read()
    assert "from features.update_db.page import update_db_page" in source
    assert "def update_db_page" not in source
