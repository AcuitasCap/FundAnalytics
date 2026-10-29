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
            "ACE code": ["0288853", 288853.0],
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
        assert clean["ace_code"].tolist() == ["0288853", "288853"]
        assert clean["period_end"].dt.strftime("%Y-%m-%d").tolist() == [
            "2025-03-31", "2025-06-30"
        ]
        assert clean.columns.tolist() == ["ace_code", "period_end", value_column]
        assert clean[value_column].tolist() == [100.0, 125.0]

    annual_upload = pd.DataFrame(
        {"ACE code": ["288853"], "Year end": [202503], "Book value": [500.0]}
    )
    clean, summary = validate_annual_book_value(annual_upload)
    assert summary["rows_clean"] == 1
    assert clean.loc[0, "ace_code"] == "288853"
    assert clean.loc[0, "year_end"] == pd.Timestamp("2025-03-31")
    assert clean.loc[0, "book_value"] == 500.0
    assert clean.columns.tolist() == ["ace_code", "year_end", "book_value"]


def test_annual_book_value_rejects_bad_ace_codes_and_duplicate_keys():
    import pytest
    from features.update_db.validation import validate_annual_book_value

    with pytest.raises(ValueError, match="ACE codes"):
        validate_annual_book_value(
            pd.DataFrame({"ACE": ["INE000000001"], "Period": [202503], "BV": [500]})
        )
    with pytest.raises(ValueError, match="Duplicate rows.*ACE code"):
        validate_annual_book_value(
            pd.DataFrame({"ACE": [288853, "288853"], "Period": [202503, 202503], "BV": [500, 600]})
        )


def test_quarterly_validators_reject_bad_ace_codes_and_duplicate_keys():
    import pytest
    from features.update_db.validation import validate_quarterly_pat, validate_quarterly_sales

    for validate in (validate_quarterly_pat, validate_quarterly_sales):
        with pytest.raises(ValueError, match="ACE codes"):
            validate(pd.DataFrame({"ACE": ["INE000000001"], "Period": [202503], "Value": [500]}))
        with pytest.raises(ValueError, match="Duplicate rows.*ACE code"):
            validate(pd.DataFrame({
                "ACE": [288853, "288853"], "Period": [202503, 202503], "Value": [500, 600]
            }))


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


def test_annual_book_value_upload_checks_conflicts_before_inserting(monkeypatch):
    from features.update_db import data, writes

    statements = []

    class Result:
        def fetchall(self):
            return []

    class Conn:
        def execute(self, statement, params):
            statements.append((str(statement), params))
            return Result()

    class Engine:
        def begin(self):
            class Context:
                def __enter__(self):
                    return Conn()

                def __exit__(self, *args):
                    return False

            return Context()

    engine = Engine()
    monkeypatch.setattr(data, "get_engine", lambda: engine)
    monkeypatch.setattr(writes, "get_engine", lambda: engine)

    writes.upload_annual_book_value(
        pd.DataFrame(
            [{"ace_code": "288853", "year_end": pd.Timestamp("2025-03-31"),
              "book_value": 500.0}]
        )
    )

    assert len(statements) == 2
    assert "stock_annual_bv_ace" in statements[0][0]
    assert "INSERT INTO fundlab.stock_annual_bv_ace" in statements[1][0]
    assert statements[1][1]["ace_codes"] == ["288853"]
    assert statements[1][1]["year_ends"] == [pd.Timestamp("2025-03-31").date()]


def test_quarterly_writers_upsert_only_selected_ace_metric(monkeypatch):
    from features.update_db import writes

    statements = []

    class Conn:
        def execute(self, statement, params):
            statements.append((str(statement), params))

    class Engine:
        def begin(self):
            class Context:
                def __enter__(self):
                    return Conn()

                def __exit__(self, *args):
                    return False

            return Context()

    monkeypatch.setattr(writes, "get_engine", lambda: Engine())
    for metric, upload in (
        ("pat", writes.upload_quarterly_pat),
        ("sales", writes.upload_quarterly_sales),
    ):
        upload(pd.DataFrame([{
            "ace_code": "0288853", "period_end": pd.Timestamp("2025-03-31"), metric: 500.0
        }]))
        sql, params = statements[-1]
        assert "INSERT INTO fundlab.stock_quarterly_financials_ace" in sql
        assert "ON CONFLICT (ace_code, period_end) DO UPDATE" in sql
        assert f"SET {metric} = EXCLUDED.{metric}" in sql
        assert params == {
            "ace_codes": ["0288853"], "period_ends": [pd.Timestamp("2025-03-31").date()],
            "values": [500.0],
        }


def test_quarterly_upload_templates_show_ace_codes(monkeypatch):
    from features.update_db import display

    messages = []
    monkeypatch.setattr(display.st, "markdown", messages.append)
    for upload_type in ("Company PAT (quarterly)", "Company sales (quarterly)"):
        display.show_expected_format(upload_type)
    assert len(messages) == 2
    assert all("ACE code" in message and "ISIN" not in message for message in messages)


def test_app_routes_only_to_extracted_update_db_page():
    source = open("app13.py", encoding="utf-8").read()
    assert "from features.update_db.page import update_db_page" in source
    assert "def update_db_page" not in source
