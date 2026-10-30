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


def test_fund_portfolio_validation_uses_ace_for_domestic_only():
    import pytest
    from features.update_db.validation import validate_fund_portfolios

    upload = pd.DataFrame({
        "Scheme Name": ["Example Fund", "Example Fund"],
        "Month-end (yyyymm)": [202501, 202501],
        "Instrument": ["Example Stock", "Cash"],
        "Holding (%)": [8.5, 1.5],
        "Asset type": ["Domestic Equities", "Cash"],
        "ACE code": [100325.0, ""],
    })
    clean, summary = validate_fund_portfolios(upload)
    assert clean["ace_code"].iloc[0] == "100325"
    assert pd.isna(clean["ace_code"].iloc[1])
    assert summary["domestic_equity_rows"] == 1
    assert clean["month_end"].iloc[0].isoformat() == "2025-01-31"
    assert clean["holding_pct"].tolist() == [8.5, 1.5]

    upload.loc[0, "ACE code"] = ""
    with pytest.raises(ValueError, match="ACE codes"):
        validate_fund_portfolios(upload)


def test_missing_portfolio_ace_report_and_writer_abort_before_any_insert(monkeypatch):
    import pytest
    from features.update_db import data, writes

    statements = []

    class Result:
        def scalars(self):
            return ["100325"]

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

    monkeypatch.setattr(writes, "get_engine", lambda: Engine())
    portfolio = pd.DataFrame([
        {"scheme_name": "New Fund", "month_end": pd.Timestamp("2025-01-31").date(),
         "instrument": "Known", "holding_pct": 5.0, "asset_type": "Domestic Equities", "ace_code": "100325"},
        {"scheme_name": "New Fund", "month_end": pd.Timestamp("2025-01-31").date(),
         "instrument": "Missing", "holding_pct": 6.0, "asset_type": "Domestic Equities", "ace_code": "999999"},
    ])
    with pytest.raises(data.MissingPortfolioAceCodesError) as caught:
        writes.upload_fund_portfolios(portfolio)
    assert caught.value.report["ace_code"].tolist() == ["999999"]
    assert caught.value.report["rows_in_upload"].tolist() == [1]
    assert len(statements) == 1
    assert "FROM fundlab.stock_master_ace" in statements[0][0]

    statements.clear()
    portfolio.loc[portfolio.index[1], "ace_code"] = pd.NA
    with pytest.raises(data.MissingPortfolioAceCodesError) as blank_code:
        writes.upload_fund_portfolios(portfolio)
    assert blank_code.value.report["ace_code"].tolist() == ["(blank)"]
    assert len(statements) == 1
    assert "FROM fundlab.stock_master_ace" in statements[0][0]


def test_valid_portfolio_writer_targets_ace_table_and_nulls_cash_code(monkeypatch):
    from features.update_db import writes

    statements = []

    class Result:
        def scalars(self):
            return ["100325"]

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

    monkeypatch.setattr(writes, "get_engine", lambda: Engine())
    writes.upload_fund_portfolios(pd.DataFrame([
        {"scheme_name": "Example Fund", "month_end": pd.Timestamp("2025-01-31").date(),
         "instrument": "Stock", "holding_pct": 8.5, "asset_type": "Domestic Equities", "ace_code": "100325"},
        {"scheme_name": "Example Fund", "month_end": pd.Timestamp("2025-01-31").date(),
         "instrument": "Cash", "holding_pct": 1.5, "asset_type": "Cash", "ace_code": pd.NA},
    ]))
    assert len(statements) == 3
    assert "INSERT INTO fundlab.fund_portfolio_ace" in statements[2][0]
    assert statements[2][1]["ace_codes"] == ["100325", None]


def test_missing_portfolio_ace_codes_display_as_a_table(monkeypatch):
    from features.update_db import display

    messages = []
    tables = []
    monkeypatch.setattr(display.st, "error", messages.append)
    monkeypatch.setattr(display.st, "dataframe", lambda frame, **kwargs: tables.append(frame))
    monkeypatch.setattr(display.st, "caption", lambda *args: None)
    display.display_missing_portfolio_ace_codes(pd.DataFrame([{
        "ace_code": "999999", "rows_in_upload": 3, "example_instrument": "Example Stock"
    }]))
    assert "stock_master_ace" in messages[0]
    assert tables[0].columns.tolist() == [
        "Missing ACE code", "Holding rows", "Example instrument"
    ]


def test_ace_stock_master_validation_accepts_alias_names_and_rejects_conflicts():
    import pytest
    from features.update_db.validation import validate_ace_stock_master

    upload = pd.DataFrame({
        "ace_code": [100325.0, "100325"],
        "company_name": ["Reliance Industries Ltd.", "Reliance Industries"],
        "industry": ["Petroleum", " petroleum "],
    })
    clean, summary = validate_ace_stock_master(upload)
    assert clean.ace_code.tolist() == ["100325", "100325"]
    assert clean.industry_key.tolist() == ["petroleum", "petroleum"]
    assert summary["unique_ace_codes"] == 1
    assert summary["rows_clean"] == 2

    conflicting_industries = upload.copy()
    conflicting_industries.loc[1, "industry"] = "Banking"
    with pytest.raises(ValueError, match="conflicting industries"):
        validate_ace_stock_master(conflicting_industries)

    conflicting_names = upload.copy()
    conflicting_names.loc[1, "ace_code"] = "100326"
    conflicting_names.loc[1, "company_name"] = "reliance industries ltd."
    with pytest.raises(ValueError, match="different ACE codes"):
        validate_ace_stock_master(conflicting_names)


def test_ace_stock_master_writer_requires_new_industry_confirmation(monkeypatch):
    import pytest
    from features.update_db import writes
    from features.update_db.validation import validate_ace_stock_master

    clean, _ = validate_ace_stock_master(pd.DataFrame({
        "ace_code": ["9999999998"],
        "company_name": ["Codex Test Company"],
        "industry": ["Codex New Industry"],
    }))
    statements = []

    class Result:
        rowcount = 1

        def __iter__(self):
            return iter([("Petroleum", False)])

        def fetchall(self):
            return []

    class Conn:
        def execute(self, statement, params=None):
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

    monkeypatch.setattr(writes, "get_engine", lambda: Engine())
    with pytest.raises(ValueError, match="Classify these new industries"):
        writes.upload_ace_stock_master(clean, {})
    assert len(statements) == 1

    statements.clear()
    writes.upload_ace_stock_master(clean, {"codex new industry": True})
    assert len(statements) == 4
    assert "INSERT INTO fundlab.stock_master_ace" in statements[2][0]
    assert statements[2][1]["is_financial"] is True
    assert "INSERT INTO fundlab.company_name_acecode_map" in statements[3][0]


def test_ace_stock_master_next_step_reminder_is_prominent(monkeypatch):
    from features.update_db import display

    warnings = []
    monkeypatch.setattr(display.st, "warning", warnings.append)
    display.display_ace_stock_master_next_steps()
    assert len(warnings) == 1
    assert "**IMPORTANT NEXT STEP" in warnings[0]
    assert "stock_price_ace" in warnings[0]
    assert "stock_quarterly_financials_ace" in warnings[0]
    assert "stock_annual_bv_ace" in warnings[0]


def test_roe_roce_validation_uses_ace_code_and_numeric_year_end():
    import pytest
    from features.update_db.validation import validate_roe_roce

    upload = pd.DataFrame({
        "ACE code": [100325.0],
        "Company name": ["Reliance Industries"],
        "Year-end (YYYYMM)": [202503],
        "RoE": [10.2],
        "RoCE": [12.8],
    })
    clean, summary = validate_roe_roce(upload)
    assert clean.loc[0, "ace_code"] == "100325"
    assert clean.loc[0, "year_end"] == 202503
    assert clean.loc[0, "year_end_date"].isoformat() == "2025-03-31"
    assert summary["unique_ace_year"] == 1

    duplicate = pd.concat([upload, upload], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate.*ace_code"):
        validate_roe_roce(duplicate)


def test_roe_roce_writer_preserves_unprovided_metric_in_ace_table(monkeypatch):
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
    writes.upload_roe_roce(pd.DataFrame([{
        "ace_code": "100325", "year_end_date": pd.Timestamp("2025-03-31").date(),
        "year_end": 202503, "roe": 11.5, "roce": pd.NA,
    }]))
    sql, params = statements[0]
    assert "INSERT INTO fundlab.stock_roe_roce_ace" in sql
    assert "COALESCE(EXCLUDED.roce, stock_roe_roce_ace.roce)" in sql
    assert params["ace_code"] == "100325"
    assert params["year_end"] == 202503
    assert params["roce"] is None


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
