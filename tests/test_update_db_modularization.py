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
