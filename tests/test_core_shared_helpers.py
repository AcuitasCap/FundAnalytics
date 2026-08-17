"""Focused coverage for shared DataFrame and fund-catalog helpers."""

import importlib
import sys
from pathlib import Path

import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_shared_helper_imports_do_not_construct_an_engine(monkeypatch):
    import core.db as db

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Shared helper imports must not construct an engine")

    monkeypatch.setattr(db, "create_engine", fail_if_called)
    monkeypatch.setattr(db, "_engine", None)
    sys.modules.pop("core.dataframes", None)
    sys.modules.pop("core.fund_catalog", None)

    importlib.import_module("core.dataframes")
    importlib.import_module("core.fund_catalog")

    assert db._engine is None


def test_fetch_funds_for_categories_has_stable_empty_schema(monkeypatch):
    import core.fund_catalog as catalog

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Empty category lookup must not access the database")

    monkeypatch.setattr(catalog, "get_engine", fail_if_called)
    catalog.fetch_funds_for_categories.clear()
    result = catalog.fetch_funds_for_categories([])

    assert result.empty
    assert result.columns.tolist() == ["fund_id", "fund_name", "category_name"]


def test_monthly_row_helper_deduplicates_exact_rows_and_rejects_conflicts():
    from core.dataframes import ensure_unique_monthly_rows

    exact = pd.DataFrame({"isin": ["A", "A"], "month": ["2026-01", "2026-01"], "value": [1, 1]})
    deduped = ensure_unique_monthly_rows(exact, ["isin", "month"], ["value"], "test")
    assert len(deduped) == 1

    conflicting = exact.copy()
    conflicting.loc[1, "value"] = 2
    try:
        ensure_unique_monthly_rows(conflicting, ["isin", "month"], ["value"], "test")
    except ValueError as error:
        assert "conflicting duplicate rows" in str(error)
    else:
        raise AssertionError("Conflicting duplicate rows must fail")


def test_app_does_not_reexport_fund_catalog_helpers():
    app = importlib.import_module("app13")
    assert not hasattr(app, "fetch_funds_by_categories")
    assert not hasattr(app, "fetch_categories")
    assert not hasattr(app, "fetch_funds_for_categories")
