import ast
import datetime as dt
import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_valuations_feature_imports_without_creating_an_engine(monkeypatch):
    import core.db

    monkeypatch.setattr(
        core.db,
        "get_engine",
        lambda: (_ for _ in ()).throw(AssertionError("Valuations import must not create an engine")),
    )
    for module_name in (
        "features.portfolio_valuations.data",
        "features.portfolio_valuations.display",
        "features.portfolio_valuations.page",
    ):
        importlib.import_module(module_name)


def test_router_uses_extracted_valuations_page_and_legacy_functions_are_removed():
    import app13
    from features.portfolio_valuations.page import portfolio_valuations_page

    assert app13.PAGE_HANDLERS["Portfolio valuations"] is portfolio_valuations_page
    tree = ast.parse((ROOT / "app13.py").read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not {
        "portfolio_valuations_page",
        "compute_portfolio_valuations_cube",
        "cached_portfolio_valuations_cube",
        "compute_portfolio_exposures_base_panel",
        "cached_portfolio_exposures_timeseries",
    } & functions


def test_valuations_layers_have_agreed_responsibilities_and_ui_contract():
    data_source = (ROOT / "features/portfolio_valuations/data.py").read_text(encoding="utf-8")
    display_source = (ROOT / "features/portfolio_valuations/display.py").read_text(encoding="utf-8")
    page_source = (ROOT / "features/portfolio_valuations/page.py").read_text(encoding="utf-8")

    assert "def cached_portfolio_valuations_cube(" in data_source
    assert "def cached_portfolio_exposures_timeseries(" in data_source
    assert "@st.cache_data(ttl=60 * 30, show_spinner=False)" in data_source
    assert "def display_valuation_time_series(" in display_source
    assert "def display_exposure_diagnostics(" in display_source
    assert "from core.fund_catalog import fetch_categories, fetch_funds_for_categories" in page_source
    assert "from core.dates import month_year_to_last_day" in page_source
    for option in (
        "Valuations of historical portfolios",
        "Historical valuations of current portfolio",
        "Financials",
        "Non-financials",
        "Total",
        "P/S",
        "P/B",
        "P/E",
        "Weighted average multiple",
        "Median multiple",
    ):
        assert option in page_source


def test_valuation_data_contracts_do_not_need_a_database_for_empty_or_invalid_inputs():
    from features.portfolio_valuations.data import compute_portfolio_valuations_cube

    expected_columns = ["fund_id", "month_end", "segment", "metric", "value"]
    assert list(
        compute_portfolio_valuations_cube(
            [], dt.date(2024, 1, 31), dt.date(2024, 12, 31), "Valuations of historical portfolios"
        ).columns
    ) == expected_columns
    with pytest.raises(ValueError, match="Unsupported agg_choice"):
        compute_portfolio_valuations_cube(
            [1],
            dt.date(2024, 1, 31),
            dt.date(2024, 12, 31),
            "Valuations of historical portfolios",
            agg_choice="invalid",
        )


def test_timeseries_contract_uses_focus_fund_and_other_funds_median(monkeypatch):
    from features.portfolio_valuations import data

    cube = pd.DataFrame(
        {
            "fund_id": [1, 2, 3, 1, 2, 3],
            "month_end": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
            "segment": ["Total"] * 6,
            "metric": ["P/E"] * 6,
            "value": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
        }
    )
    monkeypatch.setattr(data, "cached_portfolio_valuations_cube", lambda **_: cube)

    result = data.compute_portfolio_valuations_timeseries(
        fund_ids=[1, 2, 3],
        focus_fund_id=1,
        start_date=dt.date(2024, 1, 31),
        end_date=dt.date(2024, 2, 29),
        segment_choice="Total",
        metric_choice="P/E",
        mode="Valuations of historical portfolios",
    )
    assert list(result.columns) == ["month_end", "value", "series"]
    assert result.loc[result["series"] == "Universe median (others)", "value"].tolist() == [25.0, 26.0]
