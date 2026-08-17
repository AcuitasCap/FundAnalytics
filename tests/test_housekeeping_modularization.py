import importlib
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_housekeeping_modules_import_without_creating_an_engine(monkeypatch):
    import core.db

    def engine_must_not_be_created():
        raise AssertionError("Housekeeping import must not create a database engine")

    monkeypatch.setattr(core.db, "get_engine", engine_must_not_be_created)
    for module_name in (
        "features.housekeeping.data",
        "features.housekeeping.compute",
        "features.housekeeping.jobs",
        "features.housekeeping.display",
        "features.housekeeping.page",
    ):
        importlib.import_module(module_name)


def test_rolling_compute_helpers_are_pure_and_monthly():
    from features.housekeeping.compute import (
        _compute_rolling_cagr_from_monthly_nav,
        _prepare_monthly_nav_series,
    )

    dates = pd.date_range("2024-01-31", periods=13, freq="ME")
    navs = pd.DataFrame({"fund_id": [1] * 14, "nav_date": [*dates, dates[-1]], "nav_value": [100 + i for i in range(13)] + [113]})
    series = _prepare_monthly_nav_series(navs, "fund_id", "nav_date", "nav_value")[1]
    assert len(series) == 13
    assert series.iloc[-1] == 113
    rolling = _compute_rolling_cagr_from_monthly_nav(series, 12)
    assert round(float(rolling.iloc[-1]), 6) == round(113 / 100 - 1, 6)


def test_router_uses_extracted_housekeeping_page_and_debug_tool_is_removed():
    import app13

    from features.housekeeping.page import housekeeping_page

    assert app13.PAGE_HANDLERS["Housekeeping"] is housekeeping_page
    active_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "app13.py", *(ROOT / "features/housekeeping").glob("*.py")]
    )
    assert "debug_portfolio_valuation_point" not in active_sources
    assert "dbg_val_" not in active_sources


def test_adjusted_price_adapter_only_calls_external_job_when_invoked(monkeypatch):
    from features.housekeeping import jobs

    expected = ({"updates": 1}, pd.DataFrame())
    monkeypatch.setattr(jobs, "_refresh_adjusted_prices_full", lambda engine: expected)
    monkeypatch.setattr(jobs, "get_engine", lambda: "fake-engine")
    assert jobs.refresh_adjusted_prices() == expected


def test_data_layer_is_read_only_and_mutations_are_feature_jobs():
    data_source = (ROOT / "features/housekeeping/data.py").read_text(encoding="utf-8").upper()
    jobs_source = (ROOT / "features/housekeeping/jobs.py").read_text(encoding="utf-8").upper()
    assert not any(token in data_source for token in (" INSERT ", " UPDATE ", " DELETE ", " TO_SQL("))
    assert "INSERT INTO FUNDLAB.STOCK_MONTHLY_VALUATIONS" in jobs_source
