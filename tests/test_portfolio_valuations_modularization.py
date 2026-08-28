import ast
import datetime as dt
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _holdings(fund_id=1, anchor="2024-01-31", weights=None):
    weights = weights or [20, 18, 16, 14, 12, 10, 6, 4]
    return pd.DataFrame(
        {
            "fund_id": [fund_id] * len(weights),
            "holding_month_end": pd.to_datetime([anchor] * len(weights)),
            "anchor_month_end": pd.to_datetime([anchor] * len(weights)),
            "isin": [f"S{i}" for i in range(len(weights))],
            "weight_pct": weights,
            "is_financial": [False] * len(weights),
        }
    )


def _multiples(months=("2024-01-31",), count=8, offset=1.0):
    rows = []
    for month in months:
        for index in range(count):
            rows.append(
                {
                    "isin": f"S{index}",
                    "month_end": pd.Timestamp(month),
                    "ps": offset + index,
                    "pe": 10 * (offset + index),
                    "pb": 2 * (offset + index),
                }
            )
    return pd.DataFrame(rows)


def test_valuations_feature_imports_without_creating_an_engine(monkeypatch):
    import core.db

    monkeypatch.setattr(
        core.db,
        "get_engine",
        lambda: (_ for _ in ()).throw(AssertionError("Valuations import must not create an engine")),
    )
    for module_name in (
        "features.portfolio_valuations.data",
        "features.portfolio_valuations.compute",
        "features.portfolio_valuations.display",
        "features.portfolio_valuations.page",
    ):
        importlib.import_module(module_name)


def test_router_uses_extracted_valuations_page_and_layers_are_one_way():
    import app13
    from features.portfolio_valuations.page import portfolio_valuations_page

    assert app13.PAGE_HANDLERS["Portfolio valuations"] is portfolio_valuations_page
    tree = ast.parse((ROOT / "app13.py").read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "portfolio_valuations_page" not in functions
    for path in (ROOT / "features/portfolio_valuations").glob("*.py"):
        assert "import app13" not in path.read_text(encoding="utf-8")


def test_ui_uses_ranges_and_no_stored_fund_valuation_or_aggregation_controls():
    page_source = (ROOT / "features/portfolio_valuations/page.py").read_text(encoding="utf-8")
    active_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "features/portfolio_valuations").glob("*.py")
    )
    for option in (
        "Valuations of historical portfolios",
        "Historical valuations of current portfolio",
        "Financials",
        "Non-financials",
        "Total",
        "P/S",
        "P/B",
        "P/E",
        "Full valuation range",
        "Valuation range of top 50% stocks",
    ):
        assert option in page_source
    assert "Weighted average multiple" not in active_source
    assert "Median multiple" not in active_source
    assert "fund_monthly_valuations" not in active_source


def test_full_range_is_unweighted_and_requires_five_valid_stocks():
    from features.portfolio_valuations.compute import compute_fund_valuation_ranges

    result = compute_fund_valuation_ranges(
        _holdings(),
        _multiples(),
        start_date=dt.date(2024, 1, 31),
        end_date=dt.date(2024, 1, 31),
        mode="Valuations of historical portfolios",
        segment="Total",
        metric="P/S",
        valuation_range="Full valuation range",
    ).iloc[0]
    assert result["valid_stock_count"] == 8
    assert result["p25"] == 2.75
    assert result["median"] == 4.5
    assert result["p75"] == 6.25

    sparse = _multiples(count=4)
    suppressed = compute_fund_valuation_ranges(
        _holdings(),
        sparse,
        start_date=dt.date(2024, 1, 31),
        end_date=dt.date(2024, 1, 31),
        mode="Valuations of historical portfolios",
        segment="Total",
        metric="P/S",
        valuation_range="Full valuation range",
    ).iloc[0]
    assert suppressed["valid_stock_count"] == 4
    assert np.isnan(suppressed["median"])


def test_top_half_includes_crossing_stock_then_calculates_unweighted_percentiles():
    from features.portfolio_valuations.compute import compute_fund_valuation_ranges

    weights = [12, 11, 10, 9, 8, 7, 7, 6, 5, 5, 5, 5, 4, 3, 2, 1]
    result = compute_fund_valuation_ranges(
        _holdings(weights=weights),
        _multiples(count=len(weights)),
        start_date=dt.date(2024, 1, 31),
        end_date=dt.date(2024, 1, 31),
        mode="Valuations of historical portfolios",
        segment="Total",
        metric="P/S",
        valuation_range="Valuation range of top 50% stocks",
    ).iloc[0]
    assert result["selected_stock_count"] == 5
    assert result["valid_stock_count"] == 5
    assert result["p25"] == 2.0
    assert result["median"] == 3.0
    assert result["p75"] == 4.0


def test_current_portfolio_is_replicated_and_preserves_actual_anchor_month():
    from features.portfolio_valuations.compute import compute_fund_valuation_ranges

    result = compute_fund_valuation_ranges(
        _holdings(anchor="2024-01-31"),
        _multiples(months=("2024-01-31", "2024-02-29")),
        start_date=dt.date(2024, 1, 31),
        end_date=dt.date(2024, 2, 29),
        mode="Historical valuations of current portfolio",
        segment="Total",
        metric="P/S",
        valuation_range="Full valuation range",
    )
    assert result["month_end"].tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert result["anchor_month_end"].nunique() == 1
    assert result["anchor_month_end"].iloc[0] == pd.Timestamp("2024-01-31")


def test_peer_series_uses_median_of_corresponding_fund_statistics():
    from features.portfolio_valuations.compute import build_focus_peer_series

    ranges = pd.DataFrame(
        {
            "fund_id": [1, 2, 3],
            "month_end": pd.to_datetime(["2024-01-31"] * 3),
            "p25": [5.0, 10.0, 30.0],
            "median": [7.0, 20.0, 40.0],
            "p75": [9.0, 50.0, 70.0],
            "valid_stock_count": [8, 9, 10],
        }
    )
    peer = build_focus_peer_series(ranges, 1)
    peer = peer[peer["series"] == "Peer-set"].iloc[0]
    assert peer["p25"] == 20.0
    assert peer["median"] == 30.0
    assert peer["p75"] == 60.0
    assert peer["peer_fund_count"] == 2
