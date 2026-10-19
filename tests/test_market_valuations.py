"""Market valuation routing and percentile regressions (no live database writes)."""

import importlib

import pandas as pd
import pytest


def test_market_valuation_modules_import_without_connecting(monkeypatch):
    import core.db

    monkeypatch.setattr(
        core.db, "get_engine",
        lambda: (_ for _ in ()).throw(AssertionError("Import must not connect to Supabase")),
    )
    for module_name in (
        "features.market_valuations.data",
        "features.market_valuations.compute",
        "features.market_valuations.display",
        "features.market_valuations.page",
    ):
        importlib.import_module(module_name)


def test_valuation_router_exposes_fund_and_market_subpages():
    import app13
    from features.market_valuations.page import market_valuations_page
    from features.portfolio_valuations.page import portfolio_valuations_page

    assert app13.PAGE_HANDLERS["Valuations"] is app13.valuations_page
    assert app13.PAGE_HANDLERS["Fund valuations"] is portfolio_valuations_page
    assert app13.PAGE_HANDLERS["Market valuations"] is market_valuations_page


def test_market_percentiles_are_unweighted_and_reference_is_median_of_medians():
    from features.market_valuations.compute import compute_market_valuation_ranges

    frame = pd.DataFrame(
        [
            {"isin": f"A{i}", "month_end": "2025-01-31", "multiple": v}
            for i, v in enumerate([1, 2, 3, 4, 100])
        ] + [
            {"isin": f"A{i}", "month_end": "2025-02-28", "multiple": v}
            for i, v in enumerate([10, 20, 30, 40, 50])
        ]
    )
    result = compute_market_valuation_ranges(frame)

    assert result["median"].tolist() == [3, 30]
    assert result["p25"].tolist() == [2, 20]
    assert result["p75"].tolist() == [4, 40]
    assert result["historical_median"].tolist() == [16.5, 16.5]
    assert result["stock_count"].tolist() == [5, 5]


def test_market_percentiles_ignore_invalid_multiples_and_sparse_months():
    from features.market_valuations.compute import compute_market_valuation_ranges

    frame = pd.DataFrame(
        [
            {"isin": f"A{i}", "month_end": "2025-01-31", "multiple": v}
            for i, v in enumerate([1, 2, 3, 4, 5, 0, -1, None])
        ] + [
            {"isin": f"A{i}", "month_end": "2025-02-28", "multiple": v}
            for i, v in enumerate([1, 2, 3, 4])
        ]
    )
    result = compute_market_valuation_ranges(frame)

    assert result["month_end"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-31"]
    assert result.loc[0, "stock_count"] == 5
    assert result.loc[0, "median"] == 3


def test_market_query_uses_only_historical_size_bands_and_safe_metric(monkeypatch):
    from features.market_valuations import data

    statements = []

    class Engine:
        def connect(self):
            class Connection:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Connection()

    def fake_read_sql(query, conn, params):
        statements.append((str(query), params))
        return pd.DataFrame(columns=["isin", "month_end", "multiple"])

    monkeypatch.setattr(data, "get_engine", lambda: Engine())
    monkeypatch.setattr(data.pd, "read_sql", fake_read_sql)
    data.load_market_stock_multiples(
        "Mid caps", "Financials", "P/B",
        pd.Timestamp("2025-01-31").date(), pd.Timestamp("2025-03-31").date(),
    )
    query, params = statements[0]
    assert "band.band_date <= v.month_end" in query
    assert "v.pb AS multiple" in query
    assert params["size_band"] == "Mid"
    assert params["segment"] == "Financials"
    assert params["start_date"] == pd.Timestamp("2025-01-01").date()
    with pytest.raises(ValueError, match="Unsupported valuation metric"):
        data.load_market_stock_multiples(
            "Mid caps", "Total", "unsafe", params["start_date"], params["end_date"]
        )
    data.load_market_stock_multiples(
        "Nifty 50", "Total", "P/E", params["start_date"], params["end_date"]
    )
    nifty_query, nifty_params = statements[1]
    assert "fundlab.fund_portfolio" in nifty_query
    assert "fp.asset_type = 'Domestic Equities'" in nifty_query
    assert "fp.holding_weight > 0" in nifty_query
    assert "v.month_end >= c.month_start" in nifty_query
    assert "v.month_end < c.month_start + INTERVAL '1 month'" in nifty_query
    assert "stock_size_band" not in nifty_query
    assert nifty_params["fund_name"] == "Nippon India ETF Nifty 50 BeES"
    assert "size_band" not in nifty_params


def test_market_data_groups_last_trading_day_into_calendar_month(monkeypatch):
    from features.market_valuations import data

    class Engine:
        def connect(self):
            class Connection:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Connection()

    monkeypatch.setattr(data, "get_engine", lambda: Engine())
    monkeypatch.setattr(
        data.pd, "read_sql",
        lambda *args, **kwargs: pd.DataFrame(
            [{"isin": "INE000000001", "month_end": "2025-11-28", "multiple": 12.0}]
        ),
    )
    result = data.load_market_stock_multiples(
        "Large caps", "Total", "P/E",
        pd.Timestamp("2025-11-30").date(), pd.Timestamp("2025-11-30").date(),
    )
    assert result.loc[0, "month_end"] == pd.Timestamp("2025-11-30")
