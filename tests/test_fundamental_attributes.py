"""Numerical and navigation regressions for shared fundamental attributes."""

import datetime as dt
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.fundamental_attributes import CURRENT, HISTORICAL
from features.fundamental_attributes.compute import compute_attribute_comparison, stock_growth
from features.fundamental_attributes import data


START = dt.date(2024, 1, 31)
END = dt.date(2024, 3, 31)


def holdings(fund_id=1, dates=("2024-01-31",), weights=(30, 22, 18, 12, 10, 8), financials=None):
    records = []
    for date in dates:
        for index, weight in enumerate(weights):
            records.append({"fund_id": fund_id, "holding_month_end": pd.Timestamp(date),
                            "anchor_month_end": pd.Timestamp(date), "holding_key": f"{fund_id}-{index}",
                            "ace_code": f"{index:05d}", "weight_pct": weight,
                            "is_financial": financials[index] if financials else False})
    return pd.DataFrame(records)


def multiples(dates=("2024-01-31", "2024-02-29", "2024-03-31"), values=(10, 20, 30, 40, 50, 60)):
    return pd.DataFrame([{"ace_code": f"{index:05d}", "month_end": pd.Timestamp(date),
                          "pe": value, "pb": value / 10, "ps": value / 5}
                         for date in dates for index, value in enumerate(values)])


def compare(h=None, facts=None, members=None, **kwargs):
    options = dict(start_date=START, end_date=END, mode=HISTORICAL, focus_id=1, focus_label="Focus",
                   compare="Peers", segment="Total", kind="valuation", metric="P/E", minimum_stocks=1)
    options.update(kwargs)
    return compute_attribute_comparison(
        holdings() if h is None else h,
        pd.DataFrame(columns=data.MEMBER_COLUMNS) if members is None else members,
        multiples() if facts is None else facts, **options)


def test_top_half_is_selected_before_segment_and_comparators_keep_full_portfolios():
    h = pd.concat([holdings(financials=[True, False, False, False, False, False]),
                   holdings(2, financials=[True, False, False, False, False, False])])
    series, stats = compare(h, segment="Non-financials", top_half=True)
    focus = stats[stats.fund_id == 1].iloc[0]
    peer = stats[stats.fund_id == 2].iloc[0]
    assert focus.selected_stock_count == 1  # 30 financial + 22 non-financial > 50.
    assert focus["median"] == 20
    assert peer.selected_stock_count == 5
    assert peer["median"] == 40
    assert series.loc[series.series == "Peers", "median"].iloc[0] == 40


def test_top_half_uses_reported_weights_and_includes_stock_beyond_exact_fifty():
    _, stats = compare(holdings(weights=(35, 15, 10, 5, 5, 5)), top_half=True)
    assert stats.iloc[0].selected_stock_count == 3
    assert stats.iloc[0]["median"] == 20


def test_unmapped_holdings_consume_weight_without_merging_null_identifiers():
    h = holdings(weights=(45, 10, 10, 10, 10, 10))
    h.loc[0, "ace_code"] = None
    series, stats = compare(h, top_half=True)
    assert stats.iloc[0].selected_stock_count == 2
    assert stats.iloc[0].valid_stock_count == 1
    assert stats.iloc[0]["median"] == 20
    assert stats.iloc[0].coverage_pct == pytest.approx(100 * 10 / 55)


def test_repeated_ace_holdings_are_combined_before_ranking():
    h = holdings(weights=(20, 20, 15, 15, 15, 15))
    h.loc[1, "ace_code"] = h.loc[0, "ace_code"]
    _, stats = compare(h, top_half=True)
    assert stats.iloc[0].selected_stock_count == 2
    assert stats.iloc[0]["median"] == 20  # ACE 00000 (40%) and 00002 (15%).


def test_peer_median_is_median_of_fund_medians_and_only_uses_focus_dates():
    focus = holdings(dates=("2024-01-31", "2024-03-31"))
    peer1 = holdings(2, dates=("2024-02-29", "2024-03-31"), weights=(100,))
    peer2 = holdings(3, dates=("2024-03-31",))
    series, _ = compare(pd.concat([focus, peer1, peer2]))
    peers = series[series.series == "Peers"]
    assert peers.month_end.tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-03-31")]
    assert np.isnan(peers.iloc[0]["median"])
    assert peers.iloc[0].peer_fund_count == 0
    assert peers.iloc[1]["median"] == 22.5  # median of 10 and 35, rather than pooled stock median.
    assert peers.iloc[1].peer_fund_count == 2


def test_market_comparator_uses_full_segment_and_has_no_extra_months():
    members = multiples()[["ace_code", "month_end"]]
    members["is_financial"] = False
    series, stats = compare(holdings(dates=("2024-01-31", "2024-03-31")), members=members,
                            compare="Midcaps", top_half=True)
    market = series[series.series == "Midcaps"]
    assert market["median"].tolist() == [35, 35]
    assert market.valid_stock_count.tolist() == [6, 6]
    assert stats["median"].tolist() == [15, 15]
    assert len(market) == 2


def test_current_portfolio_keeps_actual_anchor_and_uses_monthly_valuations():
    series, stats = compare(holdings(dates=("2023-12-31",)), mode=CURRENT)
    assert stats.month_end.tolist() == list(pd.date_range(START, END, freq="ME"))
    assert stats.anchor_month_end.tolist() == [pd.Timestamp("2023-12-31")] * 3
    assert series[series.series == "Focus"]["median"].tolist() == [35] * 3


@pytest.mark.parametrize("mode", [HISTORICAL, CURRENT])
@pytest.mark.parametrize("metric,expected", [("P/E", 35), ("P/B", 3.5), ("P/S", 7)])
@pytest.mark.parametrize("comparator", ["Peers", "Midcaps"])
def test_valuations_match_month_year_despite_different_days(mode, metric, expected, comparator):
    h = pd.concat([holdings(dates=("2025-11-30",)), holdings(2, dates=("2025-11-27",))])
    facts = multiples(dates=("2025-11-28",))
    prior_year = multiples(dates=("2024-11-28",), values=(999,) * 6)
    members = facts[["ace_code", "month_end"]].assign(is_financial=False)
    series, stats = compare(h, facts=pd.concat([facts, prior_year]), members=members,
                            mode=mode, compare=comparator, metric=metric, minimum_stocks=5,
                            start_date=dt.date(2025, 11, 30), end_date=dt.date(2025, 11, 30))
    assert series.month_end.tolist() == [pd.Timestamp("2025-11-30")] * 2
    assert series["median"].tolist() == pytest.approx([expected, expected])
    assert stats.valid_stock_count.tolist() == [6, 6]


@pytest.mark.parametrize("mode", [HISTORICAL, CURRENT])
def test_loader_includes_entire_selected_month_and_normalizes_valuation_dates(monkeypatch, mode):
    events = []

    class Connection:
        def execute(self, query):
            pass

    class Engine:
        @contextmanager
        def begin(self):
            yield Connection()

    def read_sql(query, conn, params):
        sql = str(query)
        events.append((sql, params))
        if "fund_portfolio_ace" in sql:
            assert params["start_date"] == dt.date(2025, 11, 1)
            assert params["end_date"] == dt.date(2025, 11, 30)
            return holdings(dates=("2025-11-28",))
        assert params["fact_start"] == dt.date(2025, 11, 1)
        assert params["end_date"] == dt.date(2025, 11, 30)
        rows = multiples(dates=("2025-11-28",))
        return rows[rows.month_end.between(pd.Timestamp(params["fact_start"]), pd.Timestamp(params["end_date"]))]

    monkeypatch.setattr(data, "get_engine", lambda: Engine())
    monkeypatch.setattr(data.pd, "read_sql", read_sql)
    # The day supplied by a selector/caller must not affect the monthly result.
    h, members, facts = data.load_attribute_inputs(
        [1], dt.date(2025, 11, 30), dt.date(2025, 11, 1), mode, "Peers", "valuation")
    assert len(facts) == 6
    assert facts.month_end.eq(pd.Timestamp("2025-11-30")).all()
    series, _ = compare(h, facts=facts, mode=mode, minimum_stocks=5,
                        start_date=dt.date(2025, 11, 1), end_date=dt.date(2025, 11, 30))
    assert series.loc[series.series == "Focus", "median"].iloc[0] == 35


def test_minimum_stock_rule_and_empty_segments_keep_missing_period_rows():
    _, stats = compare(facts=multiples(values=(10, 20, 30, 40)), minimum_stocks=5)
    assert stats.iloc[0].valid_stock_count == 4
    assert np.isnan(stats.iloc[0]["median"])
    series, stats = compare(segment="Financials")
    assert stats.iloc[0].selected_stock_count == 0
    assert np.isnan(stats.iloc[0]["median"])
    assert len(series) == 2


def financials():
    dates = pd.period_range("2020Q1", "2023Q4", freq="Q").to_timestamp("Q")
    return pd.DataFrame({"ace_code": ["00000"] * 16, "period_end": dates,
                         "sales": [100] * 4 + [110] * 4 + [121] * 4 + [133.1] * 4,
                         "pat": [100] * 4 + [90] * 4 + [81] * 4 + [72.9] * 4})


@pytest.mark.parametrize("metric,expected", [("Sales", 10), ("PAT", -10)])
def test_growth_is_three_year_ttm_cagr(metric, expected):
    result = stock_growth(financials(), metric)
    assert result.iloc[-1]["value"] == pytest.approx(expected)
    assert result.iloc[-1].base_ttm == 400
    assert result.iloc[:-1]["value"].isna().all()


@pytest.mark.parametrize("index,value", [(0, None), (0, -1000), (15, 0), (15, -1000)])
def test_growth_missing_quarters_and_nonpositive_ttm_are_not_fabricated(index, value):
    frame = financials()
    if value is None:
        frame = frame.drop(index)
    else:
        frame.loc[index, "sales"] = value
    result = stock_growth(frame, "Sales").iloc[-1]
    if index == 15 and value == 0:  # A zero quarter is allowed when the TTM total remains positive.
        assert np.isfinite(result["value"])
    else:
        assert np.isnan(result["value"])


def test_growth_uses_previous_quarter_and_retains_monthly_portfolio_dates():
    facts = financials()
    facts.loc[len(facts)] = ["00000", pd.Timestamp("2024-03-31"), 100000, 100000]
    h = holdings(dates=("2024-01-31", "2024-02-29", "2024-03-31"), weights=(100,))
    series, _ = compare(h, facts=facts, kind="growth", metric="Sales")
    assert series[series.series == "Focus"]["median"].tolist() == pytest.approx([10, 10, 10])
    _, stats = compare(holdings(dates=("2024-04-30",), weights=(100,)), facts=facts,
                       start_date=dt.date(2024, 4, 30), end_date=dt.date(2024, 4, 30), kind="growth", metric="Sales")
    assert stats.iloc[0]["median"] > 100


def test_missing_stock_facts_produce_gaps_instead_of_errors():
    for kind, metric, columns in [("growth", "PAT", ["ace_code", "period_end", "sales", "pat"]),
                                   ("valuation", "P/E", ["ace_code", "month_end", "ps", "pe", "pb"])]:
        series, stats = compare(facts=pd.DataFrame(columns=columns), kind=kind, metric=metric)
        assert series["median"].isna().all()
        assert stats.valid_stock_count.eq(0).all()


@pytest.mark.parametrize("kind,mode,cohort", [("valuation", HISTORICAL, "Peers"),
                                             ("growth", HISTORICAL, "Top 500"),
                                             ("valuation", CURRENT, "Midcaps")])
def test_data_loader_uses_one_readonly_snapshot_and_historical_ace_sources(monkeypatch, kind, mode, cohort):
    events = []

    class Connection:
        def execute(self, sql):
            events.append((str(sql), {}))

    class Engine:
        @contextmanager
        def begin(self):
            yield Connection()

    def read_sql(query, conn, params):
        sql = str(query)
        events.append((sql, params))
        if "fund_portfolio_ace" in sql:
            return holdings()
        if "stock_size_band_ace" in sql:
            return pd.DataFrame({"ace_code": ["00123"], "month_end": [START], "is_financial": [True]})
        return multiples() if kind == "valuation" else financials()

    monkeypatch.setattr(data, "get_engine", lambda: Engine())
    monkeypatch.setattr(data.pd, "read_sql", read_sql)
    h, members, facts = data.load_attribute_inputs([1], START, END, mode, cohort, kind)
    assert events[0][0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert "LEFT JOIN fundlab.stock_master_ace" in events[1][0]
    assert h.ace_code.iloc[0] == "00000"
    if mode == CURRENT:
        assert "MAX(month_end) AS anchor_month_end" in events[1][0]
        assert "month_end <= :end_date" in events[1][0]
    if cohort != "Peers":
        assert "band.band_date <= m.month_end" in events[2][0]
        assert "MAX(band.band_date) AS band_date" in events[2][0]
        assert "b.band_date = snapshot.band_date" in events[2][0]
        assert "DISTINCT ON (band.ace_code)" not in events[2][0]
        assert "b.size_band IN ('Large', 'Mid', 'Small')" in events[2][0]
        assert "00123" in events[-1][1]["ace_codes"]
    if kind == "growth":
        assert "fundlab.stock_quarterly_financials_ace" in events[-1][0]
        assert events[-1][1]["fact_start"] <= dt.date(2020, 1, 1)
    else:
        assert "fundlab.stock_monthly_valuations_ace" in events[-1][0]
    assert not any(word in sql.upper().split() for sql, _ in events for word in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"))


def test_home_routes_quality_and_fund_valuations_into_fundamental_attributes():
    import app13
    from features.fundamental_attributes.page import fundamental_attributes_page

    assert app13.PAGE_HANDLERS["Fundamental attributes"] is fundamental_attributes_page
    assert "Portfolio quality" not in app13.PAGE_HANDLERS
    assert "Fund valuations" not in app13.PAGE_HANDLERS
    assert "Market valuations" in app13.PAGE_HANDLERS
