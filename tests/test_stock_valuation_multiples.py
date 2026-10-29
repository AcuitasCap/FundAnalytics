import numpy as np
import pandas as pd
import sys
import datetime as dt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from features.housekeeping.jobs import _calculate_stock_valuation_multiples


def test_stock_multiples_apply_limits_validity_and_freshness():
    frame = pd.DataFrame(
        [
            # Floor P/S, invalid negative PAT, ordinary P/B.
            {"month_end": "2023-01-31", "market_cap": 10, "ttm_sales": 100, "ttm_pat": -2, "book_value": 7,
             "quarter_anchor": "2022-12-31", "book_anchor": "2022-03-31"},
            # Cap all three values.
            {"month_end": "2025-09-30", "market_cap": 1_000, "ttm_sales": 1, "ttm_pat": 1, "book_value": 1,
             "quarter_anchor": "2025-06-30", "book_anchor": "2025-03-31"},
            # Strict no-look-ahead anchor is fresh at three months, then stale.
            {"month_end": "2024-03-31", "market_cap": 100, "ttm_sales": 20, "ttm_pat": 10, "book_value": 50,
             "quarter_anchor": "2023-12-31", "book_anchor": "2023-03-31"},
            {"month_end": "2024-04-30", "market_cap": 100, "ttm_sales": 20, "ttm_pat": 10, "book_value": 50,
             "quarter_anchor": "2023-12-31", "book_anchor": "2023-03-31"},
        ]
    )
    result, diagnostics = _calculate_stock_valuation_multiples(frame)

    assert result.loc[0, "ps"] == 0.5
    assert np.isnan(result.loc[0, "pe"])
    assert result.loc[0, "pb"] == 10 / 7
    assert result.loc[1, ["ps", "pe", "pb"]].tolist() == [30.0, 200.0, 40.0]
    assert result.loc[2, "ps"] == 5.0
    assert np.isnan(result.loc[3, "ps"])
    assert diagnostics["P/S"]["stale"] == 1
    assert diagnostics["P/E"]["non_positive"] == 1


def test_stock_multiples_reject_invalid_market_cap_and_stale_book_value():
    frame = pd.DataFrame(
        [{"month_end": "2025-03-31", "market_cap": 0, "ttm_sales": 10, "ttm_pat": 10, "book_value": 10,
          "quarter_anchor": "2024-12-31", "book_anchor": "2024-02-28"}]
    )
    result, diagnostics = _calculate_stock_valuation_multiples(frame)
    assert result[["ps", "pe", "pb"]].isna().all(axis=None)
    assert diagnostics["P/B"]["stale"] == 1
    assert diagnostics["P/S"]["invalid_market_cap"] == 1


def test_ace_refresh_reads_ace_sources_and_preserves_alignment(monkeypatch):
    from features.housekeeping import jobs

    class Engine:
        def begin(self):
            class Context:
                def __enter__(self):
                    return object()

                def __exit__(self, *args):
                    return False

            return Context()

    seen_queries = []

    def read_sql(query, _conn, params):
        sql = str(query)
        seen_queries.append(sql)
        if "FROM fundlab.stock_price_ace" in sql:
            return pd.DataFrame({
                "ace_code": ["288853", "288853"],
                "month_end": ["2024-03-31", "2024-04-30"],
                "market_cap": [100, 100],
            })
        if "FROM fundlab.stock_quarterly_financials_ace" in sql:
            return pd.DataFrame({
                "ace_code": ["288853"] * 4,
                "period_end": ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
                "sales": [20, 20, 20, 20],
                "pat": [10, 10, 10, 10],
            })
        if "FROM fundlab.stock_annual_bv_ace" in sql:
            return pd.DataFrame({
                "ace_code": ["288853"], "year_end": ["2023-03-31"], "book_value": [50]
            })
        raise AssertionError(f"Unexpected SQL: {sql}")

    monkeypatch.setattr(jobs, "get_engine", lambda: Engine())
    monkeypatch.setattr(jobs.pd, "read_sql", read_sql)
    for method in ("write", "info", "warning"):
        monkeypatch.setattr(jobs.st, method, lambda *args, **kwargs: None)

    result = jobs.rebuild_stock_monthly_valuations(
        start_date=dt.date(2024, 3, 31),
        end_date=dt.date(2024, 4, 30),
        write_to_db=False,
    )

    assert len(seen_queries) == 3
    assert result.columns.tolist() == [
        "ace_code", "month_end", "ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"
    ]
    assert result.loc[0, ["ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]].tolist() == [
        80.0, 40.0, 50.0, 1.25, 5.0, 2.0
    ]
    assert result.loc[1, ["ps", "pe", "pb"]].isna().all()


def test_incremental_ace_refresh_uses_latest_dates_and_seed_fundamentals(monkeypatch):
    from features.housekeeping import jobs

    class Engine:
        def begin(self):
            class Context:
                def __enter__(self):
                    return object()

                def __exit__(self, *args):
                    return False

            return Context()

    queries = []

    def read_sql(query, _conn, params=None):
        sql = str(query)
        queries.append(sql)
        if "MAX(month_end) AS latest_month_end" in sql and "LEFT JOIN" not in sql:
            return pd.DataFrame({
                "ace_code": ["288853"], "latest_month_end": [dt.date(2024, 3, 31)]
            })
        if "FROM fundlab.stock_price_ace p" in sql:
            return pd.DataFrame({
                "ace_code": ["288853"], "month_end": ["2024-04-30"], "market_cap": [100]
            })
        if "FROM fundlab.stock_quarterly_financials_ace q" in sql:
            assert params["first_months"] == [dt.date(2024, 4, 30)]
            return pd.DataFrame({
                "ace_code": ["288853"] * 4,
                "period_end": ["2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"],
                "sales": [20] * 4, "pat": [10] * 4,
            })
        if "FROM fundlab.stock_annual_bv_ace b" in sql:
            return pd.DataFrame({
                "ace_code": ["288853"], "year_end": ["2024-03-31"], "book_value": [50]
            })
        raise AssertionError(f"Unexpected SQL: {sql}")

    monkeypatch.setattr(jobs, "get_engine", lambda: Engine())
    monkeypatch.setattr(jobs.pd, "read_sql", read_sql)
    for method in ("write", "info", "warning"):
        monkeypatch.setattr(jobs.st, method, lambda *args, **kwargs: None)

    result = jobs.rebuild_stock_monthly_valuations(
        start_date=dt.date(2024, 4, 1),
        end_date=dt.date(2024, 4, 30),
        write_to_db=False,
        refresh_mode="incremental",
    )
    assert len(queries) == 4
    assert "p.price_date > l.latest_month_end" in queries[1]
    assert "rn <= 4" in queries[2]
    assert "rn = 1" in queries[3]
    assert result.loc[0, ["ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]].tolist() == [
        80.0, 40.0, 50.0, 1.25, 5.0, 2.0
    ]


def test_incremental_ace_refresh_stops_when_no_new_price_months(monkeypatch):
    from features.housekeeping import jobs

    class Engine:
        def begin(self):
            class Context:
                def __enter__(self):
                    return object()

                def __exit__(self, *args):
                    return False

            return Context()

    seen = []

    def read_sql(query, _conn, params=None):
        seen.append(str(query))
        if "MAX(month_end) AS latest_month_end" in str(query):
            return pd.DataFrame({
                "ace_code": ["288853"], "latest_month_end": [dt.date(2024, 4, 30)]
            })
        if "FROM fundlab.stock_price_ace p" in str(query):
            return pd.DataFrame(columns=["ace_code", "month_end", "market_cap"])
        raise AssertionError("Fundamentals should not be fetched without new prices")

    monkeypatch.setattr(jobs, "get_engine", lambda: Engine())
    monkeypatch.setattr(jobs.pd, "read_sql", read_sql)
    for method in ("write", "info"):
        monkeypatch.setattr(jobs.st, method, lambda *args, **kwargs: None)

    result = jobs.rebuild_stock_monthly_valuations(
        start_date=dt.date(2024, 4, 1),
        end_date=dt.date(2024, 4, 30),
        refresh_mode="incremental",
    )
    assert result is None
    assert len(seen) == 2
