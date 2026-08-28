import numpy as np
import pandas as pd
import sys
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
