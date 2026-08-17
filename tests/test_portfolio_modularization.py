from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.portfolio.compute import compute_active_share


def test_compute_active_share_for_partially_overlapping_portfolios():
    holdings = pd.DataFrame(
        [
            {"fund_id": 1, "month_end": "2025-01-31", "isin": "A", "weight_pct": 60},
            {"fund_id": 1, "month_end": "2025-01-31", "isin": "B", "weight_pct": 40},
            {"fund_id": 2, "month_end": "2025-01-31", "isin": "A", "weight_pct": 20},
            {"fund_id": 2, "month_end": "2025-01-31", "isin": "C", "weight_pct": 80},
        ]
    )
    result = compute_active_share(holdings, {1: 1.0}, {2: 1.0})
    assert result.loc[0, "period_label"] == "Jan 2025"
    assert result.loc[0, "active_share_pct"] == 80.0


def test_compute_active_share_empty_input_has_schema():
    result = compute_active_share(pd.DataFrame(), {1: 1.0}, {2: 1.0})
    assert list(result.columns) == ["period_date", "period_label", "active_share_pct"]
