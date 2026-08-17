from datetime import date
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.portfolio_quality.compute import (
    QUARTILE_MODE,
    ROC_MODE,
    compute_quality_analysis,
)


def _portfolio_rows():
    return pd.DataFrame([
        {"fund_id": 1, "fund_name": "Focus", "month_end": date(2024, 3, 31), "isin": "FIN", "weight_pct": 60, "asset_type": "Domestic Equities", "is_financial": True},
        {"fund_id": 1, "fund_name": "Focus", "month_end": date(2024, 3, 31), "isin": "NON", "weight_pct": 40, "asset_type": "Domestic Equities", "is_financial": False},
        {"fund_id": 2, "fund_name": "Peer", "month_end": date(2024, 3, 31), "isin": "FIN", "weight_pct": 100, "asset_type": "Domestic Equities", "is_financial": True},
    ])


def _metrics():
    history = pd.DataFrame({"year_end_date": [date(2020, 3, 31)] * 2, "roe": [10.0, 20.0], "roce": [30.0, 40.0]})
    return {"FIN": history.iloc[[0]], "NON": history.iloc[[1]]}


def test_return_on_capital_computes_focus_and_universe_median():
    result = compute_quality_analysis(
        ROC_MODE, _portfolio_rows(), _metrics(), None,
        {"selected_fund_ids": [1, 2], "focus_fund_id": 1,
         "segment_choice": "Total", "comparison_mode": "Universe median"},
    )
    values = dict(zip(result["fund_name"], result["metric"]))
    assert values["Focus"] == 22.0
    assert values["Universe median (others)"] == 10.0


def test_quartile_exposure_rebases_domestic_equity_and_adds_total():
    buckets = pd.DataFrame({
        "month_end": [date(2024, 3, 31), date(2024, 3, 31)],
        "holding_weight": [30.0, 70.0], "quality_quartile": ["Q1", "Q4"],
    })
    result = compute_quality_analysis(
        QUARTILE_MODE, _portfolio_rows(), _metrics(), buckets,
        {"selected_fund_ids": [1, 2], "focus_fund_id": 1, "segment_choice": "Total"},
    )
    assert result.loc["Q1", "Mar 2024"] == 30.0
    assert result.loc["Q4", "Mar 2024"] == 70.0
    assert result.loc["Total", "Mar 2024"] == 100.0
