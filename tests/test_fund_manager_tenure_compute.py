import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.fund_manager_tenure.compute import compute_tenure_filter, prepare_focus_fund_timeline


def _rows():
    return pd.DataFrame(
        {
            "fund_id": [1, 1, 2],
            "fund_manager": ["Former", "Current", "Closed"],
            "from_date": ["2020-01-01", "2022-01-01", "2021-01-01"],
            "to_date": ["2021-12-31", None, "2023-12-31"],
        }
    )


def test_prepare_focus_timeline_marks_open_ended_stint_current():
    timeline, current, last_update = prepare_focus_fund_timeline(_rows(), 1, pd.Timestamp("2025-01-01"))
    assert timeline["stint_is_current"].tolist() == [False, True]
    assert current["fund_manager"].tolist() == ["Current"]
    assert last_update == pd.Timestamp("2021-12-31")


def test_tenure_filter_uses_latest_closed_stint_when_no_open_ended_row():
    funds = pd.DataFrame({"fund_id": [1, 2], "fund_name": ["One", "Two"]})
    result = compute_tenure_filter(_rows(), funds, 1.0, pd.Timestamp("2025-01-01"))
    assert result["fund_manager"].tolist() == ["Current", "Closed"]
