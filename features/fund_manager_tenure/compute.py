"""Pure transformations for fund manager tenure."""

import pandas as pd


def prepare_focus_fund_timeline(
    tenure_rows: pd.DataFrame, focus_fund_id: int, today: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    """Prepare chart stints, current-manager summary, and last recorded update.

    Open-ended stints are current.  If none are open-ended, stints ending on the
    latest recorded date are treated as current and extended to ``today``.
    """
    today = (today or pd.Timestamp.today()).normalize()
    timeline = tenure_rows[tenure_rows["fund_id"] == focus_fund_id].copy()
    if timeline.empty:
        return timeline, pd.DataFrame(columns=["fund_manager", "tenure_years"]), None

    timeline["from_date"] = pd.to_datetime(timeline["from_date"])
    timeline["to_date"] = pd.to_datetime(timeline["to_date"])
    last_update = timeline["to_date"].dropna().max()
    has_open_ended = timeline["to_date"].isna().any()
    timeline["stint_is_current"] = (
        timeline["to_date"].isna()
        if has_open_ended
        else timeline["to_date"].eq(last_update)
    )
    timeline["to_date_filled"] = timeline["to_date"].fillna(today)
    if not has_open_ended:
        timeline.loc[timeline["stint_is_current"], "to_date_filled"] = today

    current = timeline[timeline["stint_is_current"]].copy()
    current["tenure_years"] = (today - current["from_date"]).dt.days / 365.25
    current = (
        current.groupby("fund_manager", as_index=False)
        .agg(from_date=("from_date", "min"), tenure_years=("tenure_years", "max"))
        .sort_values("tenure_years", ascending=False)
    )
    current["tenure_years"] = current["tenure_years"].round(1)
    return timeline.sort_values(["from_date", "to_date_filled", "fund_manager"]), current, last_update


def compute_tenure_filter(
    tenure_rows: pd.DataFrame, funds: pd.DataFrame, minimum_years: float, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Return current fund-manager pairs satisfying a tenure threshold."""
    today = (today or pd.Timestamp.today()).normalize()
    d = tenure_rows.copy()
    d["from_date"] = pd.to_datetime(d["from_date"])
    d["to_date"] = pd.to_datetime(d["to_date"])
    has_open = d.groupby("fund_id")["to_date"].transform(lambda values: values.isna().any())
    latest_to = d.groupby("fund_id")["to_date"].transform(lambda values: values.dropna().max())
    current = (has_open & d["to_date"].isna()) | (~has_open & d["to_date"].eq(latest_to))
    out = d.loc[current].copy()
    out["tenure_years"] = (today - out["from_date"]).dt.days / 365.25
    out = out[out["tenure_years"] >= minimum_years].copy()
    out["tenure_years"] = out["tenure_years"].round(1)
    out = out.merge(funds[["fund_id", "fund_name"]], on="fund_id", how="left").dropna(subset=["fund_name"])
    return out[["fund_name", "fund_manager", "tenure_years"]].sort_values(
        ["fund_name", "tenure_years"], ascending=[True, False]
    ).reset_index(drop=True)
