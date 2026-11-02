"""Exercise shared selections and independent section results without Supabase."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from features.fundamental_attributes import page
from features.fundamental_attributes import data
from features.portfolio_quality import page as quality_page


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(page, "fetch_categories", lambda: ["Large", "Mid"])
    monkeypatch.setattr(page, "fetch_funds_for_categories", lambda categories: pd.DataFrame({
        "fund_id": [1, 2, 3], "fund_name": ["Focus", "Peer", "Other"], "category_name": ["Large", "Large", "Mid"]}))

    def forbid_database():
        raise AssertionError("UI tests must not use production data")

    monkeypatch.setattr(data, "get_engine", forbid_database)
    captured = []

    def inputs(fund_ids, start, end, mode, compare, kind):
        captured.append((fund_ids, start, end, mode, compare, kind))
        holdings = pd.DataFrame([{"fund_id": fund_id, "holding_month_end": pd.Timestamp("2024-03-31"),
                                  "anchor_month_end": pd.Timestamp("2024-03-31"), "holding_key": f"{fund_id}-{index}",
                                  "ace_code": f"{index:05d}", "weight_pct": 100 / 6, "is_financial": False}
                                 for fund_id in fund_ids for index in range(6)])
        if kind == "valuation":
            facts = pd.DataFrame([{"ace_code": f"{index:05d}", "month_end": pd.Timestamp("2024-03-31"),
                                   "ps": index + 1, "pb": index + 1, "pe": index + 10} for index in range(6)])
        else:
            facts = pd.DataFrame([{"ace_code": f"{index:05d}", "period_end": quarter.to_timestamp("Q"),
                                   "sales": 100 * 1.1 ** (number // 4), "pat": 10 * 1.2 ** (number // 4)}
                                  for index in range(6)
                                  for number, quarter in enumerate(pd.period_range("2020Q1", "2023Q4", freq="Q"))])
        members = pd.DataFrame([{"ace_code": f"{index:05d}", "month_end": pd.Timestamp("2024-03-31"),
                                 "is_financial": False} for index in range(6)])
        return holdings, members, facts

    monkeypatch.setattr(page, "cached_attribute_inputs", inputs)
    monkeypatch.setattr(quality_page, "fetch_portfolio_raw", lambda *args: pd.DataFrame({
        "fund_id": [1, 2], "fund_name": ["Focus", "Peer"], "month_end": [pd.Timestamp("2024-03-31").date()] * 2,
        "ace_code": ["00000", "00000"], "weight_pct": [100, 100], "asset_type": ["Domestic Equities"] * 2,
        "is_financial": [False, False]}))
    monkeypatch.setattr(quality_page, "load_stock_roe_roce", lambda: {})
    test = AppTest.from_string("from features.fundamental_attributes.page import fundamental_attributes_page\nfundamental_attributes_page()")
    test.run()
    assert not test.exception
    test.checkbox(key="fa_category_Large").check().run()
    test.multiselect(key="fa_fund_picker").select("All funds in selected categories").run()
    assert not test.exception
    return test, captured


def submit(test, label):
    next(button for button in test.button if button.label == label).click().run()
    assert not test.exception
    assert not test.error


def test_one_shared_selection_and_shortlist_survives_category_changes(app):
    test, _ = app
    assert len(test.multiselect) == 1
    assert [select.label for select in test.selectbox].count("Focus fund") == 1
    assert test.session_state["fa_shortlisted_ids"] == [1, 2]
    test.selectbox(key="fa_focus_picker").select(2).run()
    test.checkbox(key="fa_category_Large").uncheck().run()
    test.checkbox(key="fa_category_Mid").check().run()
    assert test.session_state["fa_shortlisted_ids"] == [1, 2]
    assert test.selectbox(key="fa_focus_picker").value == 2
    assert test.selectbox(key="fa_growth_start_month").options == ["Mar", "Jun", "Sep", "Dec"]
    assert test.radio(key="fa_valuation_compare").options == ["Peers", "Largecaps", "Midcaps", "Smallcaps", "Microcaps", "Top 500"]
    test.button(key="fa_reset").click().run()
    assert test.session_state["fa_shortlisted_ids"] == []


def test_section_results_survive_other_section_submissions_and_use_common_funds(app):
    test, captured = app
    test.radio(key="pq_segment").set_value("Total")
    submit(test, "Update quality")
    assert "fa_quality_result" in test.session_state
    test.radio(key="fa_valuation_segment").set_value("Total")
    submit(test, "Update valuations")
    assert captured[-1][0] == (1, 2)
    assert "fa_quality_result" in test.session_state
    assert len(test.get("arrow_vega_lite_chart")) == 2
    test.radio(key="fa_growth_segment").set_value("Total")
    submit(test, "Update growth")
    assert captured[-1][0] == (1, 2)
    assert len(test.get("arrow_vega_lite_chart")) == 3
    test.radio(key="fa_valuation_compare").set_value("Midcaps")
    submit(test, "Update valuations")
    assert captured[-1][0] == (1,)
    assert len(test.get("arrow_vega_lite_chart")) == 3
    test.selectbox(key="fa_focus_picker").select(2).run()
    assert not test.exception
    assert len(test.get("arrow_vega_lite_chart")) == 0  # Old-focus results must not be shown.


def test_invalid_period_shows_error_only_for_that_section(app):
    test, captured = app
    test.selectbox(key="fa_valuation_start_year").select(2025)
    test.selectbox(key="fa_valuation_end_year").select(2024)
    next(button for button in test.button if button.label == "Update valuations").click().run()
    assert not test.exception
    assert test.error[0].value == "Start date must be on or before end date."
    assert captured == []
    assert test.radio(key="fa_growth_metric").options == ["Sales", "PAT"]
