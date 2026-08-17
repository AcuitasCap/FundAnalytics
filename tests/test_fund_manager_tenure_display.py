"""Regression coverage for the tenure history chart renderer."""

import importlib
import sys
from pathlib import Path

import pandas as pd

def test_history_display_renders_a_populated_plotly_timeline(monkeypatch):
    """History mode sends a non-empty date-range timeline to Streamlit."""
    monkeypatch.syspath_prepend(str(Path.cwd()))
    display = importlib.import_module("features.fund_manager_tenure.display")
    rendered, captions = [], []
    monkeypatch.setattr(
        display.st,
        "plotly_chart",
        lambda chart, **kwargs: rendered.append((chart, kwargs)),
    )
    monkeypatch.setattr(display.st, "caption", captions.append)

    timeline = pd.DataFrame(
        {
            "fund_manager": ["Former", "Current"],
            "from_date": ["2020-01-01", "2022-01-01"],
            "to_date_filled": ["2021-12-31", "2025-01-01"],
            "stint_is_current": [False, True],
        }
    )
    current = pd.DataFrame({"fund_manager": ["Current"], "tenure_years": [3.0]})

    display.display_tenure_history(timeline, current, pd.Timestamp("2021-12-31"))

    assert len(rendered) == 1
    assert rendered[0][1] == {"use_container_width": True}
    chart = rendered[0][0]
    assert chart.layout.height >= 240
    assert len(chart.layout.shapes) == 2
    assert {(shape.x0, shape.x1) for shape in chart.layout.shapes} == {
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01")),
    }
    assert {shape.fillcolor for shape in chart.layout.shapes} == {"#c7c7c7", "#1f77b4"}
    range_start, range_end = map(pd.Timestamp, chart.layout.xaxis.range)
    assert range_start <= pd.Timestamp("2020-01-01")
    assert range_end >= pd.Timestamp("2025-01-01")
    assert len(chart.data) == 1
    hover_trace = chart.data[0]
    assert len(hover_trace.x) == len(hover_trace.y) == 2
    assert {row[3] for row in hover_trace.customdata} == {"Prior", "Current"}
    assert all(row[1] < row[2] for row in hover_trace.customdata)
    assert "%{customdata[1]|%b-%Y}" in hover_trace.hovertemplate
    assert "Current fund manager: Current managing since the past 3.0 years" in captions
    assert "Last tenure update on Supabase: Dec-2021" in captions
    assert "altair" not in display._build_tenure_chart.__code__.co_names
