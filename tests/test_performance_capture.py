import datetime as dt
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.performance_returns import (
    build_up_down_capture_observations,
    build_up_down_capture_table,
    make_up_down_capture_data,
    make_up_down_capture_table,
    prepare_capture_bucket_table,
)
from services.performance_plots import (
    plot_up_down_capture_ranked_returns,
)


def _stored_returns(benchmark_returns, fund_returns):
    dates = pd.date_range("2024-01-31", periods=len(benchmark_returns), freq="ME")
    bench = pd.DataFrame(
        {
            "bench_name": "Benchmark",
            "asof_date": dates,
            "rolling_cagr": benchmark_returns,
        }
    )
    fund = pd.DataFrame(
        {
            "fund_name": "Focus",
            "asof_date": dates,
            "rolling_cagr": fund_returns,
        }
    )
    return fund, bench


def test_capture_uses_benchmark_quartiles_start_year_and_median_of_capped_ratios():
    fund, bench = _stored_returns(
        [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
        [-4.0, -0.3, -0.4, -0.1, 0.1, 0.4, 0.3, 4.0],
    )

    result = build_up_down_capture_table(
        fund,
        bench,
        focus_fund="Focus",
        benchmark_name="Benchmark",
        start_domain=dt.date(2023, 1, 31),
        end_domain=dt.date(2024, 8, 31),
    )

    top = result[result["Benchmark bucket"] == "Top quartile"].iloc[0]
    bottom = result[result["Benchmark bucket"] == "Bottom quartile"].iloc[0]
    assert top["Starting year"] == 2023
    assert top["Average benchmark return (%)"] == 35.0
    assert top["Average focus fund return (%)"] == 215.0
    assert top["Median capture (%)"] == 300.0
    assert top["Observations"] == 2
    assert bottom["Median capture (%)"] == 300.0
    assert bottom["Observations"] == 2


def test_zero_benchmark_return_is_excluded_from_capture_observation_count():
    fund, bench = _stored_returns(
        [-0.4, -0.3, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4],
        [-0.4, -0.3, -0.2, 0.2, 0.1, 0.2, 0.3, 0.4],
    )
    result = build_up_down_capture_table(
        fund,
        bench,
        focus_fund="Focus",
        benchmark_name="Benchmark",
        start_domain=dt.date(2023, 1, 31),
        end_domain=dt.date(2024, 8, 31),
    )
    assert int(result["Observations"].sum()) == 7
    assert np.isfinite(result["Median capture (%)"].dropna()).all()


def test_bucket_table_is_ranked_and_ends_with_column_averages():
    capture_table = pd.DataFrame(
        {
            "Benchmark bucket": ["Top quartile", "Top quartile", "Bottom quartile"],
            "Starting year": [2022, 2023, 2021],
            "Average benchmark return (%)": [15.0, 25.0, -10.0],
            "Average focus fund return (%)": [18.0, 20.0, -8.0],
            "Median capture (%)": [120.0, 80.0, 80.0],
            "Observations": [2, 4, 3],
        }
    )

    result = prepare_capture_bucket_table(capture_table, "Top quartile")

    assert result["Starting year"].tolist() == [2023, 2022, "Average"]
    average = result.iloc[-1]
    assert average["Average benchmark return (%)"] == 20.0
    assert average["Average focus fund return (%)"] == 19.0
    assert average["Median capture (%)"] == 100.0
    assert average["Observations"] == 3.0


def test_ranked_return_lines_preserve_benchmark_order_and_fund_alignment():
    observations = pd.DataFrame(
        {
            "Benchmark quartile": [
                "Bottom quartile", "Top quartile", "Third quartile", "Second quartile",
                "Top quartile", "Bottom quartile", "Second quartile", "Third quartile",
            ],
            "Starting year": [2021, 2023, 2022, 2021, 2020, 2020, 2022, 2023],
            "asof_date": pd.to_datetime(
                ["2022-01-31", "2024-01-31", "2023-01-31",
                 "2022-02-28", "2021-03-31", "2021-04-30",
                 "2023-02-28", "2024-03-31"]
            ),
            "Benchmark return (%)": [-10.0, 15.0, 2.0, 8.0, 25.0, -20.0, 12.0, 5.0],
            "Focus fund return (%)": [-8.0, 18.0, 1.0, 9.0, 20.0, -25.0, 13.0, 4.0],
        }
    )
    observations["Benchmark quartile"] = observations["Benchmark quartile"].astype(
        pd.CategoricalDtype(
            ["Top quartile", "Second quartile", "Third quartile", "Bottom quartile"],
            ordered=True,
        )
    )

    fig = plot_up_down_capture_ranked_returns(observations, "Focus", "Benchmark")

    assert len(fig.data) == 8
    assert [trace.mode for trace in fig.data] == ["lines"] * 8
    assert fig.layout.xaxis.showticklabels is False
    assert [annotation.text for annotation in fig.layout.annotations] == [
        "Top quartile<br>Hit-rate: 50.0%",
        "Second quartile<br>Hit-rate: 100.0%",
        "Third quartile<br>Hit-rate: 0.0%",
        "Bottom quartile<br>Hit-rate: 50.0%",
    ]
    assert [list(trace.x) for trace in fig.data[::2]] == [[0, 1], [4, 5], [8, 9], [12, 13]]
    assert [list(trace.y) for trace in fig.data[::2]] == [
        [25.0, 15.0], [12.0, 8.0], [5.0, 2.0], [-10.0, -20.0]
    ]
    assert [list(trace.y) for trace in fig.data[1::2]] == [
        [20.0, 18.0], [13.0, 9.0], [4.0, 1.0], [-8.0, -25.0]
    ]


def test_capture_observations_keep_each_matched_rolling_return():
    fund, bench = _stored_returns(
        [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
        [-0.5, -0.3, -0.4, -0.1, 0.1, 0.4, 0.3, 0.5],
    )
    observations = build_up_down_capture_observations(
        fund, bench,
        focus_fund="Focus", benchmark_name="Benchmark",
        start_domain=dt.date(2023, 1, 31), end_domain=dt.date(2024, 8, 31),
    )

    assert len(observations) == 8
    assert observations["Benchmark return (%)"].iloc[0] == -40.0
    assert observations["Focus fund return (%)"].iloc[-1] == 50.0
    assert observations["Benchmark bucket"].value_counts()["Top quartile"] == 2
    assert observations["Benchmark quartile"].tolist() == [
        "Bottom quartile", "Bottom quartile", "Third quartile", "Third quartile",
        "Second quartile", "Second quartile", "Top quartile", "Top quartile",
    ]


def test_capture_data_fetches_once_for_both_charts_and_table(monkeypatch):
    from services import performance_returns

    fund, bench = _stored_returns(
        [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
        [-0.5, -0.3, -0.4, -0.1, 0.1, 0.4, 0.3, 0.5],
    )
    calls = []

    def fund_loader(**kwargs):
        calls.append("fund")
        return fund

    def bench_loader(**kwargs):
        calls.append("bench")
        return bench

    monkeypatch.setattr(performance_returns, "load_fund_rolling", fund_loader)
    monkeypatch.setattr(performance_returns, "load_bench_rolling", bench_loader)
    summary, observations = make_up_down_capture_data(
        "Focus", "Benchmark", dt.date(2023, 1, 31), dt.date(2024, 8, 31)
    )

    assert calls == ["fund", "bench"]
    assert len(observations) == 8
    assert summary["Observations"].sum() == 8


def test_capture_wrapper_fetches_precomputed_12_month_returns(monkeypatch):
    from services import performance_returns

    calls = []
    fund_frame = pd.DataFrame({"kind": ["fund"]})
    bench_frame = pd.DataFrame({"kind": ["bench"]})
    expected = pd.DataFrame({"result": [1]})

    def fake_fund_loader(**kwargs):
        calls.append(("fund", kwargs))
        return fund_frame

    def fake_bench_loader(**kwargs):
        calls.append(("bench", kwargs))
        return bench_frame

    monkeypatch.setattr(performance_returns, "load_fund_rolling", fake_fund_loader)
    monkeypatch.setattr(performance_returns, "load_bench_rolling", fake_bench_loader)
    monkeypatch.setattr(
        performance_returns,
        "build_up_down_capture_table",
        lambda fund, bench, **kwargs: expected,
    )

    result = make_up_down_capture_table(
        "Focus", "Benchmark", dt.date(2023, 1, 31), dt.date(2024, 12, 31)
    )
    assert result is expected
    assert calls[0][1]["window_months"] == 12
    assert calls[1][1]["window_months"] == 12
    assert calls[0][1]["start"] == pd.Timestamp("2024-01-31")
    assert calls[0][1]["end"] == pd.Timestamp("2024-12-31")
