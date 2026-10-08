import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.performance_returns import CAPTURE_BUCKETS, CAPTURE_QUARTILES, window_label_series


def _capture_bucket_positions(plot_df, bucket_order=CAPTURE_BUCKETS):
    """Assign fixed x positions so each bucket keeps its supplied row order."""
    positioned = []
    spans = []
    next_position = 0
    for bucket in bucket_order:
        group = plot_df.loc[plot_df["Benchmark bucket"] == bucket].copy()
        if group.empty:
            continue
        group["x_position"] = range(next_position, next_position + len(group))
        spans.append((bucket, next_position, next_position + len(group) - 1))
        positioned.append(group)
        next_position += len(group) + 2
    return pd.concat(positioned, ignore_index=True), spans


def _style_capture_bucket_axis(fig, spans, *, years=None, hit_rates=None):
    for index, (bucket, first, last) in enumerate(spans):
        if index:
            fig.add_vline(x=first - 1.5, line_color="#d1d5db", line_dash="dot")
        label = "Middle quartiles" if bucket == "Middle two quartiles" else bucket
        if hit_rates is not None:
            label += f"<br>Hit-rate: {hit_rates[bucket]:.1f}%"
        fig.add_annotation(
            x=(first + last) / 2,
            y=-0.20 if years is not None else -0.10,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=12),
        )
    axis_options = dict(
        type="linear",
        range=[spans[0][1] - 0.5, spans[-1][2] + 0.5],
        showgrid=False,
        zeroline=False,
    )
    if years is None:
        axis_options.update(showticklabels=False, ticks="")
    else:
        axis_options.update(
            tickmode="array",
            tickvals=years["x_position"].tolist(),
            ticktext=years["Starting year"].astype(int).astype(str).tolist(),
            tickangle=0,
        )
    fig.update_xaxes(**axis_options)


def plot_up_down_capture(capture_table, focus_name, benchmark_name):
    """Plot yearly average returns, ranked within each benchmark quartile."""
    columns = [
        "Benchmark bucket",
        "Starting year",
        "Average benchmark return (%)",
        "Average focus fund return (%)",
    ]
    if capture_table is None or capture_table.empty or not set(columns).issubset(capture_table.columns):
        return None

    plot_df = capture_table.loc[:, columns].copy()
    plot_df["Benchmark bucket"] = plot_df["Benchmark bucket"].astype(str)
    plot_df = plot_df[plot_df["Benchmark bucket"].isin(CAPTURE_BUCKETS)].copy()
    for column in ["Starting year", "Average benchmark return (%)", "Average focus fund return (%)"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["Starting year", "Average benchmark return (%)"])
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(
        ["Average benchmark return (%)", "Starting year"],
        ascending=[False, True],
    )
    plot_df, spans = _capture_bucket_positions(plot_df)
    bucket_labels = [
        "Middle quartiles" if bucket == "Middle two quartiles" else bucket
        for bucket in plot_df["Benchmark bucket"]
    ]
    year_labels = plot_df["Starting year"].astype(int).astype(str).tolist()
    hover_data = list(zip(bucket_labels, year_labels))

    fig = go.Figure()
    for name, column, color in [
        (benchmark_name, "Average benchmark return (%)", "#f28e2b"),
        (focus_name, "Average focus fund return (%)", "#1f77b4"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=plot_df["x_position"].tolist(),
                y=plot_df[column].tolist(),
                customdata=hover_data,
                mode="markers",
                name=name,
                marker=dict(color=color, size=11),
                hovertemplate=(
                    "%{customdata[0]} · %{customdata[1]}<br>"
                    "Average return: %{y:.1f}%<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=30, t=45, b=115),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
    )
    _style_capture_bucket_axis(fig, spans, years=plot_df)
    fig.update_yaxes(title_text="Average rolling 1-year return (%)", ticksuffix="%", showgrid=True)
    return fig


def plot_up_down_capture_ranked_returns(observations, focus_name, benchmark_name):
    """Plot individual matched returns as lines ordered by benchmark return."""
    columns = [
        "Benchmark quartile",
        "Starting year",
        "asof_date",
        "Benchmark return (%)",
        "Focus fund return (%)",
    ]
    if observations is None or observations.empty or not set(columns).issubset(observations.columns):
        return None

    plot_df = observations.loc[:, columns].rename(
        columns={"Benchmark quartile": "Benchmark bucket"}
    ).copy()
    plot_df["Benchmark bucket"] = plot_df["Benchmark bucket"].astype(str)
    plot_df = plot_df[plot_df["Benchmark bucket"].isin(CAPTURE_QUARTILES)].copy()
    for column in ["Benchmark return (%)", "Focus fund return (%)"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df["asof_date"] = pd.to_datetime(plot_df["asof_date"], errors="coerce")
    plot_df = plot_df.dropna(
        subset=["asof_date", "Benchmark return (%)", "Focus fund return (%)"]
    )
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(
        ["Benchmark return (%)", "asof_date"], ascending=[False, True]
    )
    plot_df, spans = _capture_bucket_positions(plot_df, CAPTURE_QUARTILES)
    hit_rates = {
        bucket: 100.0 * (group["Focus fund return (%)"] > group["Benchmark return (%)"]).mean()
        for bucket, group in plot_df.groupby("Benchmark bucket")
    }

    fig = go.Figure()
    for bucket_index, (bucket, _, _) in enumerate(spans):
        group = plot_df.loc[plot_df["Benchmark bucket"] == bucket]
        hover_data = list(zip(
            group["Starting year"].astype(int).astype(str),
            group["asof_date"].dt.strftime("%b %Y"),
        ))
        for name, column, color in [
            (benchmark_name, "Benchmark return (%)", "#f28e2b"),
            (focus_name, "Focus fund return (%)", "#1f77b4"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=group["x_position"].tolist(),
                    y=group[column].tolist(),
                    customdata=hover_data,
                    mode="lines" if len(group) > 1 else "markers",
                    name=name,
                    legendgroup=name,
                    showlegend=bucket_index == 0,
                    line=dict(color=color, width=2.5),
                    marker=dict(color=color, size=8),
                    hovertemplate=(
                        f"{bucket}<br>Starting year %{{customdata[0]}}<br>"
                        "Rolling period ending %{customdata[1]}<br>"
                        "Return: %{y:.1f}%<extra>%{fullData.name}</extra>"
                    ),
                )
            )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=30, t=45, b=105),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
    )
    _style_capture_bucket_axis(fig, spans, hit_rates=hit_rates)
    fig.update_yaxes(title_text="Rolling 1-year return (%)", ticksuffix="%", showgrid=True)
    return fig


def plot_rolling(df, months, focus_name, bench_label, chart_height=560, include_cols=None):
    if df.empty:
        return None

    labels = window_label_series(df.index, months)
    plot_df = df.copy()
    plot_df["Window"] = labels.values
    plot_df = plot_df.reset_index(drop=True)

    bench_col = None
    if bench_label:
        if bench_label in plot_df.columns:
            bench_col = bench_label
        elif "Benchmark" in plot_df.columns:
            plot_df = plot_df.rename(columns={"Benchmark": bench_label})
            bench_col = bench_label

    default_cols = []
    if focus_name in plot_df.columns:
        default_cols.append(focus_name)
    if "Peer avg" in plot_df.columns:
        default_cols.append("Peer avg")
    if bench_col:
        default_cols.append(bench_col)

    ycols = [c for c in include_cols if c in plot_df.columns] if include_cols else default_cols
    if not ycols:
        return None

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet + px.colors.qualitative.Bold
    color_map = {}
    if focus_name in ycols:
        color_map[focus_name] = "#000000"
    if bench_col and bench_col in ycols:
        color_map[bench_col] = "#d62728"
    if "Peer avg" in ycols:
        color_map["Peer avg"] = "#1f77b4"

    palette_idx = 0
    for col in ycols:
        if col not in color_map:
            color_map[col] = palette[palette_idx % len(palette)]
            palette_idx += 1

    fig = px.line(
        plot_df,
        x="Window",
        y=ycols,
        labels={"value": "Return (%)", "Window": "Rolling window (start-end)"},
        title=f"{months // 12}Y Rolling CAGR",
        color_discrete_map=color_map,
    )

    for tr in fig.data:
        tr.update(line=dict(width=4 if tr.name == focus_name else 3))

    fig.update_layout(
        height=chart_height,
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        hovermode="x unified",
    )

    n = len(plot_df["Window"])
    tickvals = plot_df["Window"].tolist() if n <= 12 else plot_df["Window"].tolist()[:: math.ceil(n / 12)]
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals,
        tickangle=-20,
        tickfont=dict(size=12),
        categoryorder="array",
        categoryarray=plot_df["Window"].tolist(),
        showgrid=True,
    )
    fig.update_yaxes(tickformat=".2f", ticksuffix="%", showgrid=True)
    return fig


def plot_multi_fund_rolling(df, months, focus_name=None, chart_height=560):
    if df.empty:
        return None

    labels = window_label_series(df.index, months)
    plot_df = df.copy()
    plot_df["Window"] = labels.values
    plot_df = plot_df.reset_index(drop=True)
    series_cols = [c for c in plot_df.columns if c != "Window"]
    if not series_cols:
        return None

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet + px.colors.qualitative.Bold
    color_map = {}
    if focus_name and focus_name in series_cols:
        color_map[focus_name] = "#000000"

    palette_idx = 0
    for col in series_cols:
        if col not in color_map:
            color_map[col] = palette[palette_idx % len(palette)]
            palette_idx += 1

    fig = px.line(
        plot_df,
        x="Window",
        y=series_cols,
        labels={"value": "Return (%)", "Window": "Rolling window (start-end)"},
        title=f"{months // 12}Y Rolling CAGR - Multiple funds",
        color_discrete_map=color_map,
    )

    for tr in fig.data:
        tr.update(line=dict(width=4 if focus_name and tr.name == focus_name else 3))

    fig.update_layout(
        height=chart_height,
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        hovermode="x unified",
    )

    n = len(plot_df["Window"])
    tickvals = plot_df["Window"].tolist() if n <= 12 else plot_df["Window"].tolist()[:: math.ceil(n / 12)]
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals,
        tickangle=-20,
        tickfont=dict(size=12),
        categoryorder="array",
        categoryarray=plot_df["Window"].tolist(),
        showgrid=True,
    )
    fig.update_yaxes(tickformat=".2f", ticksuffix="%", showgrid=True)
    return fig


def style_relative_multi_horizon(df: pd.DataFrame):
    df2 = df.copy()
    for c in df2.columns:
        if c != "Fund" and not pd.api.types.is_numeric_dtype(df2[c]):
            try:
                tmp = pd.to_numeric(df2[c], errors="coerce")
                if tmp.notna().any():
                    df2[c] = tmp
            except Exception:
                pass

    num_cols = [c for c in df2.columns if c != "Fund" and pd.api.types.is_numeric_dtype(df2[c])]

    def rel_colors(dfin: pd.DataFrame):
        styles = pd.DataFrame("", index=dfin.index, columns=dfin.columns)
        for c in num_cols:
            styles[c] = dfin[c].apply(
                lambda v: "background-color:#e6f4ea;color:#0b8043"
                if pd.notna(v) and v > 0
                else "background-color:#fdecea;color:#a50e0e"
                if pd.notna(v) and v < 0
                else ""
            )
        return styles

    sty = df2.style.apply(rel_colors, axis=None)
    fmt_map = {c: "{:.2f}" for c in num_cols}
    return sty.format(fmt_map, na_rep="—").set_table_styles(
        [
            {"selector": "table", "props": "table-layout:fixed"},
            {"selector": "th.col_heading", "props": "white-space:normal; line-height:1.1; height:56px"},
        ]
    )


def df_to_table_figure(df: pd.DataFrame, title: str, fill=None):
    if df.empty:
        return None
    df_print = df.copy().reset_index()
    headers = list(df_print.columns)
    cells = [df_print[c].astype(object).astype(str).tolist() for c in headers]

    if fill is None:
        cell_fill = "white"
    elif isinstance(fill, list) and fill and isinstance(fill[0], list):
        cell_fill = fill
    else:
        cell_fill = "white"

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, fill_color="#f0f0f0", align="left", font=dict(size=12, color="black")),
                cells=dict(values=cells, align="left", fill_color=cell_fill, font=dict(size=11, color="black")),
            )
        ]
    )
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=20, r=20, t=60, b=20), height=560)
    return fig


def build_rel_fill(df_with_fund_col: pd.DataFrame, fund_col="Fund", misaligned=None):
    misaligned = set(misaligned or [])
    if fund_col not in df_with_fund_col.columns:
        tmp = df_with_fund_col.copy()
        tmp.insert(0, fund_col, tmp.index)
        df_with_fund_col = tmp

    fills = []
    for c in df_with_fund_col.columns.tolist():
        col_fill = []
        for v in df_with_fund_col[c].tolist():
            if c == fund_col:
                col_fill.append("#FFF59D" if v in misaligned else "white")
            elif v is None or (isinstance(v, str) and v.strip() == ""):
                col_fill.append("white")
            else:
                try:
                    fv = float(v)
                    if fv > 0:
                        col_fill.append("#e6f4ea")
                    elif fv < 0:
                        col_fill.append("#fdecea")
                    else:
                        col_fill.append("white")
                except Exception:
                    col_fill.append("white")
        fills.append(col_fill)
    return fills
