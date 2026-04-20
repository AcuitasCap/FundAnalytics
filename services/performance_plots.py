import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.performance_returns import window_label_series


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
