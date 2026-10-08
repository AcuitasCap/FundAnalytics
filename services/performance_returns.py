import re

import numpy as np
import pandas as pd
import streamlit as st

from services.performance_db import load_bench_rolling, load_fund_rolling


def to_eom(series):
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("M")


def coerce_num(series):
    return pd.to_numeric(series, errors="coerce")


def _fy_year_for(date_eom: pd.Timestamp) -> int:
    return date_eom.year + 1 if date_eom.month >= 4 else date_eom.year


def _fy_bounds(y_label: int):
    start = pd.Timestamp(year=y_label - 1, month=4, day=1).to_period("M").to_timestamp("M")
    end = pd.Timestamp(year=y_label, month=3, day=1).to_period("M").to_timestamp("M")
    return start, end


def _cy_bounds(y_label: int):
    start = pd.Timestamp(year=y_label, month=1, day=1).to_period("M").to_timestamp("M")
    end = pd.Timestamp(year=y_label, month=12, day=1).to_period("M").to_timestamp("M")
    return start, end


@st.cache_data(show_spinner=False)
def yearly_returns_with_custom_domain(
    nav_series: pd.Series,
    start_domain: pd.Timestamp,
    end_domain: pd.Timestamp,
    fy: bool = True,
) -> pd.Series:
    s = nav_series.dropna().sort_index()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    s = s.groupby(level=0).last()

    start_domain = pd.Timestamp(start_domain).to_period("M").to_timestamp("M")
    end_domain = pd.Timestamp(end_domain).to_period("M").to_timestamp("M")
    if s.empty or end_domain <= start_domain:
        return pd.Series(dtype=float)

    def fy_year_for(eom: pd.Timestamp) -> int:
        return eom.year + 1 if eom.month >= 4 else eom.year

    def fy_bounds(y_label: int):
        a = pd.Timestamp(year=y_label - 1, month=4, day=1).to_period("M").to_timestamp("M")
        b = pd.Timestamp(year=y_label, month=3, day=1).to_period("M").to_timestamp("M")
        return a, b

    def cy_bounds(y_label: int):
        a = pd.Timestamp(year=y_label, month=1, day=1).to_period("M").to_timestamp("M")
        b = pd.Timestamp(year=y_label, month=12, day=1).to_period("M").to_timestamp("M")
        return a, b

    if fy:
        first_label = fy_year_for(start_domain)
        last_label = fy_year_for(end_domain)
    else:
        first_label = start_domain.year
        last_label = end_domain.year

    out_vals, out_idx = [], []
    for y in range(first_label, last_label + 1):
        yr_start, yr_end = fy_bounds(y) if fy else cy_bounds(y)
        a = max(yr_start, start_domain)
        b = min(yr_end, end_domain)
        if b <= a:
            continue

        baseline = (a.to_period("M") - 1).to_timestamp("M")
        if baseline not in s.index or b not in s.index:
            out_vals.append(np.nan)
        else:
            try:
                ret = s.loc[b] / s.loc[baseline] - 1.0
            except Exception:
                ret = np.nan
            out_vals.append(ret)

        full = (a == yr_start) and (b == yr_end)
        if fy:
            lbl = f"FY{y}" if full else f"FY{y} ({a:%b %Y}-{b:%b %Y})"
        else:
            lbl = f"{y}" if full else f"{y} ({a:%b %Y}-{b:%b %Y})"
        out_idx.append(lbl)

    return pd.Series(out_vals, index=out_idx, dtype=float)


def parse_month_end_cell(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", s):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return dt.to_period("M").to_timestamp("M") if pd.notna(dt) else None

    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", s):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return dt.to_period("M").to_timestamp("M") if pd.notna(dt) else None

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    if pd.notna(dt):
        return dt.to_period("M").to_timestamp("M")
    return None


def smart_to_month_end(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return to_eom(series)
    return series.astype(str).map(parse_month_end_cell)


def clean_nav_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = (
        s.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


def debug_clean_funds(df_raw: pd.DataFrame) -> dict:
    info: dict[str, object] = {
        "raw_rows": int(len(df_raw)),
        "raw_columns": list(df_raw.columns),
        "raw_dtypes": {c: str(t) for c, t in df_raw.dtypes.items()},
    }

    rename = {
        "Fund name": "fund",
        "fund name": "fund",
        "fund": "fund",
        "month-end": "date",
        "month_end": "date",
        "date": "date",
        "NAV": "nav",
        "nav": "nav",
        "market-cap": "market_cap",
        "market cap": "market_cap",
        "market_cap": "market_cap",
        "style": "style",
    }
    fixed = {}
    for k, v in rename.items():
        for c in df_raw.columns:
            if c.lower() == k.lower():
                fixed[c] = v

    df = df_raw.rename(columns=fixed)
    info["renamed_columns"] = list(df.columns)

    if "date" in df.columns:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        info["date_non_null_after_to_datetime"] = int(parsed_dates.notna().sum())
        info["date_sample_after_to_datetime"] = [str(x) for x in parsed_dates.head(3).tolist()]
    else:
        info["date_non_null_after_to_datetime"] = "missing"

    if "nav" in df.columns:
        numeric_nav = pd.to_numeric(df["nav"], errors="coerce")
        info["nav_non_null_after_to_numeric"] = int(numeric_nav.notna().sum())
        info["nav_sample_after_to_numeric"] = [None if pd.isna(x) else float(x) for x in numeric_nav.head(3).tolist()]
    else:
        info["nav_non_null_after_to_numeric"] = "missing"

    if "fund" in df.columns:
        cleaned_fund = (
            df["fund"]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
        info["fund_non_null_after_clean"] = int(cleaned_fund.notna().sum())
        info["fund_sample_after_clean"] = cleaned_fund.head(3).tolist()
    else:
        info["fund_non_null_after_clean"] = "missing"

    try:
        clean_df = _clean_funds(df_raw.copy())
        info["final_rows"] = int(len(clean_df))
        info["final_funds"] = int(clean_df["fund"].nunique()) if not clean_df.empty else 0
        info["final_latest"] = str(clean_df["date"].max()) if ("date" in clean_df.columns and not clean_df.empty) else "NaT"
    except Exception as exc:
        info["final_error"] = repr(exc)

    return info


@st.cache_data
def _clean_funds(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Fund name": "fund",
        "fund name": "fund",
        "fund": "fund",
        "month-end": "date",
        "month_end": "date",
        "date": "date",
        "NAV": "nav",
        "nav": "nav",
        "market-cap": "market_cap",
        "market cap": "market_cap",
        "market_cap": "market_cap",
        "style": "style",
    }
    fixed = {}
    for k, v in rename.items():
        for c in df_raw.columns:
            if c.lower() == k.lower():
                fixed[c] = v
    df = df_raw.rename(columns=fixed)
    needed = {"fund", "date", "nav", "market_cap", "style"}
    if not needed.issubset(df.columns):
        raise ValueError("Funds CSV must contain: Fund name, month-end, NAV, market-cap, style")

    raw_dates = df["date"]
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    if parsed_dates.notna().any():
        df["date"] = parsed_dates.dt.to_period("M").dt.to_timestamp("M")
    else:
        df["date"] = smart_to_month_end(raw_dates)

    raw_nav = df["nav"]
    numeric_nav = pd.to_numeric(raw_nav, errors="coerce")
    if numeric_nav.notna().any():
        df["nav"] = numeric_nav
    else:
        df["nav"] = clean_nav_series(raw_nav)

    df["fund"] = (
        df["fund"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    df = df.dropna(subset=["fund", "date", "nav"]).copy()
    df["market_cap"] = df["market_cap"].fillna("").astype(str).str.strip()
    df["style"] = df["style"].fillna("").astype(str).str.strip()

    return (
        df.sort_values(["fund", "date"])
        .groupby(["fund", "date"], as_index=False)
        .last()
        .sort_values(["fund", "date"])
        .reset_index(drop=True)
    )


@st.cache_data
def _clean_bench(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "benchmark_name": "benchmark_name",
        "benchmark": "benchmark_name",
        "name": "benchmark_name",
        "month-end": "date",
        "month_end": "date",
        "date": "date",
        "NAV": "nav",
        "nav": "nav",
        "category_type": "category_type",
        "type": "category_type",
        "category_value": "category_value",
        "value": "category_value",
    }
    fixed = {}
    for k, v in rename.items():
        for c in df_raw.columns:
            if c.lower() == k.lower():
                fixed[c] = v
    df = df_raw.rename(columns=fixed)
    needed = {"benchmark_name", "date", "nav", "category_type", "category_value"}
    if not needed.issubset(df.columns):
        raise ValueError("Benchmarks CSV must contain: benchmark_name, month-end, NAV, category_type, category_value")

    raw_dates = df["date"]
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    if parsed_dates.notna().any():
        df["date"] = parsed_dates.dt.to_period("M").dt.to_timestamp("M")
    else:
        df["date"] = smart_to_month_end(raw_dates)

    raw_nav = df["nav"]
    numeric_nav = pd.to_numeric(raw_nav, errors="coerce")
    if numeric_nav.notna().any():
        df["nav"] = numeric_nav
    else:
        df["nav"] = clean_nav_series(raw_nav)

    df["benchmark_name"] = (
        df["benchmark_name"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    df = df.dropna(subset=["benchmark_name", "date", "nav"]).copy()
    df["category_type"] = df["category_type"].fillna("").astype(str).str.strip()
    df["category_value"] = df["category_value"].fillna("").astype(str).str.strip()

    return (
        df.sort_values(["benchmark_name", "date"])
        .groupby(["benchmark_name", "date"], as_index=False)
        .last()
        .sort_values(["benchmark_name", "date"])
        .reset_index(drop=True)
    )


def window_label_series(end_idx: pd.Index, months: int) -> pd.Series:
    start = (pd.to_datetime(end_idx) - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")
    end = pd.to_datetime(end_idx)
    return pd.Series([f"{s:%b %Y}-{e:%Y}" for s, e in zip(start, end)], index=end_idx)


CAPTURE_COLUMNS = [
    "Benchmark bucket",
    "Starting year",
    "Average benchmark return (%)",
    "Average focus fund return (%)",
    "Median capture (%)",
    "Observations",
]

CAPTURE_BUCKETS = ["Top quartile", "Middle two quartiles", "Bottom quartile"]
CAPTURE_VALUE_COLUMNS = [
    "Average benchmark return (%)",
    "Average focus fund return (%)",
    "Median capture (%)",
    "Observations",
]
CAPTURE_OBSERVATION_COLUMNS = [
    "asof_date",
    "Benchmark bucket",
    "Starting year",
    "Benchmark return (%)",
    "Focus fund return (%)",
]


def build_up_down_capture_observations(
    fund_roll: pd.DataFrame,
    bench_roll: pd.DataFrame,
    *,
    focus_fund: str,
    benchmark_name: str,
    start_domain,
    end_domain,
) -> pd.DataFrame:
    """Match stored 1Y returns and assign each observation a benchmark quartile."""
    empty = pd.DataFrame(columns=CAPTURE_OBSERVATION_COLUMNS)
    if fund_roll is None or bench_roll is None or fund_roll.empty or bench_roll.empty:
        return empty

    fund = fund_roll.copy()
    bench = bench_roll.copy()
    required_fund = {"fund_name", "asof_date", "rolling_cagr"}
    required_bench = {"bench_name", "asof_date", "rolling_cagr"}
    if not required_fund.issubset(fund.columns) or not required_bench.issubset(bench.columns):
        return empty

    fund = fund[fund["fund_name"] == focus_fund].copy()
    bench = bench[bench["bench_name"] == benchmark_name].copy()
    fund["asof_date"] = pd.to_datetime(fund["asof_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    bench["asof_date"] = pd.to_datetime(bench["asof_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    fund["fund_return"] = pd.to_numeric(fund["rolling_cagr"], errors="coerce")
    bench["benchmark_return"] = pd.to_numeric(bench["rolling_cagr"], errors="coerce")
    fund = fund.dropna(subset=["asof_date", "fund_return"]).drop_duplicates("asof_date", keep="last")
    bench = bench.dropna(subset=["asof_date", "benchmark_return"]).drop_duplicates("asof_date", keep="last")

    bench["window_start"] = (
        bench["asof_date"] - pd.DateOffset(months=12)
    ).dt.to_period("M").dt.to_timestamp("M")
    start_eom = pd.Timestamp(start_domain).to_period("M").to_timestamp("M")
    end_eom = pd.Timestamp(end_domain).to_period("M").to_timestamp("M")
    bench = bench[(bench["window_start"] >= start_eom) & (bench["asof_date"] <= end_eom)].copy()
    if bench.empty:
        return empty

    q25 = float(bench["benchmark_return"].quantile(0.25))
    q75 = float(bench["benchmark_return"].quantile(0.75))
    bench["Benchmark bucket"] = np.select(
        [bench["benchmark_return"] <= q25, bench["benchmark_return"] >= q75],
        ["Bottom quartile", "Top quartile"],
        default="Middle two quartiles",
    )
    bench["Starting year"] = bench["window_start"].dt.year.astype(int)

    merged = bench[
        ["asof_date", "benchmark_return", "Benchmark bucket", "Starting year"]
    ].merge(
        fund[["asof_date", "fund_return"]],
        on="asof_date",
        how="inner",
    )
    if merged.empty:
        return empty
    merged["Benchmark return (%)"] = merged["benchmark_return"] * 100.0
    merged["Focus fund return (%)"] = merged["fund_return"] * 100.0
    return merged[CAPTURE_OBSERVATION_COLUMNS].sort_values("asof_date").reset_index(drop=True)


def _summarize_capture_observations(observations: pd.DataFrame, capture_cap: float) -> pd.DataFrame:
    """Summarise matched observations by quartile and window-start year."""
    if capture_cap <= 0:
        raise ValueError("capture_cap must be positive")
    empty = pd.DataFrame(columns=CAPTURE_COLUMNS)
    if observations.empty:
        return empty

    merged = observations.copy()
    benchmark_return = merged["Benchmark return (%)"]
    fund_return = merged["Focus fund return (%)"]
    valid_capture = np.isfinite(fund_return) & np.isfinite(benchmark_return) & (benchmark_return != 0)
    merged["capture"] = np.nan
    merged.loc[valid_capture, "capture"] = (
        fund_return.loc[valid_capture] / benchmark_return.loc[valid_capture]
    ).clip(-capture_cap, capture_cap)

    rows = []
    for (bucket, starting_year), group in merged.groupby(
        ["Benchmark bucket", "Starting year"], sort=False
    ):
        captures = group["capture"].dropna()
        rows.append(
            {
                "Benchmark bucket": bucket,
                "Starting year": int(starting_year),
                "Average benchmark return (%)": float(group["Benchmark return (%)"].mean()),
                "Average focus fund return (%)": float(group["Focus fund return (%)"].mean()),
                "Median capture (%)": (
                    float(captures.median() * 100.0) if not captures.empty else np.nan
                ),
                "Observations": int(len(captures)),
            }
        )

    result = pd.DataFrame.from_records(rows, columns=CAPTURE_COLUMNS)
    bucket_order = pd.CategoricalDtype(CAPTURE_BUCKETS, ordered=True)
    result["Benchmark bucket"] = result["Benchmark bucket"].astype(bucket_order)
    return result.sort_values(["Benchmark bucket", "Starting year"]).reset_index(drop=True)


def build_up_down_capture_table(
    fund_roll: pd.DataFrame,
    bench_roll: pd.DataFrame,
    *,
    focus_fund: str,
    benchmark_name: str,
    start_domain,
    end_domain,
    capture_cap: float = 5.0,
) -> pd.DataFrame:
    """Summarise pointwise 1Y capture by benchmark quartile and window-start year.

    Quartile thresholds are calculated from all valid benchmark observations in
    the selected domain. Row calculations use dates where both the focus fund
    and benchmark have a stored 12-month return. Individual capture ratios are
    clipped to ``[-capture_cap, capture_cap]`` before taking their median.
    """
    observations = build_up_down_capture_observations(
        fund_roll,
        bench_roll,
        focus_fund=focus_fund,
        benchmark_name=benchmark_name,
        start_domain=start_domain,
        end_domain=end_domain,
    )
    return _summarize_capture_observations(observations, capture_cap)


def prepare_capture_bucket_table(capture_table: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Return one benchmark bucket, ranked by return, with an average row."""
    display_columns = ["Starting year", *CAPTURE_VALUE_COLUMNS]
    if (
        capture_table is None
        or capture_table.empty
        or bucket not in CAPTURE_BUCKETS
        or not set(CAPTURE_COLUMNS).issubset(capture_table.columns)
    ):
        return pd.DataFrame(columns=display_columns)

    bucket_table = capture_table.loc[
        capture_table["Benchmark bucket"] == bucket, display_columns
    ].copy()
    if bucket_table.empty:
        return bucket_table

    for column in CAPTURE_VALUE_COLUMNS:
        bucket_table[column] = pd.to_numeric(bucket_table[column], errors="coerce")
    bucket_table = bucket_table.sort_values(
        ["Average benchmark return (%)", "Starting year"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    average_row = {"Starting year": "Average"}
    average_row.update(
        {column: bucket_table[column].mean() for column in CAPTURE_VALUE_COLUMNS}
    )
    return pd.concat(
        [bucket_table, pd.DataFrame([average_row], columns=display_columns)],
        ignore_index=True,
    )


def _load_up_down_capture_returns(
    focus_fund: str,
    benchmark_name: str,
    start_domain,
    end_domain,
):
    """Fetch both stored 12-month return series once for the selected period."""
    first_asof = (
        pd.Timestamp(start_domain) + pd.DateOffset(months=12)
    ).to_period("M").to_timestamp("M")
    last_asof = pd.Timestamp(end_domain).to_period("M").to_timestamp("M")
    fund_roll = load_fund_rolling(
        window_months=12,
        fund_names=[focus_fund],
        start=first_asof,
        end=last_asof,
    )
    bench_roll = load_bench_rolling(
        window_months=12,
        bench_name=benchmark_name,
        start=first_asof,
        end=last_asof,
    )
    return fund_roll, bench_roll


def make_up_down_capture_data(
    focus_fund: str,
    benchmark_name: str,
    start_domain,
    end_domain,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary rows and the matched observations used to make them."""
    fund_roll, bench_roll = _load_up_down_capture_returns(
        focus_fund, benchmark_name, start_domain, end_domain
    )
    observations = build_up_down_capture_observations(
        fund_roll,
        bench_roll,
        focus_fund=focus_fund,
        benchmark_name=benchmark_name,
        start_domain=start_domain,
        end_domain=end_domain,
    )
    return _summarize_capture_observations(observations, 5.0), observations


def make_up_down_capture_table(
    focus_fund: str,
    benchmark_name: str,
    start_domain,
    end_domain,
) -> pd.DataFrame:
    """Fetch precomputed 12-month returns and build the capture table."""
    fund_roll, bench_roll = _load_up_down_capture_returns(
        focus_fund, benchmark_name, start_domain, end_domain
    )
    return build_up_down_capture_table(
        fund_roll,
        bench_roll,
        focus_fund=focus_fund,
        benchmark_name=benchmark_name,
        start_domain=start_domain,
        end_domain=end_domain,
    )


@st.cache_data(show_spinner=False)
def make_rolling_df(funds_df, selected_funds, focus_fund, bench_ser, months, start_domain, end_domain):
    fund_roll = load_fund_rolling(window_months=months, fund_names=selected_funds, start=None, end=None)
    if fund_roll.empty:
        return pd.DataFrame()

    wide = (
        fund_roll.pivot_table(
            index="asof_date",
            columns="fund_name",
            values="rolling_cagr",
            aggfunc="first",
        )
        .sort_index()
    )

    end_idx = pd.to_datetime(wide.index)
    start_idx = (end_idx - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")
    mask = pd.Series(True, index=end_idx)
    if start_domain is not None:
        mask &= start_idx >= start_domain
    if end_domain is not None:
        mask &= end_idx <= end_domain

    wide = wide.loc[mask]
    if wide.empty:
        return pd.DataFrame()

    bench_label = bench_ser.name if bench_ser is not None and hasattr(bench_ser, "name") else None
    if bench_label:
        bench_roll = load_bench_rolling(window_months=months, bench_name=bench_label, start=None, end=None)
        if not bench_roll.empty:
            wide[bench_label] = bench_roll.set_index("asof_date")["rolling_cagr"].sort_index().reindex(wide.index)

    cols_excluding_bench = [c for c in wide.columns if c != bench_label]
    peer_cols = [c for c in cols_excluding_bench if c != focus_fund]
    if peer_cols:
        wide["Peer avg"] = wide[peer_cols].mean(axis=1)
    elif "Peer avg" in wide.columns:
        wide = wide.drop(columns=["Peer avg"])

    wide = wide * 100.0
    wide.index.name = "Window"
    return wide


@st.cache_data(show_spinner=False)
def make_multi_fund_rolling(funds_df, selected_funds, months, start_domain, end_domain):
    if not selected_funds:
        return pd.DataFrame()

    roll = load_fund_rolling(window_months=months, fund_names=selected_funds, start=None, end=None)
    if roll.empty:
        return pd.DataFrame()

    wide = roll.pivot_table(
        index="asof_date",
        columns="fund_name",
        values="rolling_cagr",
        aggfunc="first",
    ).sort_index()

    end_idx = pd.to_datetime(wide.index)
    start_idx = (end_idx - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")
    mask = pd.Series(True, index=end_idx)
    if start_domain is not None:
        mask &= start_idx >= start_domain
    if end_domain is not None:
        mask &= end_idx <= end_domain

    wide = wide.loc[mask]
    if wide.empty:
        return pd.DataFrame()

    wide = wide * 100.0
    wide.index.name = "Window"
    return wide


@st.cache_data(show_spinner=False)
def make_multi_fund_rolling_df(funds_df, selected_funds, months, start_domain, end_domain):
    return make_multi_fund_rolling(funds_df, selected_funds, months, start_domain, end_domain)


@st.cache_data(show_spinner=False)
def rolling_outperf_stats(df: pd.DataFrame, focus_name: str, bench_label: str | None = None):
    if df is None or df.empty or focus_name not in df.columns:
        return None

    if bench_label and bench_label in df.columns:
        bcol = bench_label
    elif "Benchmark" in df.columns:
        bcol = "Benchmark"
    else:
        return None

    op = (df[focus_name] - df[bcol]).dropna()
    if op.empty:
        return None

    return pd.DataFrame(
        {
            "windows": [int(op.notna().count())],
            "median (ppt)": [float(np.nanmedian(op))],
            "mean   (ppt)": [float(np.nanmean(op))],
            "min    (ppt)": [float(np.nanmin(op))],
            "max    (ppt)": [float(np.nanmax(op))],
            "prob. of outperformance": [float((op > 0).mean() * 100.0)],
        }
    )
