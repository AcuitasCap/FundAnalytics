from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from services.performance_db import load_bench_from_db, load_funds_from_db
from services.performance_plots import (
    build_rel_fill,
    df_to_table_figure,
    plot_multi_fund_rolling,
    plot_rolling,
    style_relative_multi_horizon,
)
from services.performance_returns import (
    _clean_bench,
    _clean_funds,
    coerce_num,
    make_multi_fund_rolling_df,
    make_rolling_df,
    rolling_outperf_stats,
    to_eom,
    yearly_returns_with_custom_domain,
)


def performance_page(home_button):
    home_button()
    st.caption("Performance diagnostics v1 - 20 Apr 2026!")
    raw_funds_df = load_funds_from_db()
    raw_date_dtype = str(raw_funds_df["month-end"].dtype) if "month-end" in raw_funds_df.columns else "missing"
    raw_nav_dtype = str(raw_funds_df["NAV"].dtype) if "NAV" in raw_funds_df.columns else "missing"
    st.caption(
        f"Performance diagnostics: raw NAV rows = {len(raw_funds_df)} | "
        f"month-end dtype = {raw_date_dtype} | NAV dtype = {raw_nav_dtype}"
    )
    if raw_funds_df.empty:
        st.error("No fund NAV data found in database.")
        st.stop()

    funds_df = _clean_funds(raw_funds_df.copy())
    clean_latest = funds_df["date"].max() if ("date" in funds_df.columns and not funds_df.empty) else "NaT"
    st.caption(
        f"Performance diagnostics: cleaned rows = {len(funds_df)} | "
        f"clean funds = {funds_df['fund'].nunique() if 'fund' in funds_df.columns and not funds_df.empty else 0} | "
        f"clean latest = {clean_latest}"
    )

    bench_df = None
    try:
        raw_bench_df = load_bench_from_db()
        if not raw_bench_df.empty:
            bench_df = _clean_bench(raw_bench_df.copy())
    except Exception as e:
        st.warning(f"Could not load benchmark data from DB: {e}")
        bench_df = None

    fund_candidates = ["Fund", "Fund name", "fund", "fund_name"]
    fund_col = next((c for c in fund_candidates if c in funds_df.columns), None)
    date_candidates = ["month-end", "Date", "date", "nav_date"]
    date_col = next((c for c in date_candidates if c in funds_df.columns), None)

    latest_str = "N/A"
    if date_col is not None:
        try:
            latest_date = funds_df[date_col].max()
            latest_str = latest_date.strftime("%d-%b-%Y")
        except Exception:
            latest_str = str(latest_date)

    num_funds = funds_df[fund_col].nunique() if fund_col is not None else "N/A"
    st.caption(f"Data source: Supabase · Funds: {num_funds} · Latest NAV date: {latest_str}")

    all_caps = sorted(funds_df["market_cap"].dropna().unique().tolist())
    st.caption(f"Performance diagnostics: market-cap options found = {len(all_caps)}")
    caps = st.multiselect(
        "Market-cap (select one or more)",
        options=all_caps,
        default=[],
        key="perf_market_caps",
    )
    st.divider()
    if not caps:
        st.warning("Select at least one Market-cap to continue.")
        st.stop()

    filtered = funds_df[funds_df["market_cap"].isin(caps)].copy()

    if bench_df is not None and not bench_df.empty:
        bench_names = sorted(bench_df["benchmark_name"].dropna().unique().tolist())
    else:
        bench_names = []

    if not bench_names:
        st.warning("No benchmarks found in database.")
        bench_label = None
        bench_ser = None
    else:
        bench_selected = st.multiselect(
            "Benchmarks (select one or more)",
            options=bench_names,
            default=[],
            key="perf_benchmarks",
        )
        if bench_selected:
            bench_label = bench_selected[0]
            bmask = bench_df["benchmark_name"] == bench_label
            bench_ser = (
                bench_df.loc[bmask, ["date", "nav"]]
                .drop_duplicates("date")
                .set_index("date")["nav"]
                .sort_index()
            )
            bench_ser.name = bench_label
        else:
            bench_label = None
            bench_ser = None

    filtered["date"] = to_eom(filtered["date"])
    filtered["nav"] = coerce_num(filtered["nav"])
    filtered = filtered.sort_values(["fund", "date"]).drop_duplicates(subset=["fund", "date"], keep="last")

    fund_options = sorted(filtered["fund"].unique().tolist())
    st.markdown("**Funds (multi-select)**")
    funds_with_all = ["ALL"] + fund_options
    funds_selected = st.multiselect("Choose funds (include 'ALL' for all in list)", options=funds_with_all, default=[])
    if any(str(f).upper() == "ALL" for f in funds_selected):
        funds_selected = fund_options
    if not funds_selected:
        st.warning("Select at least one fund.")
        st.stop()

    focus_fund = st.selectbox("Focus fund (vs. peers)", options=["-- none --"] + funds_selected, index=0)
    if focus_fund == "-- none --":
        st.warning("Pick a Focus fund to compute Peers Avg (we exclude the focus from peers).")
        st.stop()

    st.header("Performance analysis - rolling, yearly, and P2P")

    def eom(y: int, m: int) -> pd.Timestamp:
        return pd.Timestamp(year=y, month=m, day=1).to_period("M").to_timestamp("M")

    date_years = sorted(pd.to_datetime(filtered["date"]).dt.year.unique().tolist())
    if not date_years:
        st.stop()
    min_y, max_y = min(date_years), max(date_years)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m2n = {m: i + 1 for i, m in enumerate(months)}

    with st.form("perf_filters"):
        cA1, cA2, cB1, cB2 = st.columns([1, 1, 1, 1])
        with cA1:
            start_month = st.selectbox("Start month (start-domain)", months, index=0, key="rr_start_m")
        with cA2:
            start_year = st.selectbox("Start year  (start-domain)", list(range(min_y, max_y + 1)), index=0, key="rr_start_y")
        with cB1:
            end_month = st.selectbox("End month   (end-domain)", months, index=len(months) - 1, key="rr_end_m")
        with cB2:
            end_year = st.selectbox(
                "End year    (end-domain)",
                list(range(min_y, max_y + 1)),
                index=len(range(min_y, max_y + 1)) - 1,
                key="rr_end_y",
            )

        start_domain = eom(int(start_year), m2n[start_month])
        end_domain = eom(int(end_year), m2n[end_month])

        analysis_mode = st.radio(
            "Analysis mode",
            [
                "Rolling returns vs. peer average and benchmarks",
                "Rolling returns - multiple funds",
                "Returns (Strict FY/CY endpoints)",
                "Point to point returns & relative return vs. benchmark (1/3/5/7 year CAGR)",
            ],
            index=0,
            key="perf_analysis_mode",
        )

        apply_filters = st.form_submit_button("Update", type="primary")

    filters_applied = st.session_state.get("perf_filters_applied", False)
    if apply_filters:
        filters_applied = True
        st.session_state["perf_filters_applied"] = True

    if end_domain <= start_domain:
        st.warning("End month must be after Start month.")
        return

    if not filters_applied:
        st.info("Adjust period and analysis mode above, then click **Update**.")
        return

    def window_ok(start_dt, end_dt, months_count: int) -> bool:
        return (start_dt + pd.DateOffset(months=months_count)) <= end_dt

    print_items = []
    p_act = p_bench = p_rel = p_p2p = p_relmh = False
    disp_actual = bench_df_print = disp_rel = p2p_df = rel_mh_disp = pd.DataFrame()

    if analysis_mode == "Rolling returns vs. peer average and benchmarks":
        st.subheader("3Y Rolling - Focus vs Peer avg vs Benchmark")
        series_opts_3 = [focus_fund, "Peer avg"] + ([bench_label] if bench_label else [])
        series_selected_3 = st.multiselect("Show series (3Y)", options=series_opts_3, default=series_opts_3, key="series3")

        if not window_ok(start_domain, end_domain, 36):
            st.info("Selected range too short for 3Y windows.")
        else:
            df3 = make_rolling_df(filtered, funds_selected, focus_fund, bench_ser, 36, start_domain, end_domain)
            fig3 = plot_rolling(df3, 36, focus_fund, bench_label, chart_height=560, include_cols=series_selected_3)
            if fig3 is None:
                st.info("Insufficient data or no series selected for 3Y rolling chart.")
            else:
                st.plotly_chart(fig3, use_container_width=True)
                stats3 = rolling_outperf_stats(df3, focus_fund, bench_label)
                st.subheader("3Y Rolling Outperformance Stats (Focus fund vs Benchmark)")
                st.dataframe(stats3.round(2) if stats3 is not None else pd.DataFrame({"info": ["Not enough overlapping 3Y windows"]}))
                if st.checkbox("To print", key="print_fig3"):
                    print_items.append(("3Y Rolling - Focus/Peers/Benchmark", fig3))

        st.subheader("1Y Rolling - Focus vs Peer avg vs Benchmark")
        series_opts_1 = [focus_fund, "Peer avg"] + ([bench_label] if bench_label else [])
        series_selected_1 = st.multiselect("Show series (1Y)", options=series_opts_1, default=series_opts_1, key="series1")

        if not window_ok(start_domain, end_domain, 12):
            st.info("Selected range too short for 1Y windows.")
        else:
            df1 = make_rolling_df(filtered, funds_selected, focus_fund, bench_ser, 12, start_domain, end_domain)
            fig1 = plot_rolling(df1, 12, focus_fund, bench_label, chart_height=560, include_cols=series_selected_1)
            if fig1 is None:
                st.info("Insufficient data or no series selected for 1Y rolling chart.")
            else:
                st.plotly_chart(fig1, use_container_width=True)
                stats1 = rolling_outperf_stats(df1, focus_fund, bench_label)
                st.subheader("1Y Rolling Outperformance Stats (Focus fund vs Benchmark)")
                st.dataframe(stats1.round(2) if stats1 is not None else pd.DataFrame({"info": ["Not enough overlapping 1Y windows"]}))
                if st.checkbox("To print", key="print_fig1"):
                    print_items.append(("1Y Rolling - Focus/Peers/Benchmark", fig1))

    elif analysis_mode == "Rolling returns - multiple funds":
        st.subheader("3Y Rolling - Multiple Selected Funds")
        mf3 = st.multiselect("Pick funds to plot (3Y multi-fund)", options=funds_selected, default=[focus_fund], key="mf3")
        if not window_ok(start_domain, end_domain, 36):
            st.info("Selected range too short for 3Y windows.")
        else:
            df3m = make_multi_fund_rolling_df(filtered, mf3, 36, start_domain, end_domain)
            fig3m = plot_multi_fund_rolling(df3m, 36, focus_name=focus_fund, chart_height=560)
            if fig3m is None:
                st.info("Not enough data to plot selected funds (3Y).")
            else:
                st.plotly_chart(fig3m, use_container_width=True)
                if st.checkbox("To print", key="print_fig3m"):
                    print_items.append(("3Y Rolling - Multiple funds", fig3m))

        st.subheader("1Y Rolling - Multiple Selected Funds")
        mf1 = st.multiselect("Pick funds to plot (1Y multi-fund)", options=funds_selected, default=[focus_fund], key="mf1")
        if not window_ok(start_domain, end_domain, 12):
            st.info("Selected range too short for 1Y windows.")
        else:
            df1m = make_multi_fund_rolling_df(filtered, mf1, 12, start_domain, end_domain)
            fig1m = plot_multi_fund_rolling(df1m, 12, focus_name=focus_fund, chart_height=560)
            if fig1m is None:
                st.info("Not enough data to plot selected funds (1Y).")
            else:
                st.plotly_chart(fig1m, use_container_width=True)
                if st.checkbox("To print", key="print_fig1m"):
                    print_items.append(("1Y Rolling - Multiple funds", fig1m))

    elif analysis_mode == "Returns (Strict FY/CY endpoints)":
        st.header("Yearly Returns (Strict FY/CY endpoints)")
        yr_type_strict = st.radio(
            "Year type",
            options=["Financial (Apr-Mar)", "Calendar (Jan-Dec)"],
            index=0,
            horizontal=True,
            key="strict_year_type",
        )
        use_fy_strict = yr_type_strict.startswith("Financial")

        yr_rows = {}
        for f in funds_selected:
            s = filtered.loc[filtered["fund"] == f, ["date", "nav"]].drop_duplicates("date").set_index("date")["nav"]
            yr_rows[f] = yearly_returns_with_custom_domain(s, start_domain, end_domain, fy=use_fy_strict)
        yr_df = pd.DataFrame(yr_rows).T

        yr_bench = None
        if bench_ser is not None and not bench_ser.empty:
            yr_bench = yearly_returns_with_custom_domain(bench_ser, start_domain, end_domain, fy=use_fy_strict).rename("Benchmark")

        if yr_df.empty:
            st.info("Not enough data to compute yearly returns for the selected funds.")
        else:
            cols_order = list(yr_df.columns)
            if yr_bench is not None and not yr_bench.empty:
                for c in yr_bench.index:
                    if c not in cols_order:
                        cols_order.append(c)
            yr_df = yr_df.reindex(columns=cols_order)

            st.subheader("Actual yearly returns - Funds (rows) vs Years (columns)")
            disp_actual = (yr_df.loc[funds_selected, cols_order] * 100.0).copy()
            disp_actual.insert(0, "Fund", disp_actual.index)
            disp_actual = disp_actual.reset_index(drop=True)
            num_cols = [c for c in disp_actual.columns if c != "Fund"]
            disp_actual[num_cols] = disp_actual[num_cols].round(1)
            st.dataframe(
                disp_actual.style.format({c: "{:.1f}" for c in num_cols}, na_rep="—").set_table_styles(
                    [
                        {"selector": "table", "props": "table-layout:fixed"},
                        {"selector": "th.col_heading", "props": "white-space:normal; line-height:1.1; height:56px"},
                    ]
                ),
                use_container_width=True,
            )
            p_act = st.checkbox("To print", key="print_actual_tbl")
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            st.subheader(f"Benchmark yearly returns - {bench_label}")
            if yr_bench is None or yr_bench.empty:
                st.info("Benchmark not available.")
                bench_df_print = pd.DataFrame()
            else:
                bench_df_print = ((yr_bench[cols_order] * 100.0).to_frame().T).round(2)
                bench_df_print.index = [bench_label]
                bench_df_print = bench_df_print.reset_index().rename(columns={"index": "Benchmark"})
                st.dataframe(
                    bench_df_print.style.format(
                        {c: "{:.1f}" for c in bench_df_print.columns if c != "Benchmark"},
                        na_rep="—",
                    ).set_table_styles(
                        [
                            {"selector": "table", "props": "table-layout:fixed"},
                            {"selector": "th.col_heading", "props": "white-space:normal; line-height:1.1; height:56px"},
                        ]
                    ),
                    use_container_width=True,
                )
            p_bench = st.checkbox("To print", key="print_bench_tbl")
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            st.subheader("Relative yearly returns (ppt) - Funds (rows) vs Years (columns)")
            rel_df = pd.DataFrame()
            if yr_bench is not None and not yr_bench.empty:
                common = [c for c in cols_order if c in yr_bench.index]
                if common:
                    rel_df = (yr_df.loc[funds_selected, common].subtract(yr_bench[common], axis=1) * 100.0).round(2)

            if rel_df.empty:
                st.info("Not enough overlap to compute relative returns.")
                disp_rel = pd.DataFrame()
            else:
                disp_rel = rel_df.copy()
                disp_rel.insert(0, "Fund", disp_rel.index)
                disp_rel = disp_rel.reset_index(drop=True)

                def rel_colors(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for c in df.columns:
                        if c == "Fund":
                            continue
                        styles[c] = df[c].apply(
                            lambda v: "background-color:#e6f4ea;color:#0b8043"
                            if pd.notna(v) and v > 0
                            else "background-color:#fdecea;color:#a50e0e"
                            if pd.notna(v) and v < 0
                            else ""
                        )
                    return styles

                st.write(
                    disp_rel.style.apply(rel_colors, axis=None)
                    .format({c: "{:.1f}" for c in disp_rel.columns if c != "Fund"}, na_rep="—")
                    .set_table_styles(
                        [
                            {"selector": "table", "props": "table-layout:fixed"},
                            {"selector": "th.col_heading", "props": "white-space:normal; line-height:1.1; height:56px"},
                        ]
                    )
                )
            p_rel = st.checkbox("To print", key="print_rel_tbl")

    else:
        st.header("Point-to-Point (P2P) Returns - Custom period CAGR")
        p2p_start = start_domain
        p2p_end = end_domain

        def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
            return (b.year - a.year) * 12 + (b.month - a.month)

        def series_return_between(s: pd.Series, start_eom: pd.Timestamp, end_eom: pd.Timestamp):
            s = s.dropna().sort_index()
            if start_eom not in s.index or end_eom not in s.index or end_eom <= start_eom:
                return np.nan

            m = months_between(start_eom, end_eom)
            if m <= 0:
                return np.nan

            try:
                ratio = s.loc[end_eom] / s.loc[start_eom]
                return ratio - 1.0 if m < 12 else ratio ** (12.0 / m) - 1.0
            except Exception:
                return np.nan

        if p2p_end <= p2p_start:
            st.warning("P2P End must be after Start.")
        else:
            rows = []
            for f in funds_selected:
                s = filtered.loc[filtered["fund"] == f, ["date", "nav"]].drop_duplicates("date").set_index("date")["nav"]
                m = months_between(p2p_start, p2p_end)
                metric_label = "Abs % (P2P)" if m < 12 else "CAGR % (P2P)"
                val = series_return_between(s, p2p_start, p2p_end)
                rows.append(
                    {
                        "Fund": f,
                        "Start": f"{p2p_start:%b %Y}",
                        "End": f"{p2p_end:%b %Y}",
                        "Months": m,
                        metric_label: None if np.isnan(val) else round(val * 100.0, 2),
                    }
                )

            if bench_ser is not None and not bench_ser.empty:
                m = months_between(p2p_start, p2p_end)
                metric_label = "Abs % (P2P)" if m < 12 else "CAGR % (P2P)"
                bval = series_return_between(bench_ser, p2p_start, p2p_end)
                rows.append(
                    {
                        "Fund": bench_label,
                        "Start": f"{p2p_start:%b %Y}",
                        "End": f"{p2p_end:%b %Y}",
                        "Months": m,
                        metric_label: None if np.isnan(bval) else round(bval * 100.0, 2),
                    }
                )

            p2p_df = pd.DataFrame(rows)
            ret_cols = [c for c in p2p_df.columns if isinstance(c, str) and ("CAGR" in c or "Abs" in c)]
            for c in ret_cols:
                p2p_df[c] = pd.to_numeric(p2p_df[c], errors="coerce").round(1)
            if ret_cols:
                p2p_df = p2p_df.sort_values(by=ret_cols[0], ascending=False)

            st.dataframe(p2p_df.style.format({c: "{:.1f}%" for c in ret_cols}, na_rep="—"), use_container_width=True)
            p_p2p = st.checkbox("To print", key="print_p2p_tbl")

            st.subheader("Relative CAGR vs Benchmark - 1Y / 3Y / 5Y / 7Y (as of P2P end month)")

            def end_aligned_cagr(series: pd.Series, end_eom: pd.Timestamp, months_count: int) -> float:
                s = series.dropna().sort_index()
                if end_eom not in s.index:
                    return np.nan
                start_eom = (end_eom - pd.DateOffset(months=months_count)).to_period("M").to_timestamp("M")
                if start_eom not in s.index or end_eom <= start_eom:
                    return np.nan
                try:
                    return (s.loc[end_eom] / s.loc[start_eom]) ** (12.0 / months_count) - 1.0
                except Exception:
                    return np.nan

            horizons = [(12, "1Y"), (36, "3Y"), (60, "5Y"), (84, "7Y")]
            if bench_ser is not None and not bench_ser.empty:
                bench_cagrs = {lbl: end_aligned_cagr(bench_ser, p2p_end, m) for m, lbl in horizons}
            else:
                bench_cagrs = {lbl: np.nan for _, lbl in horizons}

            rel_rows = []
            for f in funds_selected:
                s = filtered.loc[filtered["fund"] == f, ["date", "nav"]].drop_duplicates("date").set_index("date")["nav"]
                row = {"Fund": f}
                for m, lbl in horizons:
                    fc = end_aligned_cagr(s, p2p_end, m)
                    bc = bench_cagrs.get(lbl, np.nan)
                    row[lbl] = None if (np.isnan(fc) or np.isnan(bc)) else round((fc - bc) * 100.0, 2)
                rel_rows.append(row)

            rel_mh_df = pd.DataFrame(rel_rows).set_index("Fund")
            if rel_mh_df.empty:
                st.info("Not enough data to compute relative multi-horizon returns.")
                rel_mh_disp = pd.DataFrame()
            else:
                st.dataframe(style_relative_multi_horizon(rel_mh_df), use_container_width=True)
                rel_mh_disp = rel_mh_df.copy()
            p_relmh = st.checkbox("To print", key="print_rel_mh")

    st.markdown("---")
    if st.button("Print charts"):
        if not print_items:
            st.warning('Nothing selected. Tick "To print" under the charts/tables you want.')
        else:
            try:
                pdf_bytes = BytesIO()
                c = canvas.Canvas(pdf_bytes, pagesize=landscape(A4))
                W, H = landscape(A4)
                c.setFont("Helvetica-Bold", 18)
                c.drawString(40, H - 40, f"Performance charts & metrics - {focus_fund}")
                c.setFont("Helvetica", 10)
                c.drawString(40, H - 58, f"Benchmark: {bench_label}")
                c.showPage()

                for caption, fig in print_items:
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(40, H - 40, caption)
                    png = fig.to_image(format="png", scale=2)
                    img = ImageReader(BytesIO(png))
                    max_w, max_h = W - 80, H - 100
                    iw, ih = img.getSize()
                    scale = min(max_w / iw, max_h / ih)
                    c.drawImage(img, 40 + (max_w - iw * scale) / 2, 40, width=iw * scale, height=ih * scale)
                    c.showPage()

                if p_act and not disp_actual.empty:
                    figA = df_to_table_figure(disp_actual.round(2), "Actual yearly returns - Funds vs Years", fill="white")
                    if figA is not None:
                        png = figA.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Actual yearly returns - Funds vs Years")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w / iw, max_h / ih)
                        c.drawImage(img, 40, 40, width=iw * scale, height=ih * scale)
                        c.showPage()

                if p_bench and not bench_df_print.empty:
                    figB = df_to_table_figure(bench_df_print.round(2), f"Benchmark yearly returns - {bench_label}", fill="white")
                    if figB is not None:
                        png = figB.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, f"Benchmark yearly returns - {bench_label}")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w / iw, max_h / ih)
                        c.drawImage(img, 40, 40, width=iw * scale, height=ih * scale)
                        c.showPage()

                if p_rel and not disp_rel.empty:
                    fills_rel = build_rel_fill(disp_rel, fund_col="Fund", misaligned=None)
                    figR = df_to_table_figure(disp_rel.round(2), "Relative yearly returns (ppt)", fill=fills_rel)
                    if figR is not None:
                        png = figR.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Relative yearly returns (ppt)")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w / iw, max_h / ih)
                        c.drawImage(img, 40, 40, width=iw * scale, height=ih * scale)
                        c.showPage()

                if p_p2p and not p2p_df.empty:
                    figP = df_to_table_figure(p2p_df, "Point-to-Point (P2P) Returns", fill="white")
                    if figP is not None:
                        png = figP.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Point-to-Point (P2P) Returns")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w / iw, max_h / ih)
                        c.drawImage(img, 40, 40, width=iw * scale, height=ih * scale)
                        c.showPage()

                if p_relmh and not rel_mh_disp.empty:
                    dfmh = rel_mh_disp.copy().reset_index()
                    fills_mh = build_rel_fill(dfmh, fund_col="Fund", misaligned=None)
                    figMH = df_to_table_figure(dfmh.set_index("Fund"), "Relative CAGR vs Benchmark - 1Y / 3Y / 5Y / 7Y (ppt)", fill=fills_mh)
                    if figMH is not None:
                        png = figMH.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Relative CAGR vs Benchmark - 1Y / 3Y / 5Y / 7Y (ppt)")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w / iw, max_h / ih)
                        c.drawImage(img, 40, 40, width=iw * scale, height=ih * scale)
                        c.showPage()

                c.save()
                pdf_bytes.seek(0)
                st.download_button(
                    label="Download PDF",
                    file_name=f"Performance_{focus_fund.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    data=pdf_bytes.getvalue(),
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}. Ensure 'kaleido' and 'reportlab' are installed.")
