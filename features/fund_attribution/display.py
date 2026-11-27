import pandas as pd
import streamlit as st

from features.fund_attribution.data import (
    BENCH_NAME_MID150, BENCH_NAME_NIFTY100, BENCH_NAME_NIFTY50,
    BENCH_NAME_NIFTY500, BENCH_NAME_SMALL250, CASH_ANNUAL_RETURN,
    load_attrib_fundamental_history,
)
from features.fund_attribution.fundamentals import FUNDAMENTAL_FIELDS, build_fundamental_history_workbook


def display_attribution_results(
    stock_df, hit_df, cat_df, diag, bench_mode, *, selected_keys, original_summary, editor_key, fundamentals,
):
    """Render the active run and return the checked keys when a button is submitted."""
    start = diag["effective_start_month"]
    if start != diag["requested_start_month"]:
        st.info(f"The first portfolio in the selected period is {start:%b %Y}. Attribution starts there.")

    benchmark_names = {"Vs NIFTY 50": BENCH_NAME_NIFTY50, "Vs NIFTY 500": BENCH_NAME_NIFTY500}
    if bench_mode == "Like-for-like":
        needed = [BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250, BENCH_NAME_NIFTY50]
    else:
        needed = [benchmark_names[bench_mode]]
    missing = [name for name in needed if name in diag["missing_benchmarks"]]
    if missing:
        st.warning("Attribution uses zero returns where benchmark NAV is unavailable for " + ", ".join(missing) + ".")

    invalid_prices = diag["adj_price_invalid_rows_df"]
    if not invalid_prices.empty:
        st.warning(f"Missing or non-positive adjusted prices were treated as zero return for {len(invalid_prices):,} stock periods.")

    summary = diag["portfolio_return_summary_df"]
    if original_summary is not None:
        original = original_summary.set_index("Series")
        comparison = summary.set_index("Series").join(
            original.rename(columns={"Total return (%)": "Original total return (%)", "CAGR (%)": "Original CAGR (%)"})
        )
        comparison["Change in CAGR (pp)"] = (comparison["CAGR (%)"] - comparison["Original CAGR (%)"]).round(2)
        summary = comparison.reset_index()
        st.caption(
            "Checked stocks match their benchmarks each period. Their rupee alpha can still vary because the fund "
            "and benchmark compound from different balances."
        )

    st.dataframe(summary, width="stretch", hide_index=True)
    if bench_mode in benchmark_names:
        st.caption(f"Benchmark series is {benchmark_names[bench_mode]}. Cash earns {CASH_ANNUAL_RETURN:.0%} annually when included.")
    else:
        st.caption(
            "Like-for-like uses Nifty 100 TRI for large caps and Nifty Midcap 150 TRI for mid caps. "
            "Small caps, microcaps, and stocks without a size band or ACE code use Nifty Smallcap 250 TRI. "
            "Mapped overseas and other equities use Nifty 50 TRI."
        )

    stock_rows = stock_df.index.isin(diag["equity_keys"])
    selectable = stock_df.loc[stock_rows].copy()
    selectable.insert(0, "Use benchmark", selectable.index.isin(selected_keys))

    with st.form("fa_attribution_scenario_form"):
        st.subheader("Stock-level attribution")
        st.caption("Stock and benchmark returns are cumulative for holdings under one year and annualized from one year onward.")
        if selectable.empty:
            st.info("There are no stock positions to substitute in this portfolio.")
        else:
            selectable = st.data_editor(
                selectable, disabled=stock_df.columns.tolist(), width="stretch", hide_index=True,
                column_config={"Use benchmark": st.column_config.CheckboxColumn("Use benchmark")}, key=editor_key,
            )
        if (~stock_rows).any():
            with st.expander("Cash and other positions"):
                st.dataframe(stock_df.loc[~stock_rows], width="stretch", hide_index=True)

        st.subheader("Stock fundamentals during holding periods")
        if fundamentals is None:
            st.warning("Stock fundamentals could not be loaded.")
        elif fundamentals.empty:
            st.info("No stocks were held for more than one year in the selected period.")
        else:
            st.caption("Average weights are shares of the full fund across all reports in the review period. "
                       "Stocks absent from a report count as zero. Growth uses the first and last reports that list the stock, "
                       "including gaps between trades. PAT and sales use trailing 12 months.")
            fundamentals_with_cagr = fundamentals.copy()
            fundamentals_with_cagr.insert(2, "Stock CAGR", stock_df["Stock return (%)"])
            st.dataframe(fundamentals_with_cagr, width="stretch", hide_index=True, column_config={
                "Average weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Stock CAGR": st.column_config.NumberColumn(format="%.2f%%", help="Matches the stock return in the attribution table above."),
            })

        st.subheader("Hit rate by rupee alpha")
        st.dataframe(hit_df, width="stretch", hide_index=True)
        st.subheader("Category-level attribution")
        st.dataframe(cat_df, width="stretch", hide_index=True)

        calculate, reset = st.columns([3, 1])
        with calculate:
            apply = st.form_submit_button("Calculate assuming benchmark returns", type="primary", disabled=selectable.empty)
        with reset:
            clear = st.form_submit_button("Reset")

    if clear:
        return ()
    if apply:
        return tuple(selectable.index[selectable["Use benchmark"]])
    return None


def display_fundamental_data_download(stocks, start_month, end_month, fund_id):
    st.subheader("Review underlying fundamental data")
    metric = st.radio("Metric", list(FUNDAMENTAL_FIELDS), horizontal=True, key="fa_fundamental_review_metric")
    if not stocks:
        st.info("No stocks were reported in the selected period.")
        return

    def prepare_workbook():
        ace_codes = tuple(sorted({code for _, code in stocks if code is not None}))
        values = load_attrib_fundamental_history(ace_codes, start_month, end_month, metric)
        return build_fundamental_history_workbook(stocks, start_month, end_month, metric, values)

    metric_slug = metric.lower().replace(" ", "_")
    st.download_button(
        "Download", data=prepare_workbook, on_click="ignore",
        file_name=f"fund_{fund_id}_{metric_slug}_{start_month:%Y-%m}_to_{end_month:%Y-%m}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"fa_fundamental_review_download_{metric_slug}",
    )
