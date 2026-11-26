import pandas as pd
import streamlit as st

from features.fund_attribution.data import (
    BENCH_NAME_MID150, BENCH_NAME_NIFTY100, BENCH_NAME_NIFTY50,
    BENCH_NAME_NIFTY500, BENCH_NAME_SMALL250, CASH_ANNUAL_RETURN,
)


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
        st.caption("Like-for-like uses Nifty 100 for large caps, Midcap 150 for mid caps, and Smallcap 250 for small caps. "
                   "Other equities use Nifty 50.")

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
            st.caption("Growth is annualized from the first entry to the end of the final holding period, including gaps between trades. "
                       "The holding-period column counts only time actually held. PAT and sales use trailing 12 months.")
            fundamentals_with_cagr = fundamentals.copy()
            fundamentals_with_cagr.insert(2, "Stock CAGR", stock_df["Stock return (%)"])
            st.dataframe(fundamentals_with_cagr, width="stretch", hide_index=True, column_config={
                "Holding period (years)": st.column_config.NumberColumn(format="%.2f"),
                "Stock CAGR": st.column_config.NumberColumn(format="%.2f%%"),
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
