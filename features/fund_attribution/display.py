import pandas as pd
import streamlit as st

from features.fund_attribution.data import (
    BENCH_NAME_MID150, BENCH_NAME_NIFTY100, BENCH_NAME_NIFTY50,
    BENCH_NAME_NIFTY500, BENCH_NAME_SMALL250,
)

def display_attribution_results(stock_df, hit_df, cat_df, diag, bench_mode):
        # Benchmark warnings
        miss = diag.get("missing_benchmarks", [])
        if bench_mode == "Like-for-like":
            needed = [BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250]
            miss_needed = [x for x in needed if x in miss]
            if miss_needed:
                st.warning("Missing benchmark NAV series in DB (will default missing benchmark returns to 0): " + ", ".join(miss_needed))
        else:
            needed = BENCH_NAME_NIFTY50 if bench_mode == "Vs NIFTY 50" else BENCH_NAME_NIFTY500
            if needed in miss:
                st.warning(f"Missing benchmark NAV series in DB for {needed} (benchmark returns will default to 0).")

        adj_invalid = diag.get("adj_price_invalid_rows_df", pd.DataFrame())
        if isinstance(adj_invalid, pd.DataFrame) and not adj_invalid.empty:
            st.warning(
                f"Rows with missing/non-positive adj_price were treated as 0 return: {len(adj_invalid):,} stock-month rows."
            )

        st.subheader("1) Stock-level attribution")
        st.dataframe(stock_df, use_container_width=True)

        st.subheader("2) Hit-rate summary (alpha contribution based)")
        st.dataframe(hit_df, use_container_width=True)

        st.subheader("3) Category-level attribution")
        st.dataframe(cat_df, use_container_width=True)

        dom_sum = diag.get("domestic_decomp_summary_df")
        dom_period_label = diag.get("domestic_decomp_period_label")
        dom_debug = diag.get("domestic_decomp_debug", {})
        dom_contrib = diag.get("domestic_decomp_contrib_df")
        dom_tri = diag.get("dom_triangulation_df")
        dom_tri_summary = diag.get("dom_triangulation_summary_df")
        dom_tri_mode = diag.get("dom_triangulation_mode")

        if isinstance(dom_sum, pd.DataFrame) and not dom_sum.empty:
            if isinstance(dom_period_label, str) and dom_period_label:
                st.subheader(f"4) Domestic sleeve decomposition: {dom_period_label}")
            else:
                st.subheader("4) Domestic sleeve decomposition")
            st.dataframe(dom_sum, use_container_width=True)

            with st.expander("Diagnostics (domestic decomposition)"):
                if isinstance(dom_debug, dict) and dom_debug:
                    st.write({
                        "start_multiple": dom_debug.get("start_multiple"),
                        "end_multiple": dom_debug.get("end_multiple"),
                    })
                if isinstance(dom_contrib, pd.DataFrame) and not dom_contrib.empty:
                    st.dataframe(dom_contrib, use_container_width=True)
                if isinstance(dom_tri_mode, str) and dom_tri_mode:
                    st.caption(f"Triangulation mode: {dom_tri_mode}")
                if isinstance(dom_tri, pd.DataFrame) and not dom_tri.empty:
                    st.markdown("**Triangulation (holding-window endpoint fundamentals)**")
                    st.dataframe(dom_tri, use_container_width=True)
                if isinstance(dom_tri_summary, pd.DataFrame) and not dom_tri_summary.empty:
                    st.dataframe(dom_tri_summary, use_container_width=True)

