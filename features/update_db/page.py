from __future__ import annotations

import hashlib
import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from core.fund_catalog import fetch_categories
from core.navigation import home_button
from .corporate_actions import validate_stock_corporate_actions
from .data import (
    CompanyNameAceConflictError, MissingPortfolioAceCodesError, _read_any,
    find_company_name_ace_conflicts, find_missing_portfolio_ace_codes,
    load_ace_industry_flags, load_existing_fund_names, load_stock_master_isins,
)
from .display import (
    display_ace_stock_master_next_steps, display_company_name_ace_conflicts,
    display_missing_portfolio_ace_codes, display_upload_success,
    display_validation_success, show_expected_format,
)
from .validation import (
    validate_ace_stock_master, validate_annual_book_value, validate_bench_navs,
    validate_fund_manager_tenure,
    validate_fund_navs, validate_fund_portfolios, validate_quarterly_pat,
    validate_quarterly_sales, validate_roe_roce, validate_stock_dividends,
    validate_stock_master, validate_stock_prices_mc,
)
from .writes import (
    upload_ace_stock_master, upload_annual_book_value, upload_bench_navs,
    upload_corporate_actions,
    upload_fund_manager_tenure, upload_fund_navs, upload_fund_portfolios,
    upload_quarterly_pat, upload_quarterly_sales, upload_roe_roce,
    upload_stock_dividends, upload_stock_master, upload_stock_prices_mc,
)

def update_db_page():
    home_button()
    st.header("Update underlying data")

    upload_type = st.selectbox(
        "What would you like to update???",
        [
            "Fund NAVs",
            "Benchmark NAVs",
            "Fund portfolios",
            "Fund manager tenure",
            "Stock ISIN, industry, financial/non-financial",
            "Update stock master",
            "Company RoE / RoCE",
            "Stock prices and market cap",
            "Stock dividends (DPS + Ex-date)",
            "Stock corporate actions (Bonus/Split)",
            "Company PAT (quarterly)",
            "Company sales (quarterly)",
            "Company book value (annual)",
        ],
    )

    corp_action_type = None
    if upload_type == "Stock corporate actions (Bonus/Split)":
        corp_action_type = st.radio(
            "Corporate action type",
            options=["BONUS", "SPLIT"],
            key="update_db_corp_action_type",
            horizontal=True,
        )

    upload_file_types = ["xlsx"]
    if upload_type == "Stock corporate actions (Bonus/Split)":
        upload_file_types = ["csv", "xlsx", "xls"]

    uploaded = st.file_uploader(
        "Upload file",
        type=upload_file_types,
        key=f"upload_{upload_type}",
    )

    show_expected_format(upload_type)

    if not uploaded:
        st.info("Please upload the appropriate Excel file to continue.")
        return

    try:
        if upload_type == "Stock corporate actions (Bonus/Split)":
            df_raw = _read_any(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return

    state_key_df = f"validated_df_{upload_type}"
    state_key_ok = f"validated_ok_{upload_type}"
    validated_file_hash = None
    hash_key = f"validated_file_hash_{upload_type}"
    if upload_type in {"Fund portfolios", "Update stock master", "Company RoE / RoCE"}:
        validated_file_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get(hash_key) != validated_file_hash:
            st.session_state[state_key_ok] = False
            if upload_type == "Update stock master":
                st.session_state.pop("confirm_ace_industry_classifications", None)

    if st.button("Validate (dry run)"):
        try:
            if upload_type == "Fund NAVs":
                df_clean, summary = validate_fund_navs(df_raw)
            elif upload_type == "Benchmark NAVs":
                df_clean, summary = validate_bench_navs(df_raw)
            elif upload_type == "Fund portfolios":
                df_clean, summary = validate_fund_portfolios(df_raw)
                missing = find_missing_portfolio_ace_codes(df_clean)
                if not missing.empty:
                    raise MissingPortfolioAceCodesError(missing)
            elif upload_type == "Stock ISIN, industry, financial/non-financial":
                df_clean, summary = validate_stock_master(df_raw)
            elif upload_type == "Update stock master":
                df_clean, summary = validate_ace_stock_master(df_raw)
                name_conflicts = find_company_name_ace_conflicts(df_clean)
                if not name_conflicts.empty:
                    raise CompanyNameAceConflictError(name_conflicts)
                known_industries = load_ace_industry_flags()
                summary["new_industries"] = (
                    df_clean.loc[
                        ~df_clean["industry_key"].isin(set(known_industries)),
                        ["industry_key", "industry"],
                    ]
                    .drop_duplicates("industry_key")
                    .sort_values("industry")
                    .to_dict("records")
                )
            elif upload_type == "Company RoE / RoCE":
                df_clean, summary = validate_roe_roce(df_raw)
            elif upload_type == "Stock prices and market cap":
                df_clean, summary = validate_stock_prices_mc(df_raw, valid_isins=load_stock_master_isins())
            elif upload_type == "Stock dividends (DPS + Ex-date)":
                # EXPECTED output columns: ['isin','ex_date','dps']
                df_clean, summary = validate_stock_dividends(df_raw)
            elif upload_type == "Stock corporate actions (Bonus/Split)":
                df_clean, summary = validate_stock_corporate_actions(
                    df_raw=df_raw,
                    action_type=corp_action_type,
                )
            elif upload_type == "Company PAT (quarterly)":
                df_clean, summary = validate_quarterly_pat(df_raw)
            elif upload_type == "Company sales (quarterly)":
                df_clean, summary = validate_quarterly_sales(df_raw)
            elif upload_type == "Company book value (annual)":
                df_clean, summary = validate_annual_book_value(df_raw)
            elif upload_type == "Fund manager tenure":
                df_clean, summary = validate_fund_manager_tenure(
                    df_raw, existing_fund_names=load_existing_fund_names()
                )
            else:
                st.error("Unsupported upload type.")
                return

            st.session_state[state_key_df] = df_clean
            st.session_state[state_key_ok] = True
            st.session_state[f"validated_summary_{upload_type}"] = summary
            if validated_file_hash is not None:
                st.session_state[hash_key] = validated_file_hash

            display_validation_success(summary)

        except MissingPortfolioAceCodesError as exc:
            display_missing_portfolio_ace_codes(exc.report)
            st.session_state[state_key_ok] = False
            return
        except CompanyNameAceConflictError as exc:
            display_company_name_ace_conflicts(exc.report)
            st.session_state[state_key_ok] = False
            return
        except ValueError as ve:
            st.error(f"Validation error: {ve}")
            st.session_state[state_key_ok] = False
            return
        except Exception as e:
            st.error(f"Unexpected error during validation: {e}")
            st.session_state[state_key_ok] = False
            return

    if st.session_state.get(state_key_ok):
        df_clean = st.session_state.get(state_key_df)
        summary = st.session_state.get(f"validated_summary_{upload_type}", {}) or {}

        resolutions = {}
        industry_classifications = {}
        can_proceed = True
        block_reason = ""

        if upload_type == "Fund manager tenure":
            missing = summary.get("missing_funds", []) or []

            if missing:
                st.warning(
                    "Some funds in this upload are not present in the fund master. "
                    "Resolve them below before uploading."
                )

                existing_names = load_existing_fund_names()

                categories = fetch_categories()
                if not categories:
                    can_proceed = False
                    block_reason = "No categories available in fundlab.category (excluding 'portfolio')."

                with st.expander("Resolve missing fund names (rename or create)", expanded=True):
                    for new_name in missing:
                        st.markdown(f"**{new_name}**")
                        c1, c2 = st.columns([2, 2])

                        old_name = c1.text_input(
                            "Old name (optional – fill only if this is a rename)",
                            key=f"fm_old_{new_name}",
                            placeholder="e.g., ICICI Value Discovery",
                        ).strip()

                        if old_name == "":
                            cat_sel = c2.selectbox(
                                "Category (required for new fund)",
                                options=["-- Select --"] + categories,
                                key=f"fm_cat_{new_name}",
                            )
                            category_name = None if cat_sel == "-- Select --" else cat_sel

                            if category_name is None:
                                can_proceed = False
                                block_reason = f"Select a category for new fund: {new_name}"
                        else:
                            c2.write("Category not required for rename.")
                            category_name = None

                            if old_name not in existing_names:
                                can_proceed = False
                                block_reason = (
                                    f"Old name not found in fund master: '{old_name}' "
                                    f"(for rename to '{new_name}')"
                                )

                        resolutions[new_name] = {"old_name": old_name, "category_name": category_name}

        if upload_type == "Update stock master":
            new_industries = summary.get("new_industries", [])
            if new_industries:
                st.warning("Classify each new industry before updating the ACE stock master.")
                for item in new_industries:
                    industry_key = item["industry_key"]
                    radio_key = "ace_industry_" + hashlib.sha256(
                        industry_key.encode("utf-8")
                    ).hexdigest()[:16]
                    choice = st.radio(
                        f"Is '{item['industry']}' financial or non-financial?",
                        ["-- Select --", "Financial", "Non-financial"],
                        key=radio_key,
                        horizontal=True,
                    )
                    if choice == "-- Select --":
                        can_proceed = False
                        block_reason = "Choose a classification for every new industry."
                    else:
                        industry_classifications[industry_key] = choice == "Financial"
                if not st.checkbox(
                    "I confirm these new industry classifications",
                    key="confirm_ace_industry_classifications",
                ):
                    can_proceed = False
                    if not block_reason:
                        block_reason = "Confirm the new industry classifications before upload."

        if not can_proceed:
            st.error(block_reason)
            return

        if st.button("Confirm upload to database"):
            if df_clean is None:
                st.error("No validated data found in session. Please run validation again.")
                return

            try:
                if upload_type == "Fund NAVs":
                    upload_fund_navs(df_clean)
                elif upload_type == "Benchmark NAVs":
                    upload_bench_navs(df_clean)
                elif upload_type == "Fund portfolios":
                    upload_fund_portfolios(df_clean)
                elif upload_type == "Stock ISIN, industry, financial/non-financial":
                    upload_stock_master(df_clean)
                elif upload_type == "Update stock master":
                    upload_ace_stock_master(df_clean, industry_classifications)
                elif upload_type == "Company RoE / RoCE":
                    upload_roe_roce(df_clean)
                elif upload_type == "Stock prices and market cap":
                    upload_stock_prices_mc(df_clean)
                elif upload_type == "Stock dividends (DPS + Ex-date)":
                    upload_stock_dividends(df_clean)
                elif upload_type == "Stock corporate actions (Bonus/Split)":
                    summary = upload_corporate_actions(
                        df=df_clean,
                        source_file=getattr(uploaded, "name", None),
                    )
                    display_upload_success(summary)
                    return
                elif upload_type == "Company PAT (quarterly)":
                    upload_quarterly_pat(df_clean)
                elif upload_type == "Company sales (quarterly)":
                    upload_quarterly_sales(df_clean)
                elif upload_type == "Company book value (annual)":
                    upload_annual_book_value(df_clean)
                elif upload_type == "Fund manager tenure":
                    upload_fund_manager_tenure(df_clean, resolutions)

                display_upload_success()
                if upload_type == "Update stock master":
                    display_ace_stock_master_next_steps()

            except MissingPortfolioAceCodesError as exc:
                display_missing_portfolio_ace_codes(exc.report)
            except CompanyNameAceConflictError as exc:
                display_company_name_ace_conflicts(exc.report)
            except ValueError as exc:
                st.error(f"Upload blocked: {exc}")
            except SQLAlchemyError as e:
                msg = str(getattr(e, "orig", e))
                lower_msg = msg.lower()

                ace_upload = upload_type in {
                    "Company book value (annual)",
                    "Company PAT (quarterly)",
                    "Company sales (quarterly)",
                    "Company RoE / RoCE",
                }
                if ace_upload and "foreign key constraint" in lower_msg:
                    st.error(
                        "Some ACE codes in this file do not exist in ACE Stock Master. "
                        f"Database message: {msg}"
                    )
                elif ace_upload and (
                    "duplicate key value" in lower_msg or "unique constraint" in lower_msg
                ):
                    period_label = (
                        "year-end"
                        if upload_type in {"Company book value (annual)", "Company RoE / RoCE"}
                        else "quarter-end"
                    )
                    st.error(
                        f"These ACE code + {period_label} combinations already exist. "
                        f"Database message: {msg}"
                    )
                elif "foreign key constraint" in lower_msg:
                    st.error(
                        "❌ Some ISINs in this file do not exist in Stock Master.\n"
                        "Please update Stock Master before uploading this data.\n\n"
                        f"Database message: {msg}"
                    )
                elif "duplicate key value" in lower_msg or "unique constraint" in lower_msg:
                    st.error(
                        "❌ Duplicate rows detected against existing database records..\n"
                        "These ISIN + period combinations already exist.\n\n"
                        f"Database message: {msg}"
                    )
                else:
                    st.error(
                        f"Database error while uploading:\n"
                        f"{e.__class__.__name__}: {msg}"
                    )
            except Exception as e:
                st.error(f"Unexpected error during upload: {e}")
