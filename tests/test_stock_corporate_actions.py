import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.update_db.corporate_actions import (
    set_engine_getter,
    upload_stock_corporate_actions,
    validate_stock_corporate_actions,
)


def test_engine_resolution_has_no_dependency_on_app_bootstrap():
    source = (Path(__file__).resolve().parents[1] / "features" / "update_db" / "corporate_actions.py").read_text(encoding="utf-8")

    assert "import app13" not in source
    assert "from core.db import get_engine" in source


def test_validate_accepts_column_variants_and_dedup_keeps_last():
    df_raw = pd.DataFrame(
        [
            {
                "ISIN": "INE000A01001",
                "Ex Date": "2025-01-10",
                "Denominator (D)": 1,
                "Numerator (N)": 2,
            },
            {
                "ISIN": "INE000A01001",
                "Ex Date": "2025-01-10",
                "Denominator (D)": 3,
                "Numerator (N)": 4,
            },
            {
                "ISIN": "INE000A01002",
                "Ex Date": "2025-02-15",
                "Denominator (D)": 5,
                "Numerator (N)": 6,
            },
        ]
    )

    cleaned, summary = validate_stock_corporate_actions(df_raw, action_type="bonus")

    assert list(cleaned.columns) == ["isin", "action_type", "ex_date", "denominator", "numerator"]
    assert len(cleaned) == 2
    assert summary["rows_in_file"] == 3
    assert summary["rows_after_dedup"] == 2
    assert summary["action_type"] == "BONUS"
    assert cleaned.loc[cleaned["isin"] == "INE000A01001", "denominator"].iloc[0] == 3
    assert cleaned.loc[cleaned["isin"] == "INE000A01001", "numerator"].iloc[0] == 4


def test_validate_rejects_blank_isin():
    df_raw = pd.DataFrame(
        [{"isin": "", "ex_date": "2025-01-10", "denominator": 1, "numerator": 1}]
    )

    try:
        validate_stock_corporate_actions(df_raw, action_type="BONUS")
        assert False, "Expected ValueError for blank isin"
    except ValueError as e:
        assert "invalid rows found" in str(e).lower()


def test_validate_rejects_invalid_ex_date():
    df_raw = pd.DataFrame(
        [{"isin": "INE000A01001", "ex_date": "not-a-date", "denominator": 1, "numerator": 1}]
    )

    try:
        validate_stock_corporate_actions(df_raw, action_type="BONUS")
        assert False, "Expected ValueError for invalid ex_date"
    except ValueError as e:
        assert "invalid rows found" in str(e).lower()


def test_validate_rejects_non_positive_denominator_or_numerator():
    df_raw_1 = pd.DataFrame(
        [{"isin": "INE000A01001", "ex_date": "2025-01-10", "denominator": 0, "numerator": 1}]
    )
    df_raw_2 = pd.DataFrame(
        [{"isin": "INE000A01001", "ex_date": "2025-01-10", "denominator": 1, "numerator": -2}]
    )

    for df_raw in [df_raw_1, df_raw_2]:
        try:
            validate_stock_corporate_actions(df_raw, action_type="SPLIT")
            assert False, "Expected ValueError for invalid denominator/numerator"
        except ValueError as e:
            assert "invalid rows found" in str(e).lower()


def test_upload_overwrites_on_conflict_and_preserves_created_at():
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS fundlab")
        conn.exec_driver_sql(
            """
            CREATE TABLE fundlab.stock_corporate_action (
                isin TEXT NOT NULL,
                action_type TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                denominator INTEGER NOT NULL,
                numerator INTEGER NOT NULL,
                source_file TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (isin, action_type, ex_date)
            )
            """
        )

    set_engine_getter(lambda: engine)
    try:
        df1 = pd.DataFrame(
            [
                {
                    "isin": "INE000A01001",
                    "action_type": "BONUS",
                    "ex_date": dt.date(2025, 1, 10),
                    "denominator": 1,
                    "numerator": 2,
                }
            ]
        )
        upload_stock_corporate_actions(df1, source_file="first.xlsx")

        with engine.begin() as conn:
            row1 = conn.execute(
                sa.text(
                    """
                    SELECT denominator, numerator, source_file, created_at
                    FROM fundlab.stock_corporate_action
                    WHERE isin = :isin AND action_type = :action_type AND ex_date = :ex_date
                    """
                ),
                {"isin": "INE000A01001", "action_type": "BONUS", "ex_date": "2025-01-10"},
            ).fetchone()

        df2 = pd.DataFrame(
            [
                {
                    "isin": "INE000A01001",
                    "action_type": "BONUS",
                    "ex_date": dt.date(2025, 1, 10),
                    "denominator": 3,
                    "numerator": 5,
                }
            ]
        )
        upload_stock_corporate_actions(df2, source_file="second.xlsx")

        with engine.begin() as conn:
            row2 = conn.execute(
                sa.text(
                    """
                    SELECT denominator, numerator, source_file, created_at
                    FROM fundlab.stock_corporate_action
                    WHERE isin = :isin AND action_type = :action_type AND ex_date = :ex_date
                    """
                ),
                {"isin": "INE000A01001", "action_type": "BONUS", "ex_date": "2025-01-10"},
            ).fetchone()

        assert row1 is not None
        assert row2 is not None
        assert int(row1[0]) == 1
        assert int(row1[1]) == 2
        assert row1[2] == "first.xlsx"
        assert int(row2[0]) == 3
        assert int(row2[1]) == 5
        assert row2[2] == "second.xlsx"
        assert row2[3] == row1[3]
    finally:
        set_engine_getter(None)
