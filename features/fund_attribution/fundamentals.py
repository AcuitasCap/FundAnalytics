"""Stock fundamental growth across actual holding endpoints."""

import math

import pandas as pd

from core.dataframes import ensure_unique_monthly_rows
from core.dates import to_month_end


METRICS = {"PAT growth": "ttm_pat", "Sales growth": "ttm_sales", "Book value growth": "book_value"}


def _format_growth(start, end, years, multiple_trades=False):
    prefix = "Multiple trades, " if multiple_trades else ""
    if not (math.isfinite(start) and math.isfinite(end)):
        return f"{prefix}N/A"
    if start == 0:
        return f"{prefix}N/A (zero start)"
    if start < 0 and end < 0:
        losses = "Multiple trades, losses" if multiple_trades else "Losses"
        if end > start:
            return f"{losses}, improving"
        if end < start:
            return f"{losses}, worsening"
        return f"{losses}, unchanged"
    if end < 0:
        return "Multiple trades, turned to losses" if multiple_trades else "Turned to losses"
    gross = 1 + (end - start) / abs(start)
    suffix = " end to end growth" if multiple_trades else ""
    return f"{prefix}{gross ** (1 / years) - 1:.2%}{suffix}"


def build_fundamental_growth_table(windows, values):
    """Annualize endpoint growth over the full span, including gaps between trades."""
    columns = ["Stock name", "Average weight (%)", *METRICS]
    if not windows:
        return pd.DataFrame(columns=columns)

    values = values.copy()
    keys = ["ace_code", "month_end"]
    values["month_end"] = pd.to_datetime(values["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
    for metric in METRICS.values():
        values[metric] = pd.to_numeric(values[metric], errors="coerce").astype(float)
    values = ensure_unique_monthly_rows(values, keys, list(METRICS.values()), "Attribution fundamentals")
    lookup = values.set_index(keys).to_dict("index")

    rows = []
    for window in sorted(windows, key=lambda item: (-item["average_weight_pct"], item["stock_name"])):
        first_month = to_month_end(window["first_month"])
        last_month = to_month_end(window["last_month"])
        span_months = (last_month.year - first_month.year) * 12 + last_month.month - first_month.month
        multiple_trades = span_months > window["holding_months"]
        code = window["ace_code"]
        start = lookup.get((code, first_month), {})
        end = lookup.get((code, last_month), {})
        rows.append({
            "instrument_key": window["instrument_key"],
            "Stock name": window["stock_name"],
            "Average weight (%)": window["average_weight_pct"],
            **{label: _format_growth(start.get(metric, math.nan), end.get(metric, math.nan),
                                     span_months / 12, multiple_trades)
               for label, metric in METRICS.items()},
        })
    return pd.DataFrame(rows, columns=["instrument_key", *columns]).set_index("instrument_key")
