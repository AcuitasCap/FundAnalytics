"""Stock fundamental growth across actual holding endpoints."""

import math
from io import BytesIO

import pandas as pd

from core.dataframes import ensure_unique_monthly_rows
from core.dates import to_month_end


FUNDAMENTAL_FIELDS = {"PAT": "ttm_pat", "Sales": "ttm_sales", "Book value": "book_value"}
METRICS = {f"{label} growth": field for label, field in FUNDAMENTAL_FIELDS.items()}


def _format_growth(start, end, years, multiple_trades=False, *, first_month, last_month):
    prefix = "Multiple trades, " if multiple_trades else ""
    missing_months = [
        month.strftime("%b %Y")
        for value, month in ((start, first_month), (end, last_month))
        if not math.isfinite(value)
    ]
    if missing_months:
        label = "missing data for" if multiple_trades else "Missing data for"
        return f"{prefix}{label} {' and '.join(missing_months)}"
    if start == 0:
        return f"{prefix}0 to {end:,.2f} (zero starting value)"
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
            **{
                label: _format_growth(
                    start.get(metric, math.nan), end.get(metric, math.nan),
                    span_months / 12, multiple_trades, first_month=first_month, last_month=last_month,
                )
                for label, metric in METRICS.items()
            },
        })
    return pd.DataFrame(rows, columns=["instrument_key", *columns]).set_index("instrument_key")


def build_fundamental_history_workbook(stocks, start_month, end_month, metric, values) -> bytes:
    """Export the source values with a calendar month column and a row for every stock."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    months = pd.period_range(start_month, end_month, freq="M")
    values = values.copy()
    values["month"] = pd.to_datetime(values["month_end"]).dt.to_period("M")
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    keys = ["ace_code", "month"]
    values = ensure_unique_monthly_rows(values, keys, ["value"], "Fundamental history export")
    lookup = values.set_index(keys)["value"].dropna().to_dict()

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = metric
    sheet.freeze_panes = "B2"
    sheet.append(["Stock name", *months.to_timestamp("M").date])
    longest_name = max((len(name) for name, _ in stocks), default=0)
    sheet.column_dimensions["A"].width = max(18, min(longest_name + 3, 255))
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="243B53")
        cell.font = Font(bold=True, color="FFFFFF")
        if cell.column > 1:
            cell.number_format = "mmm yyyy"
            sheet.column_dimensions[get_column_letter(cell.column)].width = 16

    for name, code in sorted(stocks, key=lambda stock: stock[0].casefold()):
        sheet.append([name, *(lookup.get((code, month)) for month in months)])
        row = sheet[sheet.max_row]
        row[0].data_type = "s"  # Stock names are literal text, even if a name starts with '='.
        for cell in row[1:]:
            cell.number_format = "#,##0.00"
    sheet.auto_filter.ref = sheet.dimensions

    with BytesIO() as output:
        workbook.save(output)
        return output.getvalue()
