"""Regression checks for the extracted Performance feature."""

import ast
import importlib
from pathlib import Path


def test_performance_page_and_services_import_without_app_legacy_helpers():
    page = importlib.import_module("pages.performance_page")
    database = importlib.import_module("services.performance_db")
    returns = importlib.import_module("services.performance_returns")
    plots = importlib.import_module("services.performance_plots")

    assert callable(page.performance_page)
    assert callable(database.load_funds_from_db)
    assert callable(database.load_bench_from_db)
    assert callable(returns.make_rolling_df)
    assert callable(plots.plot_rolling)


def test_app13_does_not_define_extracted_performance_symbols():
    source = Path("app13.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    removed_legacy_names = {
        "_legacy_performance_page",
        "load_fund_rolling",
        "load_bench_rolling",
        "load_funds_from_db",
        "load_bench_from_db",
        "to_eom",
        "coerce_num",
        "yearly_returns_with_custom_domain",
        "parse_month_end_cell",
        "smart_to_month_end",
        "clean_nav_series",
        "_clean_funds",
        "_clean_bench",
        "make_rolling_df",
        "make_multi_fund_rolling",
        "make_multi_fund_rolling_df",
        "plot_rolling",
        "plot_multi_fund_rolling",
        "rolling_outperf_stats",
    }

    assert not (defined_names & removed_legacy_names)
    page_handlers = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PAGE_HANDLERS" for target in node.targets)
    )
    handlers = page_handlers.value
    assert isinstance(handlers, ast.Dict)
    performance_handler = next(
        value
        for key, value in zip(handlers.keys, handlers.values)
        if isinstance(key, ast.Constant) and key.value == "Performance"
    )
    assert isinstance(performance_handler, ast.Lambda)
    assert isinstance(performance_handler.body, ast.Call)
    assert isinstance(performance_handler.body.func, ast.Name)
    assert performance_handler.body.func.id == "performance_page_v2"
