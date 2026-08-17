import ast
import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_attribution_feature_imports_without_creating_an_engine(monkeypatch):
    import core.db

    monkeypatch.setattr(
        core.db,
        "get_engine",
        lambda: (_ for _ in ()).throw(AssertionError("Attribution import must not create an engine")),
    )
    for module_name in (
        "features.fund_attribution.data",
        "features.fund_attribution.compute",
        "features.fund_attribution.display",
        "features.fund_attribution.page",
        "features.fund_attribution.smoke",
    ):
        importlib.import_module(module_name)


def test_router_uses_extracted_attribution_page_and_legacy_functions_are_removed():
    import app13
    from features.fund_attribution.page import fund_attribution_page

    assert app13.PAGE_HANDLERS["Fund attribution"] is fund_attribution_page
    tree = ast.parse((ROOT / "app13.py").read_text(encoding="utf-8"))
    functions = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not {
        "fund_attribution_page",
        "_load_attrib_raw_window",
        "_compute_attribution",
        "_compute_domestic_triangulation_tables",
        "_run_attrib_smoke_test",
    } & functions


def test_attribution_layers_preserve_the_ui_and_dependency_contract():
    data_source = (ROOT / "features/fund_attribution/data.py").read_text(encoding="utf-8")
    compute_source = (ROOT / "features/fund_attribution/compute.py").read_text(encoding="utf-8")
    display_source = (ROOT / "features/fund_attribution/display.py").read_text(encoding="utf-8")
    page_source = (ROOT / "features/fund_attribution/page.py").read_text(encoding="utf-8")

    assert "@st.cache_data(ttl=3600, show_spinner=False)" in data_source
    assert "def _load_attrib_raw_window(" in data_source
    assert "def _compute_attribution(" in compute_source
    assert "def _compute_domestic_triangulation_tables(" in compute_source
    assert "get_engine" not in compute_source
    assert "streamlit" not in compute_source
    assert "def display_attribution_results(" in display_source
    assert "from core.fund_catalog import fetch_categories, fetch_funds_for_categories" in page_source
    for value in (
        "fa_category", "fa_focus_fund", "fa_bench_mode", "fa_universe_mode",
        "fa_form", "fa_start_year", "fa_start_month", "fa_end_year", "fa_end_month",
        "fa_lookback", "fa_lens", "fa_dom_triangulation_mode",
        "Vs NIFTY 50", "Vs NIFTY 500", "Like-for-like",
        "Domestic equities only", "Full holdings", "Earnings (P/E)", "Sales (P/S)",
        "Book (P/B)", "Fallback to price growth", "Exclude invalid from weighted average",
    ):
        assert value in page_source


def test_attribution_smoke_check_validates_live_result_structure_not_frozen_market_values():
    smoke_source = (ROOT / "features/fund_attribution/smoke.py").read_text(encoding="utf-8")

    assert "structural smoke test" in smoke_source
    assert "required_domestic_metrics" in smoke_source
    assert "exp_dom" not in smoke_source
    assert "exp_bench" not in smoke_source
    assert "exp_winners" not in smoke_source
