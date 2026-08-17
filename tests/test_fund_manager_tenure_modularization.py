import ast
from pathlib import Path


def test_manager_tenure_router_uses_feature_page():
    source = Path("app13.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and node.module == "features.fund_manager_tenure.page"
        for alias in node.names
    }
    assert "fund_manager_tenure_page" in imported
    assert '"Fund manager tenure": fund_manager_tenure_page' in source
    assert "_legacy_fund_manager_tenure_page" not in source
    assert "def fund_manager_tenure_page" not in source


def test_compute_layer_has_no_streamlit_or_database_imports():
    source = Path("features/fund_manager_tenure/compute.py").read_text(encoding="utf-8")
    assert "streamlit" not in source
    assert "sqlalchemy" not in source
