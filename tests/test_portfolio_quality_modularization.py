import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_quality_page_is_registered_from_feature_package():
    tree = ast.parse(Path("app13.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "features.portfolio_quality.page" and any(alias.name == "portfolio_quality_page" for alias in node.names) for node in imports)


def test_quality_feature_has_one_way_dependencies():
    for path in Path("features/portfolio_quality").glob("*.py"):
        assert "import app13" not in path.read_text(encoding="utf-8")


def test_quality_legacy_implementation_was_removed_from_app_bootstrap():
    tree = ast.parse(Path("app13.py").read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(name.startswith("_legacy_") and "quality" in name for name in functions)


def test_quality_layers_have_the_agreed_responsibilities():
    compute_source = Path("features/portfolio_quality/compute.py").read_text(encoding="utf-8")
    data_source = Path("features/portfolio_quality/data.py").read_text(encoding="utf-8")
    display_source = Path("features/portfolio_quality/display.py").read_text(encoding="utf-8")
    assert "def compute_quality_analysis(" in compute_source
    assert "streamlit" not in compute_source
    assert "get_engine" not in compute_source
    assert "def fetch_portfolio_raw(" in data_source
    assert "def fetch_quality_bucket_rows(" in data_source
    assert "def display_quality_analysis(" in display_source
