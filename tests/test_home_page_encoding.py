"""Regression coverage for user-facing Unicode in active runtime modules."""

from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app13.py"


def test_home_page_source_uses_intended_unicode_labels():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "**Performance** \N{EN DASH} NAV-based rolling returns" in source
    assert 'st.button("\N{CHART WITH UPWARDS TREND} Performance")' in source
    assert 'st.button("\N{BAR CHART} Portfolio quality")' in source
    assert 'st.button("\N{BROOM} Housekeeping")' in source


def test_active_runtime_source_has_no_mojibake_markers():
    root = APP_PATH.parent
    paths = [
        APP_PATH,
    ]
    for folder in ("core", "features", "pages", "services"):
        paths.extend((root / folder).rglob("*.py"))

    markers = ("\u00c3", "\u00c2", "\u00e2", "\u00f0", "\ufffd")
    offenders = [
        str(path.relative_to(root))
        for path in paths
        if path.exists()
        and any(marker in path.read_text(encoding="utf-8") for marker in markers)
    ]
    assert not offenders, f"Mojibake markers found in: {', '.join(offenders)}"
