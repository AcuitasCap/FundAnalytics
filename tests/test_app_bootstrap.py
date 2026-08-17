"""Smoke coverage for app bootstrap and page routing."""

import importlib
import subprocess
import sys
from pathlib import Path

import streamlit  # Preload dependency; this test guards app13's own import path.

sys.path.append(str(Path(__file__).resolve().parents[1]))
EXPECTED_PAGES = {
    "Home",
    "Performance",
    "Portfolio quality",
    "Portfolio valuations",
    "Fund attribution",
    "Portfolio",
    "Fund manager tenure",
    "Update DB",
    "Housekeeping",
}


def _import_app():
    return importlib.import_module("app13")


def test_app_import_does_not_construct_engine_or_run_subprocesses(monkeypatch):
    import core.db as db

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Import-time subprocess calls are not allowed")

    monkeypatch.setattr(subprocess, "check_output", fail_if_called)
    monkeypatch.setattr(db, "create_engine", fail_if_called)
    monkeypatch.setattr(db, "_engine", None)
    sys.modules.pop("app13", None)
    module = _import_app()

    assert not hasattr(module, "engine")
    assert db._engine is None


def test_router_registers_all_pages():
    module = _import_app()
    assert set(module.PAGE_HANDLERS) == EXPECTED_PAGES
    assert all(callable(handler) for handler in module.PAGE_HANDLERS.values())


def test_dispatch_page_uses_registered_handler_and_unknown_pages_fall_back_home(monkeypatch):
    module = _import_app()
    calls = []
    monkeypatch.setattr(module, "PAGE_HANDLERS", {"Known": lambda: calls.append("known")})
    monkeypatch.setattr(module, "home_page", lambda: calls.append("home"))

    module.dispatch_page("Known")
    module.dispatch_page("Missing")

    assert calls == ["known", "home"]
