# ruff: noqa: S101
"""Verify the host-compatible guard around the authoritative project preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

    from _pytest.monkeypatch import MonkeyPatch

_REPOSITORY_ROOT = Path(__file__).parents[2]
_RUNTIME_PATH = _REPOSITORY_ROOT / "scripts" / "config_preflight_runtime.py"


def _load_runtime() -> ModuleType:
    """Load the standalone guard without importing the application package."""
    spec = importlib.util.spec_from_file_location("config_preflight_runtime_under_test", _RUNTIME_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("runtime_version", [(3, 8, 20), (3, 9, 19)])
def test_unsupported_runtime_fails_before_project_import(
    runtime_version: tuple[int, int, int],
    monkeypatch: MonkeyPatch,
) -> None:
    """Fail before importing project code on an unsupported host runtime."""
    runtime = _load_runtime()

    def unexpected_import(*_args: object, **_kwargs: object) -> None:
        pytest.fail("project import ran before the minimum-version guard")

    monkeypatch.setattr(runtime.sys, "version_info", runtime_version)
    monkeypatch.setattr(runtime.runpy, "run_module", unexpected_import)

    assert runtime.main() == 1
