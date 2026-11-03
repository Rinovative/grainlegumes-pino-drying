"""Deterministic scientific validation and severity policies.

Provides:
- policy: shared blocking-integrity and advisory-diagnostic severity contract
- sentinels: bounded no-COMSOL sentinel workload planning
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_validation_policy as policy
    from . import generation_validation_sentinels as sentinels

_MODULES = {
    "policy": "generation_validation_policy",
    "sentinels": "generation_validation_sentinels",
}


__all__ = ["policy", "sentinels"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
