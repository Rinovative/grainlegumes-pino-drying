"""
Registered scientific task specifications.

Provides:
- registry: task registration and lookup
- spec: immutable task-schema definitions
- steady_flow: maintained steady-flow task contract
- transient_drying: maintained transient grain-drying task contract
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import domain_task_registry as registry
    from . import domain_task_spec as spec
    from . import domain_task_steady_flow as steady_flow
    from . import domain_task_transient_drying as transient_drying

_MODULES = {
    "registry": "domain_task_registry",
    "spec": "domain_task_spec",
    "steady_flow": "domain_task_steady_flow",
    "transient_drying": "domain_task_transient_drying",
}
__all__ = ["registry", "spec", "steady_flow", "transient_drying"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
