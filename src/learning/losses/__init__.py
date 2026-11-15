"""Supervised, physics-informed, and transient loss services.

Provides:
- factory: semantic training-loss construction
- pino: physics-informed loss composition
- state: group-weighted MSE or absolute H1 state loss of both tasks
- transient: transient drying loss composition
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import learning_losses_factory as factory
    from . import learning_losses_pino as pino
    from . import learning_losses_state as state
    from . import learning_losses_transient as transient

_MODULES = {
    "factory": "learning_losses_factory",
    "pino": "learning_losses_pino",
    "state": "learning_losses_state",
    "transient": "learning_losses_transient",
}
__all__ = ["factory", "pino", "state", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
