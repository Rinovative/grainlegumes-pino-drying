"""
Strict experiment defaults and YAML configuration services.

Provides:
- defaults: canonical experiment defaults
- loader: YAML admission, resolution, and semantic validation
- preflight: resolved configuration and runtime-device preflight
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import experiments_config_defaults as defaults
    from . import experiments_config_loader as loader
    from . import experiments_config_preflight as preflight

_MODULES = {
    "defaults": "experiments_config_defaults",
    "loader": "experiments_config_loader",
    "preflight": "experiments_config_preflight",
}
__all__ = ["defaults", "loader", "preflight"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
