"""
Shared filesystem and atomic-publication infrastructure.

Provides:
- locking: process-safe file and directory coordination
- paths: canonical repository and storage-path resolution
- serialization: deterministic serialization and atomic persistence
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import common_locking as locking
    from . import common_paths as paths
    from . import common_serialization as serialization

# The queue-log CLI is executable-only and intentionally absent from the public surface.
_MODULES = {
    "locking": "common_locking",
    "paths": "common_paths",
    "serialization": "common_serialization",
}
__all__ = ["locking", "paths", "serialization"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
