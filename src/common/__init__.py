"""
Shared filesystem and atomic-publication infrastructure.

Provides:
- code_identity: declared scientific-code projections and their syntax digests
- identity_registry: one-owner bindings of content digests to existing targets
- locking: process-safe file and directory coordination
- names: compact path-safe number and text name tokens
- paths: canonical repository and storage-path resolution
- random: deterministic independent seed streams
- serialization: deterministic serialization and atomic persistence
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import common_code_identity as code_identity
    from . import common_identity_registry as identity_registry
    from . import common_locking as locking
    from . import common_names as names
    from . import common_paths as paths
    from . import common_random as random
    from . import common_serialization as serialization

_MODULES = {
    "code_identity": "common_code_identity",
    "identity_registry": "common_identity_registry",
    "locking": "common_locking",
    "names": "common_names",
    "paths": "common_paths",
    "random": "common_random",
    "serialization": "common_serialization",
}
__all__ = ["code_identity", "identity_registry", "locking", "names", "paths", "random", "serialization"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
