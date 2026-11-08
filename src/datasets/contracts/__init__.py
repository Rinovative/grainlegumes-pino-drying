"""
Dataset view, identity, and transient channel contracts.

Provides:
- identity: deterministic Dataset and membership identities
- partition: immutable Dataset case roles and exact fixed membership
- transient: canonical transient sample-data contract
- views: typed buildable Dataset-view registry
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_contracts_identity as identity
    from . import dataset_contracts_partition as partition
    from . import dataset_contracts_transient as transient
    from . import dataset_contracts_views as views

_MODULES = {
    "identity": "dataset_contracts_identity",
    "partition": "dataset_contracts_partition",
    "transient": "dataset_contracts_transient",
    "views": "dataset_contracts_views",
}
__all__ = ["identity", "partition", "transient", "views"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
