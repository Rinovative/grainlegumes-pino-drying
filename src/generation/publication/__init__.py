"""
Canonical Generation storage, inventory, and campaign-evidence publication.

Provides:
- campaign_evidence: terminal campaign and transfer evidence admission
- inventory: parameter ownership and effective-consumer inspection
- storage: canonical HDF5 conversion and admission
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_publication_campaign_evidence as campaign_evidence
    from . import generation_publication_inventory as inventory
    from . import generation_publication_storage as storage

_MODULES = {
    "campaign_evidence": "generation_publication_campaign_evidence",
    "inventory": "generation_publication_inventory",
    "storage": "generation_publication_storage",
}
__all__ = ["campaign_evidence", "inventory", "storage"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
