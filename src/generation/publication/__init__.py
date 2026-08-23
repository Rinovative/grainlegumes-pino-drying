"""
Canonical Generation storage, inventory, and campaign-evidence publication.

Provides:
- attempt: unsuccessful-attempt retention and replay evidence admission
- campaign_evidence: terminal campaign and transfer evidence admission
- completion_composite: partial-parent and replacement composite evidence admission
- inventory: parameter ownership and effective-consumer inspection
- storage: canonical HDF5 conversion and admission
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_publication_attempt as attempt
    from . import generation_publication_campaign_evidence as campaign_evidence
    from . import generation_publication_completion_composite as completion_composite
    from . import generation_publication_inventory as inventory
    from . import generation_publication_storage as storage

_MODULES = {
    "attempt": "generation_publication_attempt",
    "campaign_evidence": "generation_publication_campaign_evidence",
    "completion_composite": "generation_publication_completion_composite",
    "inventory": "generation_publication_inventory",
    "storage": "generation_publication_storage",
}
__all__ = ["attempt", "campaign_evidence", "completion_composite", "inventory", "storage"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
