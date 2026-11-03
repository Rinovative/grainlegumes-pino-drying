"""
Canonical Generation storage and campaign-evidence publication.

Provides:
- attempt: unsuccessful-attempt retention and evidence admission
- campaign_evidence: terminal campaign and transfer evidence admission
- catalog: bounded case-outcome and exact raw-source inspection
- storage: canonical HDF5 conversion and admission
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_publication_attempt as attempt
    from . import generation_publication_campaign_evidence as campaign_evidence
    from . import generation_publication_catalog as catalog
    from . import generation_publication_storage as storage

_MODULES = {
    "attempt": "generation_publication_attempt",
    "campaign_evidence": "generation_publication_campaign_evidence",
    "catalog": "generation_publication_catalog",
    "storage": "generation_publication_storage",
}
__all__ = ["attempt", "campaign_evidence", "catalog", "storage"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
