"""
Scientific field, moisture, permeability, physics, and task contracts.

Provides:
- field_sets: named input and output field collections
- fields: field definitions, units, and tensor semantics
- moisture: canonical dry-basis, wet-basis, and bulk-moisture conversions
- permeability: permeability representations and validation
- physics: boundary, derivative, and Brinkman operator services
- tasks: registered scientific task specifications
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import domain_field_sets as field_sets
    from . import domain_fields as fields
    from . import domain_moisture as moisture
    from . import domain_permeability as permeability
    from . import physics, tasks

_MODULES = {
    "field_sets": "domain_field_sets",
    "fields": "domain_fields",
    "moisture": "domain_moisture",
    "permeability": "domain_permeability",
    "physics": "physics",
    "tasks": "tasks",
}
__all__ = ["field_sets", "fields", "moisture", "permeability", "physics", "tasks"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
