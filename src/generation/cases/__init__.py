"""
Deterministic Generation case planning, sampling, fields, and schedules.

Provides:
- case: scientific case identities and input-bundle construction
- config: resolved campaign and batch configuration
- fields: deterministic spatial field generation
- sampling: case-level parameter sampling and OOD allocation
- schedule: deterministic transient boundary schedules
- seeding: stable seed derivation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_cases_case as case
    from . import generation_cases_config as config
    from . import generation_cases_fields as fields
    from . import generation_cases_sampling as sampling
    from . import generation_cases_schedule as schedule
    from . import generation_cases_seeding as seeding

_MODULES = {
    "case": "generation_cases_case",
    "config": "generation_cases_config",
    "fields": "generation_cases_fields",
    "sampling": "generation_cases_sampling",
    "schedule": "generation_cases_schedule",
    "seeding": "generation_cases_seeding",
}
__all__ = ["case", "config", "fields", "sampling", "schedule", "seeding"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
