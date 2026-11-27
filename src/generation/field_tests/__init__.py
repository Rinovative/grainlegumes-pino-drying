"""
Per-parameter Generation field tests: frozen rosters run as static node-job steps outside every Source.

Provides:
- dispatch: gate-first node-job submission, scheduler reconciliation, status, and cancellation
- outputs: case.h5 dataset access and completion evidence shared by the gate and the analysis
- planning: stage-A screening and stage-B confirmation roster builders
- record: record paths, write-once attempt outcomes, member seals, the effective members, and member and dispatch locks
- roster: roster schema, digest, persistence, and base Source binding
- runner: one member's step from gate check and binding to COMSOL and seal
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_field_tests_dispatch as dispatch
    from . import generation_field_tests_outputs as outputs
    from . import generation_field_tests_planning as planning
    from . import generation_field_tests_record as record
    from . import generation_field_tests_roster as roster
    from . import generation_field_tests_runner as runner

_MODULES = {
    "dispatch": "generation_field_tests_dispatch",
    "outputs": "generation_field_tests_outputs",
    "planning": "generation_field_tests_planning",
    "record": "generation_field_tests_record",
    "roster": "generation_field_tests_roster",
    "runner": "generation_field_tests_runner",
}
__all__ = ["dispatch", "outputs", "planning", "record", "roster", "runner"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
