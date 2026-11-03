"""
Experiment configuration, execution, tracking, tuning, and validation services.

Provides:
- config: strict defaults and YAML configuration resolution
- console: persistent line-oriented experiment lifecycle reporting
- identity: exact parent metadata and read-only child navigation
- notebook_support: read-only notebook context and presentation preparation
- registry: one run directory per experiment identity
- run: experiment lifecycle, identity, persistence, and resume
- source_binding: the frozen source tree in which an experiment unit last ran
- tracking: W&B observer lifecycle and metric publication
- tuning: search-space and Optuna-study services
- validation: explicit production data-pipeline validation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import config, tuning, validation
    from . import experiments_console as console
    from . import experiments_notebook_support as notebook_support
    from . import experiments_run as run
    from . import experiments_run_identity as identity
    from . import experiments_run_registry as registry
    from . import experiments_run_source_binding as source_binding
    from . import experiments_tracking as tracking

# The cli subpackage is executable-only and intentionally absent from the public surface.
_MODULES = {
    "config": "config",
    "console": "experiments_console",
    "identity": "experiments_run_identity",
    "notebook_support": "experiments_notebook_support",
    "registry": "experiments_run_registry",
    "run": "experiments_run",
    "source_binding": "experiments_run_source_binding",
    "tracking": "experiments_tracking",
    "tuning": "tuning",
    "validation": "validation",
}
__all__ = [
    "config",
    "console",
    "identity",
    "notebook_support",
    "registry",
    "run",
    "source_binding",
    "tracking",
    "tuning",
    "validation",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
