"""
Reference-simulation responsibility packages and orchestration services.

Provides:
- benchmark: isolated transient COMSOL resource-scaling evidence
- benchmark_config: validated resource-scaling suites and immutable case selections
- benchmark_report: resource-scaling interpretation and report rendering
- campaign: campaign planning, execution evidence, and finalization
- completion: bounded supplemental completion planning for partial campaigns
- controller: service-backed workflow execution and continuation
- cases: deterministic case planning and construction services
- contracts: immutable scientific vocabularies and registries
- publication: canonical storage and terminal-evidence publication
- readiness: fail-closed production-readiness reporting
- run: immutable campaign, benchmark, and workflow run plans
- runtime: native solver, workspace, and batch execution services
- smoke: paired technical-runtime smoke evidence
- validation: pilot and deterministic scientific validation services
- workflow: transfer, package, retention, and cleanup lifecycle
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cases, contracts, publication, runtime, validation
    from . import generation_benchmark as benchmark
    from . import generation_benchmark_config as benchmark_config
    from . import generation_benchmark_report as benchmark_report
    from . import generation_campaign as campaign
    from . import generation_campaign_completion as completion
    from . import generation_controller as controller
    from . import generation_readiness as readiness
    from . import generation_run as run
    from . import generation_smoke as smoke
    from . import generation_workflow as workflow

_MODULES = {
    "benchmark": "generation_benchmark",
    "benchmark_config": "generation_benchmark_config",
    "benchmark_report": "generation_benchmark_report",
    "campaign": "generation_campaign",
    "completion": "generation_campaign_completion",
    "controller": "generation_controller",
    "cases": "cases",
    "contracts": "contracts",
    "publication": "publication",
    "readiness": "generation_readiness",
    "run": "generation_run",
    "runtime": "runtime",
    "smoke": "generation_smoke",
    "validation": "validation",
    "workflow": "generation_workflow",
}
__all__ = [
    "benchmark",
    "benchmark_config",
    "benchmark_report",
    "campaign",
    "cases",
    "completion",
    "contracts",
    "controller",
    "publication",
    "readiness",
    "run",
    "runtime",
    "smoke",
    "validation",
    "workflow",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public package or service on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
