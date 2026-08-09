"""
Reference-simulation generation, COMSOL execution, and publication services.

Provides:
- campaign_runtime: exact-commit campaign submission and status evidence
- case: deterministic scientific input bundles and simulation identities
- cluster: bounded node-worker and scheduler command services
- config: campaign, batch, scientific, and execution configuration resolution
- fields: independent bed, pressure, and initial-moisture fields
- inventory: mechanical parameter ownership and consumer auditing
- materials: role-neutral material and authoritative sampling-block contracts
- preflight: non-solving native CPU environment and path validation
- profiles: immutable simulation-profile and logical export contracts
- registry: strict typed scientific parameter schemas
- runtime: COMSOL execution and atomic case publication
- sampling: deterministic independent blockwise designs
- schedule: compositional mixed inlet schedules
- source: exact source-repository commit provenance
- storage: canonical HDF5 conversion and validation
- workflow: transfer evidence, dataset gates, storage status, and cleanup
- workspace: persistent-root and disposable-workspace safety
"""

from . import generation_campaign_runtime as campaign_runtime
from . import generation_case as case
from . import generation_cluster as cluster
from . import generation_config as config
from . import generation_fields as fields
from . import generation_inventory as inventory
from . import generation_materials as materials
from . import generation_preflight as preflight
from . import generation_profiles as profiles
from . import generation_registry as registry
from . import generation_runtime as runtime
from . import generation_sampling as sampling
from . import generation_schedule as schedule
from . import generation_source as source
from . import generation_storage as storage
from . import generation_workflow as workflow
from . import generation_workspace as workspace

__all__ = [
    "campaign_runtime",
    "case",
    "cluster",
    "config",
    "fields",
    "inventory",
    "materials",
    "preflight",
    "profiles",
    "registry",
    "runtime",
    "sampling",
    "schedule",
    "source",
    "storage",
    "workflow",
    "workspace",
]
