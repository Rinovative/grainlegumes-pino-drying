"""
Reference-simulation generation, COMSOL execution, and publication services.

Provides:
- case: deterministic case-input and isolated-work-directory services
- cluster: bounded node-worker and scheduler command services
- config: generation-configuration loading and validation
- fields: Python spatial-field generation
- profiles: authoritative steady-flow and transient-drying profile contracts
- runtime: COMSOL execution, export collection, and publication
- sampling: deterministic design-of-experiments sampling
"""

from . import generation_case as case
from . import generation_cluster as cluster
from . import generation_config as config
from . import generation_fields as fields
from . import generation_profiles as profiles
from . import generation_runtime as runtime
from . import generation_sampling as sampling

__all__ = ["case", "cluster", "config", "fields", "profiles", "runtime", "sampling"]
