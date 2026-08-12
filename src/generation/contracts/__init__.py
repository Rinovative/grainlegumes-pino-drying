"""
Immutable Generation vocabularies, scientific registries, and source contracts.

Provides:
- comsol_spreadsheet: canonical COMSOL Spreadsheet export parsing
- descriptors: stable cross-package profile and vocabulary descriptors
- mapping: semantic Generation export-mapping identities
- materials: resolved material-science contracts
- paths: persistent storage-root safety contracts
- porosity: calibrated porosity support and response contracts
- profiles: canonical simulation-profile schemas
- provenance: scientific-source provenance admission
- registry: typed scientific parameter registry
- scalar_handoff: exact transient scalar-source admission and immutable entries
- source: source-repository provenance validation
- vocabulary: campaign, evaluation, and membership vocabulary
- EvaluationRegime: canonical package evaluation-regime type
- FieldContract: immutable named physical-field descriptor
- IdMembership: canonical learned-membership type
- ProfileContract: immutable profile-level field contract
- available_material_families: supported material-family vocabulary
- available_profile_ids: supported simulation-profile vocabulary
- evaluation_regimes: canonical package evaluation regimes
- get_profile_contract: immutable profile-contract resolution
- id_memberships: canonical learned-membership vocabulary
- material_roles: canonical campaign material roles
- validate_git_commit: source-commit identity validation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_contracts_comsol_spreadsheet as comsol_spreadsheet
    from . import generation_contracts_descriptors as descriptors
    from . import generation_contracts_mapping as mapping
    from . import generation_contracts_materials as materials
    from . import generation_contracts_paths as paths
    from . import generation_contracts_porosity as porosity
    from . import generation_contracts_profiles as profiles
    from . import generation_contracts_provenance as provenance
    from . import generation_contracts_registry as registry
    from . import generation_contracts_scalar_handoff as scalar_handoff
    from . import generation_contracts_source as source
    from . import generation_contracts_vocabulary as vocabulary
    from .generation_contracts_descriptors import (
        EvaluationRegime,
        FieldContract,
        IdMembership,
        ProfileContract,
        available_material_families,
        available_profile_ids,
        evaluation_regimes,
        get_profile_contract,
        id_memberships,
        material_roles,
        validate_git_commit,
    )

_MODULES = {
    "comsol_spreadsheet": "generation_contracts_comsol_spreadsheet",
    "descriptors": "generation_contracts_descriptors",
    "mapping": "generation_contracts_mapping",
    "materials": "generation_contracts_materials",
    "paths": "generation_contracts_paths",
    "porosity": "generation_contracts_porosity",
    "profiles": "generation_contracts_profiles",
    "provenance": "generation_contracts_provenance",
    "registry": "generation_contracts_registry",
    "scalar_handoff": "generation_contracts_scalar_handoff",
    "source": "generation_contracts_source",
    "vocabulary": "generation_contracts_vocabulary",
}
_DESCRIPTOR_EXPORTS = frozenset(
    {
        "EvaluationRegime",
        "FieldContract",
        "IdMembership",
        "ProfileContract",
        "available_material_families",
        "available_profile_ids",
        "evaluation_regimes",
        "get_profile_contract",
        "id_memberships",
        "material_roles",
        "validate_git_commit",
    }
)
__all__ = [
    "EvaluationRegime",
    "FieldContract",
    "IdMembership",
    "ProfileContract",
    "available_material_families",
    "available_profile_ids",
    "comsol_spreadsheet",
    "descriptors",
    "evaluation_regimes",
    "get_profile_contract",
    "id_memberships",
    "mapping",
    "material_roles",
    "materials",
    "paths",
    "porosity",
    "profiles",
    "provenance",
    "registry",
    "scalar_handoff",
    "source",
    "validate_git_commit",
    "vocabulary",
]


def __getattr__(name: str) -> object:
    """Resolve one declared contract module or stable descriptor symbol."""
    module_name = _MODULES.get(name)
    if module_name is not None:
        value = import_module(f"{__name__}.{module_name}")
    elif name in _DESCRIPTOR_EXPORTS:
        value = getattr(import_module(f"{__name__}.generation_contracts_descriptors"), name)
    else:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    globals()[name] = value
    return value
