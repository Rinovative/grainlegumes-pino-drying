"""
Immutable Generation vocabularies, scientific registries, and source contracts.

Provides:
- campaign_identity: canonical campaign role/material/count/seed naming
- comsol_spreadsheet: canonical COMSOL Spreadsheet export parsing
- descriptors: stable cross-package profile and vocabulary descriptors
- materials: resolved material-science contracts
- material_registry: stable material identifier, slug, and order mapping
- paths: persistent storage-root safety contracts
- porosity: calibrated porosity support and response contracts
- profiles: canonical simulation-profile schemas
- provenance: scientific-source provenance admission
- registry: typed scientific parameter registry
- scalar_handoff: exact transient scalar-source admission and immutable entries
- source: source-repository provenance validation
- templates: configured COMSOL template byte identity
- vocabulary: campaign, evaluation, and membership vocabulary
- EvaluationRegime: canonical package evaluation-regime type
- FieldContract: immutable named physical-field descriptor
- IdMembership: canonical learned-membership type
- ProfileContract: immutable profile-level field contract
- available_profile_ids: supported simulation-profile vocabulary
- evaluation_regimes: canonical package evaluation regimes
- get_profile_contract: immutable profile-contract resolution
- id_memberships: canonical learned-membership vocabulary
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_contracts_campaign_identity as campaign_identity
    from . import generation_contracts_comsol_spreadsheet as comsol_spreadsheet
    from . import generation_contracts_descriptors as descriptors
    from . import generation_contracts_material_registry as material_registry
    from . import generation_contracts_materials as materials
    from . import generation_contracts_paths as paths
    from . import generation_contracts_porosity as porosity
    from . import generation_contracts_profiles as profiles
    from . import generation_contracts_provenance as provenance
    from . import generation_contracts_registry as registry
    from . import generation_contracts_scalar_handoff as scalar_handoff
    from . import generation_contracts_source as source
    from . import generation_contracts_templates as templates
    from . import generation_contracts_vocabulary as vocabulary
    from .generation_contracts_descriptors import (
        EvaluationRegime,
        FieldContract,
        IdMembership,
        ProfileContract,
        available_profile_ids,
        evaluation_regimes,
        get_profile_contract,
        id_memberships,
    )

_MODULES = {
    "campaign_identity": "generation_contracts_campaign_identity",
    "comsol_spreadsheet": "generation_contracts_comsol_spreadsheet",
    "descriptors": "generation_contracts_descriptors",
    "materials": "generation_contracts_materials",
    "material_registry": "generation_contracts_material_registry",
    "paths": "generation_contracts_paths",
    "porosity": "generation_contracts_porosity",
    "profiles": "generation_contracts_profiles",
    "provenance": "generation_contracts_provenance",
    "registry": "generation_contracts_registry",
    "scalar_handoff": "generation_contracts_scalar_handoff",
    "source": "generation_contracts_source",
    "templates": "generation_contracts_templates",
    "vocabulary": "generation_contracts_vocabulary",
}
_DESCRIPTOR_EXPORTS = frozenset(
    {
        "EvaluationRegime",
        "FieldContract",
        "IdMembership",
        "ProfileContract",
        "available_profile_ids",
        "evaluation_regimes",
        "get_profile_contract",
        "id_memberships",
    }
)
__all__ = [
    "EvaluationRegime",
    "FieldContract",
    "IdMembership",
    "ProfileContract",
    "available_profile_ids",
    "campaign_identity",
    "comsol_spreadsheet",
    "descriptors",
    "evaluation_regimes",
    "get_profile_contract",
    "id_memberships",
    "material_registry",
    "materials",
    "paths",
    "porosity",
    "profiles",
    "provenance",
    "registry",
    "scalar_handoff",
    "source",
    "templates",
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
