"""
===============================================================================
generation_contracts_descriptors.py
===============================================================================
Expose stable read-only generation schema descriptors across package boundaries.
Responsibilities:
  - Describe canonical profile fields and units without exposing module layout
  - Expose supported profile, material, role, regime, and OOD-group vocabulary
  - Validate source-commit provenance through one public contract boundary
Design principles:
  - Descriptors are derived from the authoritative Generation schema owners
  - Dataset consumers depend on immutable contracts rather than implementation files
  - Scientific values and active campaign selections remain resolved configuration
This module does NOT:
  - Load campaigns, generate cases, validate HDF5 payloads, or execute workflows
  - Duplicate configured material roles, counts, seeds, or package requests
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from . import generation_contracts_materials as materials
from . import generation_contracts_profiles as profiles
from . import generation_contracts_source as source
from . import generation_contracts_vocabulary as vocabulary

IdMembership: TypeAlias = vocabulary.IdMembership
EvaluationRegime: TypeAlias = vocabulary.EvaluationRegime


@dataclass(frozen=True, slots=True)
class FieldContract:
    """Describe one canonical Generation field and its physical unit."""

    name: str
    unit: str

    def as_dict(self) -> dict[str, str]:
        """Return the canonical serializable field descriptor."""
        return {"name": self.name, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class ProfileContract:
    """Describe one supported Generation profile at the storage boundary."""

    id: str
    available_learning_views: tuple[str, ...]
    airflow_source: str
    coordinate_fields: tuple[FieldContract, ...]
    stationary_fixed_fields: tuple[FieldContract, ...]
    static_fields: tuple[FieldContract, ...]
    transient_fields: tuple[FieldContract, ...]
    schedule_fields: tuple[FieldContract, ...]
    scalar_inputs: tuple[FieldContract, ...]

    def field(self, name: str) -> FieldContract:
        """Return one unambiguous profile field by logical name."""
        matches = tuple(
            field
            for group in (
                self.coordinate_fields,
                self.stationary_fixed_fields,
                self.static_fields,
                self.transient_fields,
                self.schedule_fields,
                self.scalar_inputs,
            )
            for field in group
            if field.name == name
        )
        if not matches:
            message = f"Generation profile {self.id!r} has no field {name!r}."
            raise ValueError(message)
        units = {field.unit for field in matches}
        if len(units) != 1:
            message = f"Generation profile field {name!r} has ambiguous units {sorted(units)}."
            raise ValueError(message)
        return matches[0]


def _fields(names: tuple[str, ...], units: tuple[str, ...]) -> tuple[FieldContract, ...]:
    """Return immutable aligned field descriptors."""
    return tuple(FieldContract(name, unit) for name, unit in zip(names, units, strict=True))


def get_profile_contract(profile_id: str) -> ProfileContract:
    """Return one immutable profile schema derived from Generation owners."""
    profile = profiles.resolve_profile(profile_id)
    is_transient = profile.id == profiles.TRANSIENT_DRYING_PROFILE
    return ProfileContract(
        id=profile.id,
        available_learning_views=profile.available_learning_views,
        airflow_source=profile.airflow_source,
        coordinate_fields=(FieldContract("x", "m"), FieldContract("y", "m")),
        stationary_fixed_fields=_fields(
            profiles.STATIONARY_FIXED_FIELDS,
            profiles.STATIONARY_FIXED_UNITS,
        ),
        static_fields=_fields(
            profiles.static_field_names(profile.id),
            profiles.static_field_units(profile.id),
        ),
        transient_fields=(_fields(profiles.TRANSIENT_FIELD_NAMES, profiles.TRANSIENT_FIELD_UNITS) if is_transient else ()),
        schedule_fields=(_fields(profiles.SCHEDULE_FIELDS, profiles.SCHEDULE_UNITS) if is_transient else ()),
        scalar_inputs=_fields(
            profiles.scalar_input_fields(profile.id),
            profiles.scalar_input_units(profile.id),
        ),
    )


def available_profile_ids() -> tuple[str, ...]:
    """Return the supported simulation-profile identifiers."""
    return profiles.available_profiles()


def available_material_families() -> tuple[str, ...]:
    """Return the role-neutral configured-material vocabulary."""
    return materials.available_material_families()


def evaluation_regimes() -> tuple[str, ...]:
    """Return the canonical package evaluation-regime vocabulary."""
    return vocabulary.EVALUATION_REGIMES


def material_roles() -> tuple[str, ...]:
    """Return the canonical campaign material-role vocabulary."""
    return vocabulary.MATERIAL_ROLES


def id_memberships() -> tuple[str, ...]:
    """Return the canonical learned in-distribution membership vocabulary."""
    return vocabulary.ID_MEMBERSHIPS


def validate_git_commit(value: object) -> str:
    """Return one exact full lowercase source-commit identifier."""
    return source.validate_git_commit(value)
