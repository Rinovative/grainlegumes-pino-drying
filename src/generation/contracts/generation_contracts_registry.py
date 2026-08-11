"""
===============================================================================
generation_contracts_registry.py
===============================================================================
Validate and resolve the typed scientific parameter registry.
Responsibilities:
  - Validate exact schemas for all supported parameter kinds
  - Enforce finite domains, units, coupled-set integrity, and OOD separation
  - Resolve the narrow maintained derivation identifiers
Design principles:
  - Every scientific value has one typed owner
  - Unknown keys and executable expressions fail closed
  - Unresolved values are admitted only while inspecting non-executable templates
This module does NOT:
  - Define material-family roles or VP2 block membership
  - Sample designs, generate fields, or evaluate arbitrary expressions
===============================================================================
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from . import generation_contracts_porosity as porosity_service
from . import generation_contracts_profiles as profile_service

PARAMETER_KINDS: Final = (
    "fixed",
    "interval",
    "conditional_interval",
    "integer",
    "categorical",
    "simplex",
    "parameter_set",
    "derived",
)
INTERVAL_TRANSFORMS: Final = ("linear", "log", "logit", "phase")
DERIVATION_IDENTIFIERS: Final = (
    "copy",
    "complement_of_one",
    "product",
    "mean",
    "schedule_time_average",
)

_SCIENTIFIC_METADATA_KEYS: Final = frozenset(
    {
        "nominal",
        "distribution",
        "classification",
        "profile_applicability",
        "atomic_record",
        "provenance",
    }
)
_CLASSIFICATIONS: Final = ("sampled", "fixed", "derived", "coupled_record")
_MINIMUM_OOD_GAP_FRACTION: Final = 0.15
_MINIMUM_OOD_WIDTH_FRACTION: Final = 0.25


_KIND_KEYS: Final = MappingProxyType(
    {
        "fixed": (frozenset({"kind", "unit", "value"}), frozenset()),
        "interval": (
            frozenset({"kind", "unit", "lower", "upper", "transform"}),
            frozenset({"block", "ood_group", "ood"}),
        ),
        "conditional_interval": (
            frozenset(
                {
                    "kind",
                    "unit",
                    "transform",
                    "support_kind",
                    "support_resolver",
                    "conditioning_coordinates",
                    "material_inputs",
                    "parameter_ood",
                }
            ),
            frozenset({"block", "ood_group"}),
        ),
        "integer": (
            frozenset({"kind", "unit", "lower", "upper"}),
            frozenset({"block", "ood_group", "ood"}),
        ),
        "categorical": (
            frozenset({"kind", "unit", "choices"}),
            frozenset({"ood_group", "ood_choices"}),
        ),
        "simplex": (
            frozenset({"kind", "unit", "components"}),
            frozenset(
                {
                    "block",
                    "ood_group",
                    "ood_values",
                    "selection",
                    "alpha",
                    "minimum_each",
                    "maximum_each",
                }
            ),
        ),
        "parameter_set": (
            frozenset({"kind", "components", "units", "sets"}),
            frozenset({"ood_group", "ood_sets"}),
        ),
        "derived": (
            frozenset({"kind", "unit", "derivation", "sources"}),
            frozenset(),
        ),
    }
)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one isolated string-keyed mapping."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> None:
    """Require all mandatory keys and reject every undeclared key."""
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise ValueError(message)


def _unit(value: Any, *, label: str) -> str:
    """Return one explicit safe unit declaration."""
    if not isinstance(value, str) or not value or value.strip() != value or any(character in value for character in ("\x00", "\n", "\r")):
        message = f"{label} must be explicit non-empty single-line text."
        raise ValueError(message)
    return value


def _finite(value: Any, *, label: str, allow_unresolved: bool) -> float | None:
    """Return one finite real or an allowed unresolved template marker."""
    if value is None and allow_unresolved:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must be one finite real value."
        raise ValueError(message)
    return float(value)


def _required_finite(value: Any, *, label: str) -> float:
    """Return one finite value from a configuration position that cannot be null."""
    result = _finite(value, label=label, allow_unresolved=False)
    if result is None:
        message = f"{label} must be resolved."
        raise ValueError(message)
    return result


def _name_sequence(value: Any, *, label: str, minimum: int = 1) -> tuple[str, ...]:
    """Return one unique ordered sequence of non-empty names."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        message = f"{label} must be an ordered name sequence."
        raise TypeError(message)
    names = tuple(value)
    if len(names) < minimum or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        message = f"{label} must contain at least {minimum} unique non-empty names."
        raise ValueError(message)
    return names


def _validate_block_metadata(entry: dict[str, Any], *, label: str) -> None:
    """Validate optional sampling-block and OOD-group labels."""
    for key in ("block", "ood_group"):
        if key in entry and (not isinstance(entry[key], str) or not entry[key]):
            message = f"{label}.{key} must be non-empty text."
            raise ValueError(message)


def _transform_coordinate(value: float, transform: str) -> float:
    """Map a physical value to the registry transform coordinate."""
    if transform in {"linear", "phase"}:
        return value
    if transform == "log":
        if value <= 0:
            message = "Log-transformed values must be strictly positive."
            raise ValueError(message)
        return math.log(value)
    if not 0 < value < 1:
        message = "Logit-transformed values must lie strictly inside (0, 1)."
        raise ValueError(message)
    return math.log(value / (1.0 - value))


def _validate_interval_domain(lower: float, upper: float, transform: str, *, label: str) -> None:
    """Validate one ordered transform-compatible interval."""
    if lower > upper:
        message = f"{label}.lower must not exceed {label}.upper."
        raise ValueError(message)
    if transform == "log" and lower <= 0:
        message = f"{label} log interval must be strictly positive."
        raise ValueError(message)
    if transform == "logit" and not 0 < lower <= upper < 1:
        message = f"{label} logit interval must lie inside (0, 1)."
        raise ValueError(message)
    if transform == "phase" and not 0.0 <= lower <= upper <= 2.0 * math.pi:
        message = f"{label} phase interval must lie inside [0, 2*pi]."
        raise ValueError(message)


def _validate_interval_ood(entry: dict[str, Any], *, label: str, allow_unresolved: bool) -> None:
    """Require every OOD tail to have the binding transformed separation."""
    if "ood" not in entry:
        return
    raw_intervals = entry["ood"]
    if raw_intervals is None and allow_unresolved:
        entry["ood"] = []
        return
    if not isinstance(raw_intervals, list) or not raw_intervals:
        message = f"{label}.ood must be a non-empty list of disjoint intervals."
        raise TypeError(message)
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_intervals):
        ood = _mapping(raw, label=f"{label}.ood[{index}]")
        _exact_keys(
            ood,
            required=frozenset({"lower", "upper"}),
            optional=frozenset({"hard_boundary"}),
            label=f"{label}.ood[{index}]",
        )
        hard_boundary = ood.get("hard_boundary", False)
        if not isinstance(hard_boundary, bool):
            message = f"{label}.ood[{index}].hard_boundary must be boolean."
            raise TypeError(message)
        normalized.append(
            {
                "lower": _finite(
                    ood["lower"],
                    label=f"{label}.ood[{index}].lower",
                    allow_unresolved=allow_unresolved,
                ),
                "upper": _finite(
                    ood["upper"],
                    label=f"{label}.ood[{index}].upper",
                    allow_unresolved=allow_unresolved,
                ),
                "hard_boundary": hard_boundary,
            }
        )
    entry["ood"] = normalized
    id_lower_value = entry["lower"]
    id_upper_value = entry["upper"]
    if id_lower_value is None or id_upper_value is None:
        return
    transform = str(entry.get("transform", "linear"))
    id_lower = float(id_lower_value)
    id_upper = float(id_upper_value)
    id_lower_t = _transform_coordinate(id_lower, transform)
    id_upper_t = _transform_coordinate(id_upper, transform)
    id_width = id_upper_t - id_lower_t
    if id_width <= 0:
        message = f"{label} must have positive transformed width when OOD tails are configured."
        raise ValueError(message)
    previous_upper: float | None = None
    for index, interval in enumerate(normalized):
        lower = interval["lower"]
        upper = interval["upper"]
        if lower is None or upper is None:
            continue
        _validate_interval_domain(lower, upper, transform, label=f"{label}.ood[{index}]")
        if previous_upper is not None and lower <= previous_upper:
            message = f"{label}.ood intervals must be strictly ordered and mutually disjoint."
            raise ValueError(message)
        previous_upper = upper
        if not (upper < id_lower or lower > id_upper):
            message = f"{label}.ood[{index}] must be disjoint from ID with a nonzero gap."
            raise ValueError(message)
        lower_t = _transform_coordinate(lower, transform)
        upper_t = _transform_coordinate(upper, transform)
        gap = id_lower_t - upper_t if upper < id_lower else lower_t - id_upper_t
        width = upper_t - lower_t
        if gap < _MINIMUM_OOD_GAP_FRACTION * id_width:
            message = f"{label}.ood[{index}] transformed gap must be at least {_MINIMUM_OOD_GAP_FRACTION:g} of ID width."
            raise ValueError(message)
        if width < _MINIMUM_OOD_WIDTH_FRACTION * id_width and not interval["hard_boundary"]:
            message = f"{label}.ood[{index}] transformed width must be at least {_MINIMUM_OOD_WIDTH_FRACTION:g} of ID width."
            raise ValueError(message)


def _validate_nominal(value: Any, *, label: str) -> Any:
    """Validate one finite scalar or complete finite nominal structure."""
    if isinstance(value, Mapping):
        if not value or not all(isinstance(key, str) and key for key in value):
            message = f"{label} nominal mappings must have non-empty string keys."
            raise ValueError(message)
        return {key: _validate_nominal(item, label=f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            message = f"{label} nominal sequences must not be empty."
            raise ValueError(message)
        return [_validate_nominal(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must contain only finite numeric values."
        raise ValueError(message)
    return float(value)


def _validate_scientific_metadata(entry: dict[str, Any], *, label: str) -> None:
    """Validate optional decision metadata retained beside runtime semantics."""
    if "nominal" in entry:
        entry["nominal"] = _validate_nominal(entry["nominal"], label=f"{label}.nominal")
    if "distribution" in entry and (not isinstance(entry["distribution"], str) or not entry["distribution"]):
        message = f"{label}.distribution must be non-empty text."
        raise ValueError(message)
    if entry.get("classification") not in _CLASSIFICATIONS:
        message = f"{label}.classification must be one of {list(_CLASSIFICATIONS)}."
        raise ValueError(message)
    profiles = _name_sequence(entry.get("profile_applicability"), label=f"{label}.profile_applicability")
    if any(profile not in profile_service.available_profiles() for profile in profiles):
        message = f"{label}.profile_applicability contains an unknown profile."
        raise ValueError(message)
    entry["profile_applicability"] = list(profiles)
    if "atomic_record" in entry and (not isinstance(entry["atomic_record"], str) or not entry["atomic_record"]):
        message = f"{label}.atomic_record must be non-empty text."
        raise ValueError(message)
    if "provenance" in entry:
        entry["provenance"] = _mapping(entry["provenance"], label=f"{label}.provenance")


def _validate_weight_vectors(
    value: Any,
    *,
    components: tuple[str, ...],
    label: str,
    allow_unresolved: bool,
) -> list[dict[str, float]]:
    """Validate explicit complete simplex vectors."""
    if value is None and allow_unresolved:
        return []
    if not isinstance(value, list):
        message = f"{label} must be a list of complete simplex vectors."
        raise TypeError(message)
    result: list[dict[str, float]] = []
    identities: set[tuple[float, ...]] = set()
    for index, raw in enumerate(value):
        vector = _mapping(raw, label=f"{label}[{index}]")
        if set(vector) != set(components):
            message = f"{label}[{index}] must define exactly {list(components)}."
            raise ValueError(message)
        normalized = {name: _finite(vector[name], label=f"{label}[{index}].{name}", allow_unresolved=False) for name in components}
        weights = {name: float(number) for name, number in normalized.items() if number is not None}
        if any(number < 0 for number in weights.values()) or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            message = f"{label}[{index}] must be non-negative and sum to one."
            raise ValueError(message)
        identity = tuple(weights[name] for name in components)
        if identity in identities:
            message = f"{label}[{index}] duplicates an earlier simplex vector."
            raise ValueError(message)
        identities.add(identity)
        result.append(weights)
    return result


def _validate_coupled_sets(
    value: Any,
    *,
    components: tuple[str, ...],
    label: str,
    allow_unresolved: bool,
) -> list[dict[str, Any]]:
    """Validate complete uniquely identified coupled parameter sets."""
    if value is None and allow_unresolved:
        return []
    if not isinstance(value, list):
        message = f"{label} must be a list of complete identified sets."
        raise TypeError(message)
    result: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, label=f"{label}[{index}]")
        _exact_keys(item, required=frozenset({"id", "values"}), optional=frozenset(), label=f"{label}[{index}]")
        identity = item["id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            message = f"{label}[{index}].id must be unique non-empty text."
            raise ValueError(message)
        identities.add(identity)
        values = _mapping(item["values"], label=f"{label}[{index}].values")
        if set(values) != set(components):
            message = f"{label}[{index}].values must define exactly {list(components)}."
            raise ValueError(message)
        normalized = {name: _finite(values[name], label=f"{label}[{index}].values.{name}", allow_unresolved=False) for name in components}
        result.append({"id": identity, "values": {name: float(number) for name, number in normalized.items() if number is not None}})
    return result


def _validate_entry(name: str, value: Any, *, allow_unresolved: bool) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Validate one exact typed registry entry."""
    label = f"parameter_registry.{name}"
    entry = _mapping(value, label=label)
    kind = entry.get("kind")
    if kind not in PARAMETER_KINDS:
        message = f"{label}.kind must be one of {list(PARAMETER_KINDS)}, got {kind!r}."
        raise ValueError(message)
    required, optional = _KIND_KEYS[kind]
    optional = optional | _SCIENTIFIC_METADATA_KEYS
    _exact_keys(entry, required=required, optional=optional, label=label)
    _validate_block_metadata(entry, label=label)
    _validate_scientific_metadata(entry, label=label)

    if kind != "parameter_set":
        entry["unit"] = _unit(entry["unit"], label=f"{label}.unit")
    if kind == "fixed":
        entry["value"] = _finite(entry["value"], label=f"{label}.value", allow_unresolved=allow_unresolved)
    elif kind == "interval":
        transform = entry["transform"]
        if transform not in INTERVAL_TRANSFORMS:
            message = f"{label}.transform must be one of {list(INTERVAL_TRANSFORMS)}."
            raise ValueError(message)
        entry["lower"] = _finite(entry["lower"], label=f"{label}.lower", allow_unresolved=allow_unresolved)
        entry["upper"] = _finite(entry["upper"], label=f"{label}.upper", allow_unresolved=allow_unresolved)
        if entry["lower"] is not None and entry["upper"] is not None:
            _validate_interval_domain(float(entry["lower"]), float(entry["upper"]), transform, label=label)
        _validate_interval_ood(entry, label=label, allow_unresolved=allow_unresolved)
    elif kind == "conditional_interval":
        if entry["transform"] != "log":
            message = f"{label}.transform must be 'log' for the maintained conditional interval."
            raise ValueError(message)
        if entry["support_kind"] != "conditional":
            message = f"{label}.support_kind must be 'conditional'."
            raise ValueError(message)
        if entry["support_resolver"] != "kozeny_carman_anchor_factor":
            message = f"{label}.support_resolver is not a maintained conditional-support resolver."
            raise ValueError(message)
        conditioning = _name_sequence(
            entry["conditioning_coordinates"],
            label=f"{label}.conditioning_coordinates",
        )
        material_inputs = _name_sequence(
            entry["material_inputs"],
            label=f"{label}.material_inputs",
        )
        expected_conditioning = ("kappa_mean",)
        expected_inputs = (
            "permeability.nominal",
            "packing_porosity_mean_support",
            "density_calibration.reference.eps_bed_cal_ref",
        )
        if conditioning != expected_conditioning or material_inputs != expected_inputs:
            message = f"{label} must declare conditioning {list(expected_conditioning)} and material inputs {list(expected_inputs)}."
            raise ValueError(message)
        if entry["parameter_ood"] != "conditional_relative_tails":
            message = f"{label}.parameter_ood must be 'conditional_relative_tails'."
            raise ValueError(message)
        if entry.get("nominal") != 1.0:
            message = f"{label}.nominal must be exactly 1.0."
            raise ValueError(message)
        entry["conditioning_coordinates"] = list(conditioning)
        entry["material_inputs"] = list(material_inputs)
    elif kind == "integer":
        for bound in ("lower", "upper"):
            number = _finite(entry[bound], label=f"{label}.{bound}", allow_unresolved=allow_unresolved)
            if number is not None and not number.is_integer():
                message = f"{label}.{bound} must be an integer value."
                raise ValueError(message)
            entry[bound] = None if number is None else int(number)
        if entry["lower"] is not None and entry["upper"] is not None and entry["lower"] > entry["upper"]:
            message = f"{label}.lower must not exceed {label}.upper."
            raise ValueError(message)
        _validate_interval_ood(entry, label=label, allow_unresolved=allow_unresolved)
        if "ood" in entry:
            for index, interval in enumerate(entry["ood"]):
                for bound in ("lower", "upper"):
                    number = interval[bound]
                    if number is not None and not float(number).is_integer():
                        message = f"{label}.ood[{index}].{bound} must be an integer value."
                        raise ValueError(message)
                    interval[bound] = None if number is None else int(number)
    elif kind == "categorical":
        choices = entry["choices"]
        if choices is None and allow_unresolved:
            choices = []
        if not isinstance(choices, list) or (not choices and not allow_unresolved):
            message = f"{label}.choices must be a non-empty list in executable configurations."
            raise ValueError(message)
        if len({_choice_identity(choice) for choice in choices}) != len(choices):
            message = f"{label}.choices contains duplicates."
            raise ValueError(message)
        entry["choices"] = copy.deepcopy(choices)
        if "ood_choices" in entry:
            ood_choices = entry["ood_choices"]
            if ood_choices is None and allow_unresolved:
                ood_choices = []
            if not isinstance(ood_choices, list) or any(
                _choice_identity(choice) in {_choice_identity(item) for item in choices} for choice in ood_choices
            ):
                message = f"{label}.ood_choices must be a disjoint list."
                raise ValueError(message)
            entry["ood_choices"] = copy.deepcopy(ood_choices)
    elif kind == "simplex":
        components = _name_sequence(entry["components"], label=f"{label}.components", minimum=2)
        entry["components"] = list(components)
        configured_shape = {"selection", "alpha", "minimum_each", "maximum_each"}.intersection(entry)
        if configured_shape:
            if configured_shape != {"selection", "alpha", "minimum_each", "maximum_each"}:
                message = f"{label} truncated-simplex configuration is incomplete."
                raise ValueError(message)
            if entry["selection"] != "truncated_dirichlet":
                message = f"{label}.selection must be 'truncated_dirichlet'."
                raise ValueError(message)
            alpha = entry["alpha"]
            if not isinstance(alpha, list) or len(alpha) != len(components):
                message = f"{label}.alpha must match the simplex components."
                raise ValueError(message)
            entry["alpha"] = [_required_finite(item, label=f"{label}.alpha[{index}]") for index, item in enumerate(alpha)]
            if any(item <= 0 for item in entry["alpha"]):
                message = f"{label}.alpha values must be positive."
                raise ValueError(message)
            minimum = _required_finite(
                entry["minimum_each"],
                label=f"{label}.minimum_each",
            )
            maximum = _required_finite(
                entry["maximum_each"],
                label=f"{label}.maximum_each",
            )
            entry["minimum_each"] = minimum
            entry["maximum_each"] = maximum
            if not 0 <= entry["minimum_each"] < entry["maximum_each"] <= 1:
                message = f"{label} simplex component bounds are invalid."
                raise ValueError(message)
            if len(components) * entry["minimum_each"] > 1 or len(components) * entry["maximum_each"] < 1:
                message = f"{label} simplex component bounds have no feasible complete vector."
                raise ValueError(message)
        if "ood_values" in entry:
            entry["ood_values"] = _validate_weight_vectors(
                entry["ood_values"],
                components=components,
                label=f"{label}.ood_values",
                allow_unresolved=allow_unresolved,
            )
    elif kind == "parameter_set":
        components = _name_sequence(entry["components"], label=f"{label}.components")
        units = _mapping(entry["units"], label=f"{label}.units")
        if set(units) != set(components):
            message = f"{label}.units must define exactly {list(components)}."
            raise ValueError(message)
        entry["components"] = list(components)
        entry["units"] = {name: _unit(units[name], label=f"{label}.units.{name}") for name in components}
        primary = "sets"
        ood_key = "ood_sets"
        entry[primary] = _validate_coupled_sets(
            entry[primary],
            components=components,
            label=f"{label}.{primary}",
            allow_unresolved=allow_unresolved,
        )
        if not entry[primary] and not allow_unresolved:
            message = f"{label}.{primary} must contain at least one complete set."
            raise ValueError(message)
        if ood_key in entry:
            entry[ood_key] = _validate_coupled_sets(
                entry[ood_key],
                components=components,
                label=f"{label}.{ood_key}",
                allow_unresolved=allow_unresolved,
            )
            primary_ids = {item["id"] for item in entry[primary]}
            if primary_ids.intersection(item["id"] for item in entry[ood_key]):
                message = f"{label}.{ood_key} identities must be disjoint from ID set identities."
                raise ValueError(message)
            primary_values = {tuple(item["values"][name] for name in components) for item in entry[primary]}
            if any(tuple(item["values"][name] for name in components) in primary_values for item in entry[ood_key]):
                message = f"{label}.{ood_key} values must be disjoint from ID set values."
                raise ValueError(message)
    else:
        derivation = entry["derivation"]
        if derivation not in DERIVATION_IDENTIFIERS:
            message = f"{label}.derivation must be one of {list(DERIVATION_IDENTIFIERS)}."
            raise ValueError(message)
        sources = _name_sequence(entry["sources"], label=f"{label}.sources")
        expected_sources = 2 if derivation in {"product", "mean"} else 1
        if len(sources) != expected_sources:
            message = f"{label}.sources must contain exactly {expected_sources} value(s) for {derivation!r}."
            raise ValueError(message)
        entry["sources"] = list(sources)
    return entry


def _choice_identity(value: Any) -> str:
    """Return a stable identity for one finite scalar or string choice."""
    if isinstance(value, str) and value:
        return f"str:{value}"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"Categorical choices must be non-empty strings or finite numbers, got {value!r}."
        raise ValueError(message)
    return f"number:{float(value):.17g}"


def ood_separation_fractions() -> tuple[float, float]:
    """Return minimum transformed gap and width fractions for OOD tails."""
    return _MINIMUM_OOD_GAP_FRACTION, _MINIMUM_OOD_WIDTH_FRACTION


def validate_parameter_registry(value: Any, *, allow_unresolved: bool = False) -> dict[str, dict[str, Any]]:
    """
    Validate one complete typed parameter registry.

    Parameters
    ----------
    value : Any
        String-keyed registry mapping.
    allow_unresolved : bool, optional
        Admit ``null`` values and empty coupled inventories only for a declared
        non-executable scientific template.

    """
    registry = _mapping(value, label="parameter_registry")
    if not registry:
        message = "parameter_registry must not be empty."
        raise ValueError(message)
    return {name: _validate_entry(name, entry, allow_unresolved=allow_unresolved) for name, entry in registry.items()}


def effective_dimension(entry: Mapping[str, Any]) -> int:
    """Return the numerical design dimension owned by one registry entry."""
    kind = entry["kind"]
    if kind in {"interval", "conditional_interval", "integer"}:
        return 1
    if kind == "simplex":
        return len(entry["components"]) - 1
    return 0


def transformed_coordinate(value: float, entry: Mapping[str, Any]) -> float:
    """Return one registry-owned transformed coordinate for separation checks."""
    transform = str(entry.get("transform", "linear"))
    return _transform_coordinate(float(value), transform)


def resolve_conditional_support(
    entry: Mapping[str, Any],
    *,
    values: Mapping[str, Any],
    material_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one declared conditional scalar support from complete case context."""
    if entry.get("kind") != "conditional_interval" or entry.get("support_kind") != "conditional":
        message = "Conditional support resolution requires one validated conditional_interval entry."
        raise ValueError(message)
    resolver = entry.get("support_resolver")
    if resolver != "kozeny_carman_anchor_factor":
        message = f"Unsupported conditional-support resolver {resolver!r}."
        raise ValueError(message)
    registry = material_contract.get("parameter_registry")
    if not isinstance(registry, Mapping) or "kappa_mean" not in registry:
        message = "Conditional support requires the resolved material parameter registry."
        raise ValueError(message)
    kappa_entry = registry["kappa_mean"]
    if not isinstance(kappa_entry, Mapping) or "nominal" not in kappa_entry:
        message = "Conditional support requires the material permeability nominal."
        raise ValueError(message)
    return porosity_service.resolve_anchor_factor_support(
        sampled_kappa_mean=float(values["kappa_mean"]),
        material_kappa_nominal=float(kappa_entry["nominal"]),
        eps_bed_cal_ref=float(values["eps_bed_cal_ref"]),
        packing_porosity_mean_support=material_contract["packing_porosity_mean_support"],
        eps_min_global=float(values["eps_min_global"]),
        eps_max_global=float(values["eps_max_global"]),
        ood_gap_fraction=_MINIMUM_OOD_GAP_FRACTION,
        ood_width_fraction=_MINIMUM_OOD_WIDTH_FRACTION,
    )


def resolve_derived_values(
    registry: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, Any],
    *,
    defer_missing: bool = False,
) -> dict[str, Any]:
    """Resolve maintained scalar derivations without evaluating expressions."""
    resolved = copy.deepcopy(dict(values))
    pending = {name for name, entry in registry.items() if entry["kind"] == "derived" and name not in resolved}
    while pending:
        progress = False
        for name in tuple(sorted(pending)):
            entry = registry[name]
            sources = tuple(entry["sources"])
            if any(source not in resolved for source in sources):
                continue
            numbers = [float(resolved[source]) for source in sources]
            if not all(math.isfinite(number) for number in numbers):
                message = f"Derived parameter {name!r} received a non-finite source."
                raise ValueError(message)
            derivation = entry["derivation"]
            if derivation == "copy":
                result = numbers[0]
            elif derivation == "complement_of_one":
                result = 1.0 - numbers[0]
            elif derivation == "product":
                result = numbers[0] * numbers[1]
            elif derivation == "mean":
                result = 0.5 * (numbers[0] + numbers[1])
            else:
                continue
            if not math.isfinite(result):
                message = f"Derived parameter {name!r} is non-finite."
                raise ValueError(message)
            resolved[name] = result
            pending.remove(name)
            progress = True
        if not progress:
            if defer_missing:
                break
            details = {name: list(registry[name]["sources"]) for name in sorted(pending)}
            message = f"Derived parameters have unresolved or cyclic sources: {details}."
            raise ValueError(message)
    return resolved
