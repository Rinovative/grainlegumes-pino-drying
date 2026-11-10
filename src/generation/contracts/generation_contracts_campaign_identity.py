"""Derive and validate source campaign names from their original designs.

Responsibilities:
  - Encode ordered material design sizes per design kind, domain profile, root seed and a deviating transient horizon
  - Derive the domain profile from the design kinds and own the short filesystem token of every supported profile
  - Own the revision grammar and the one Source label that a revision derives from a campaign name
  - Validate readable Source labels and their length as one filesystem component
  - Reject frozen scientific digests that conflict with the resolved campaign

Design principles:
  - Names describe the original design independently of successful selections
  - The configured horizon belongs to the simulation definition and is not repeated in names; only a deviating one, such as a short test run, is named
  - Material slugs and order come from the maintained material registry
  - A revision only relabels where a Source lives; it never enters seeds or the scientific digest
  - Seed domains, qualifiers, and filesystem component length are validated explicitly

This module does NOT:
  - Allocate candidates or choose successful Dataset membership
  - Load campaign configuration or persist campaign plans
"""

from __future__ import annotations

import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

from src import common

from . import generation_contracts_material_registry as material_registry
from .generation_contracts_vocabulary import DESIGN_KINDS

_DOMAIN_TOKENS: Final = MappingProxyType({"reference_range": "ref-range", "reference_and_parameter_extension": "ref-param-ext"})
SUPPORTED_DOMAIN_PROFILES: Final[frozenset[str]] = frozenset(_DOMAIN_TOKENS)
# A material's design kinds take one of these shapes: a test design alone (held-out material) or the ID designs.
DESIGN_SHAPES: Final = (("test",), ("train", "validation", "test"), DESIGN_KINDS)
_MAX_COMPONENT_BYTES: Final = 255
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z0-9]+(?:[-_+][a-z0-9]+)*(?:__[a-z0-9]+(?:[-_+][a-z0-9]+)*)*\Z")
_REVISION = re.compile(r"[a-z0-9]+(?:[-_+][a-z0-9]+)*\Z")


def domain_profile(designs: Mapping[str, Mapping[str, int]]) -> str:
    """Return the physical domain of a design set: the reference range, extended when any material has a parameter-OOD design."""
    return "reference_and_parameter_extension" if any("parameter_ood" in kinds for kinds in designs.values()) else "reference_range"


def domain_token(profile: str) -> str:
    """Return the short path-safe token that names a supported domain profile."""
    token = _DOMAIN_TOKENS.get(profile) if isinstance(profile, str) else None
    if token is None:
        message = f"Unsupported source domain profile {profile!r}."
        raise ValueError(message)
    return token


def horizon_token(horizon_hours: object) -> str:
    """Return the ``h-<hours>`` name token of a transient physical horizon, e.g. ``h-4`` or ``h-4p5``."""
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, (int, float)) or not math.isfinite(horizon_hours) or horizon_hours <= 0:
        message = "Physical horizon must be a positive finite number of hours."
        raise ValueError(message)
    return f"h-{common.names.number_token(horizon_hours)}"


def deviating_horizon_hours(effective_hours: float, configured_hours: float) -> float | None:
    """Return the effective horizon only when it differs from the configured one, else ``None`` (no name token).

    Both values are compared by their name token, so ``168`` and ``168.0`` are equal.
    """
    same = common.names.number_token(effective_hours) == common.names.number_token(configured_hours)
    return None if same else effective_hours


def canonical_campaign_name(
    designs: Mapping[str, Mapping[str, int]],
    *,
    root_seed: int,
    horizon_hours: float | None,
) -> str:
    """Name the full source design, independent of successful selections.

    Each material is ``<slug>-<n>...`` with its design sizes in :data:`DESIGN_KINDS` order, one number per kind of its
    shape (:data:`DESIGN_SHAPES`): ``kidney-bean-128`` is a test design alone, ``lentil-512-64-128-128`` the Train,
    validation, test and parameter-OOD designs. Materials follow registry order, then come the derived domain token and
    ``sobol-seed-<root_seed>``. A horizon from :func:`deviating_horizon_hours` is named as ``__h-<hours>``; ``None``
    (steady flow or the configured horizon) adds no horizon token.
    """
    token = domain_token(domain_profile(designs))
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or not 0 <= root_seed < 2**32:
        message = "Campaign root_seed must be an unsigned 32-bit integer."
        raise ValueError(message)
    ordered = material_registry.ordered_materials(list(designs))
    if not ordered:
        message = "Campaign must declare at least one material."
        raise ValueError(message)
    members: list[str] = []
    for material in ordered:
        kinds = tuple(kind for kind in DESIGN_KINDS if kind in designs[material])
        if kinds not in DESIGN_SHAPES or len(kinds) != len(designs[material]):
            message = f"Design kinds of {material!r} must be one of {[list(shape) for shape in DESIGN_SHAPES]}, got {sorted(designs[material])}."
            raise ValueError(message)
        counts = [designs[material][kind] for kind in kinds]
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts):
            message = f"Original design sizes for {material!r} must be positive integers."
            raise ValueError(message)
        members.append("-".join((material_registry.material_slug(material), *map(str, counts))))
    components = ["+".join(members), token, f"sobol-seed-{root_seed}"]
    if horizon_hours is not None:
        components.append(horizon_token(horizon_hours))
    name = "__".join(components)
    if len(name.encode("ascii")) > _MAX_COMPONENT_BYTES:
        message = "Canonical campaign name exceeds a filesystem component limit."
        raise ValueError(message)
    return name


def revision_label(value: object) -> str:
    """Validate a revision label and return it unchanged.

    Parameters
    ----------
    value : object
        Candidate label: lowercase letters and digits joined by single ``-``, ``_`` or ``+``.

    Returns
    -------
    str
        The accepted label.

    Raises
    ------
    ValueError
        If the label is not text or does not follow the grammar.

    """
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        message = f"Revision {value!r} must use lowercase letters and digits joined by single '-', '_' or '+'."
        raise ValueError(message)
    return value


def revised_source_label(campaign_name: str, revision: object) -> str:
    """Return the Source label of a campaign under a revision: ``<campaign_name>__rev-<revision>``.

    Parameters
    ----------
    campaign_name : str
        Canonical campaign name of the original design.
    revision : object
        Revision label following :func:`revision_label`.

    Returns
    -------
    str
        The label that names the revised Source; seeds and the scientific digest do not change.

    Raises
    ------
    ValueError
        If the revision does not follow the label grammar.

    """
    return f"{campaign_name}__rev-{revision_label(revision)}"


def source_name(campaign_name: object, campaign_digest: object) -> str:
    """Validate a readable Source label and the digest it is frozen with; the digest is not part of the name."""
    if not isinstance(campaign_name, str) or _NAME.fullmatch(campaign_name) is None:
        message = "Source design name is malformed."
        raise ValueError(message)
    if not isinstance(campaign_digest, str) or _DIGEST.fullmatch(campaign_digest) is None:
        message = "Source campaign digest must be one lowercase SHA-256 value."
        raise ValueError(message)
    if len(campaign_name.encode("ascii")) > _MAX_COMPONENT_BYTES:
        message = "Source name exceeds a filesystem component limit."
        raise ValueError(message)
    return campaign_name


def assert_campaign_semantics(name: str, expected_digest: str, existing_digest: str) -> None:
    """Reject a same-name campaign with different frozen scientific semantics."""
    if _DIGEST.fullmatch(expected_digest) is None or _DIGEST.fullmatch(existing_digest) is None:
        message = "Campaign semantic digest must be one lowercase SHA-256 value."
        raise ValueError(message)
    if expected_digest != existing_digest:
        message = (
            f"Source {name!r} has conflicting frozen scientific semantics: expected {expected_digest}, existing {existing_digest}. "
            "Add --revision LABEL to the command you run (a Dataset plan run also needs --new-source) to keep the changed scientific content "
            "beside it; the existing Source cannot be overwritten."
        )
        raise ValueError(message)
