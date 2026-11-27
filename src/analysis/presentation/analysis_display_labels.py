"""Own concise scientific display labels while preserving canonical identities separately.

Responsibilities:
  - Humanize maintained task, material, physical domain, run role, evaluation protocol and rollout mode identifiers
  - Hold the authoritative metadata required for one visible dataset label
  - Disambiguate colliding dataset labels with the shortest stable identity prefix
  - Make short run and Dataset labels unique with one separating facet, shortened by common affixes and a middle ellipsis
  - Write table column keys as headings, record tokens and qualified metric labels in sentence case, generation case ids as
    ``case 48`` and an exact identity shortened in place
  - Preserve canonical identities as metadata rather than parsing presentation text

Design principles:
  - Presentation labels never become storage, scientific, or persistence identities
  - Source domains and Experiment evaluation roles remain distinct visible concepts
  - Collision suffixes are deterministic and appear only when required for uniqueness
  - A field key inside a table key keeps its plain symbol and a water density names its stored basis (``w_surf (water density)``);
    a normalized or relative metric heading carries ``[-]``

This module does NOT:
  - Discover datasets, infer metadata, or validate persisted artifacts
  - Define task capabilities or scientific field semantics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from src.analysis import generation_inputs

from . import analysis_field_labels as field_labels

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_COLLISION_MINIMUM_SIZE = 2
_FACET_MINIMUM_WIDTH = 24
_FACET_TOKEN_SEPARATOR = re.compile(r"/|__")
_LABEL_BREAKS = re.compile(r"(?<=\+)|(?<=__)|(?<= )")  # a wrapped dataset label may break after a space, a ``+`` or a ``__``
_TASK_LABELS = {
    "steady_flow": "Airflow",
    "transient_drying": "Drying",
}
_EVALUATION_ROLE_LABELS = {
    "id": "ID",
    "eval": "ID",  # the split role a steady artifact stores for its ID set
    "ood": "OOD",
    "material_ood": "Material OOD",
    "parameter_ood": "Parameter OOD",
}
_EVALUATION_PROTOCOL_LABELS = {
    "normalized_drying_group_macro_rmse": "Drying group macro RMSE",
    "normalized_group_macro_rmse": "Normalized group macro RMSE",
}
_EVALUATION_MODE_LABELS = {
    "autonomous_full": "Full rollout",
    "rolling_origin": "Rolling origin",
    "teacher_forced_one_step": "Teacher-forced one step",
}
# Table headings of keys whose words read poorly; every other key reads as its words (``table_heading``).
_TABLE_HEADINGS = {
    "frame": "Artifact",
    "dataset_role": "Role",
    "material_family": "Material",
    "architecture_family": "Model family",
    "model_kind": "Model family",
    "physics_enabled": "Physics loss",
    "requested_horizon": "Horizon",
    "contributing_record_count": "Records",
    "sample_count": "Cases",
    "optimisation_seconds": "Optimization time [s]",  # the stored key keeps its spelling; the headings read as the configuration's
    "total_seconds": "Total time [s]",
    "attempted_seconds": "Attempted time [s]",
    "inference_seconds_per_case": "Inference time per case [s]",
}
_FIELD_KEY_PARTS = (4, 3, 2, 1)  # the most underscore-separated parts a field key spans (``X_0_db_field``), longest first
_DIMENSIONLESS_PREFIXES = ("normalized_", "physical_relative_")  # table keys of dimensionless metrics
_ACRONYMS = frozenset({"comsol", "cuda", "cpu", "fno", "h1", "id", "l2", "mae", "mse", "ood", "rmse", "rno"})
_TOKEN_SPELLINGS = {"adamw": "AdamW"}  # tokens whose name is neither lower case nor an acronym
_CASE_ID = re.compile(r"case_(\d+)")


@dataclass(frozen=True, slots=True)
class DatasetDisplayMetadata:
    """Hold the authoritative metadata required to label one dataset."""

    task_id: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str | None
    domain_profile: str | None
    canonical_identity: str

    def __post_init__(self) -> None:
        """Validate independently owned display and identity metadata."""
        _require_identifier(self.task_id, label="task_id")
        _require_identifier(self.material_family, label="material_family")
        _require_identifier(self.sampling_regime, label="sampling_regime")
        for label, value in (
            ("campaign_purpose", self.campaign_purpose),
            ("domain_profile", self.domain_profile),
        ):
            if value is not None:
                _require_identifier(value, label=label)
        _require_identity(self.canonical_identity)


def humanize_identifier(identifier: str) -> str:
    """
    Convert one underscore-delimited identifier to a concise display label.

    Parameters
    ----------
    identifier : str
        Non-empty canonical identifier. It is retained unchanged by callers.

    Returns
    -------
    str
        Title-cased display text with underscores replaced by spaces.

    Raises
    ------
    TypeError
        If ``identifier`` is not text.
    ValueError
        If ``identifier`` is blank or has empty underscore-delimited components.

    """
    _require_identifier(identifier, label="identifier")
    return " ".join(component.upper() if component.casefold() in _ACRONYMS else component.capitalize() for component in identifier.split("_"))


def task_display_label(task_id: str) -> str:
    """
    Return the centralized human-readable label for one task identifier.

    Parameters
    ----------
    task_id : str
        Canonical task identifier.

    Returns
    -------
    str
        Humanized task label.

    """
    _require_identifier(task_id, label="task_id")
    return _TASK_LABELS.get(task_id, humanize_identifier(task_id))


def material_display_label(material_family: str) -> str:
    """Return the one human-readable label of a material family that every view writes (``field_pea`` reads ``Field pea``)."""
    _require_identifier(material_family, label="material_family")
    return material_family.replace("_", " ").capitalize()


def regime_display_label(sampling_regime: str) -> str:
    """Return the centralized human-readable label for one sampling regime."""
    return humanize_identifier(sampling_regime)


def evaluation_role_display_label(dataset_role: str) -> str:
    """Return the run-relative scientific role for one Evaluation artifact."""
    _require_identifier(dataset_role, label="dataset_role")
    return _EVALUATION_ROLE_LABELS.get(dataset_role, humanize_identifier(dataset_role))


def material_role_display_label(material_family: str, dataset_role: str) -> str:
    """Return one dataset as its material and run-relative Evaluation role, one unit inside a dotted run label (``Chickpea, ID``)."""
    return f"{material_display_label(material_family)}, {evaluation_role_display_label(dataset_role)}"


def evaluation_protocol_display_label(protocol: str) -> str:
    """Return the shared human-facing label for one exact Evaluation objective."""
    _require_identifier(protocol, label="protocol")
    return _EVALUATION_PROTOCOL_LABELS.get(protocol, humanize_identifier(protocol))


def evaluation_mode_display_label(mode: str) -> str:
    """Return the display name of one transient evaluation mode (``autonomous_full`` reads ``Full rollout``)."""
    _require_identifier(mode, label="mode")
    return _EVALUATION_MODE_LABELS.get(mode, humanize_identifier(mode))


def qualified_label(qualifier: str, label: str) -> str:
    """Return ``label`` after a prose ``qualifier`` as one sentence-case label (``Cumulative drying group macro RMSE``)."""
    first, _, rest = label.partition(" ")
    return " ".join(part for part in (qualifier, first if first.isupper() else first.lower(), rest) if part)


def token_display_label(token: str) -> str:
    """Return one lower-case record token as words in sentence case with acronyms upper case (``no_eligible_checkpoint``)."""
    _require_identifier(token, label="token")
    text = " ".join(_TOKEN_SPELLINGS.get(word) or (word.upper() if word in _ACRONYMS else word) for word in token.split("_"))
    return text[:1].upper() + text[1:]


def compact_identity_label(identity: str) -> str:
    """Return an exact identity as a view shows it in place, ``head…tail`` beyond 24 characters; the exact identity stands elsewhere."""
    _require_identity(identity)
    return _ellipsized(identity, _FACET_MINIMUM_WIDTH)


def case_display_label(case_id: str) -> str:
    """Return a generation case id as the views write it (``case_0048`` reads ``case 48``); another id stays as it is."""
    match = _CASE_ID.fullmatch(case_id)
    return f"case {int(match.group(1))}" if match else case_id


def table_heading(key: str) -> str:
    """
    Return the display heading of one table column key with its unit: a declared heading, else the key in words.

    The words read in sentence case with acronyms upper case; a field key inside the key stays one word in its plain symbol,
    a water density with its basis (``normalized_rmse_w_surf`` reads ``Normalized RMSE w_surf (water density) [-]``). A
    bracketed unit reads as ``field_labels.display_unit`` gives it (``kg/m^3`` as ``kg/m³``), a trailing seconds token ``s``
    as ``[s]``; a normalized or relative metric without one is dimensionless ``[-]``, its case counts stay without a unit.
    """
    base, _, unit = key.partition(" [")
    unit = field_labels.display_unit(unit.removesuffix("]")) if unit else ""
    heading = _TABLE_HEADINGS.get(base) or _EVALUATION_PROTOCOL_LABELS.get(base)
    if heading is None:
        parts, words = [part for part in re.split(r"[_ ]", base) if part], []
        while parts:
            size = next(size for size in _FIELD_KEY_PARTS if size == 1 or _is_field_key("_".join(parts[:size])))
            word, parts = "_".join(parts[:size]), parts[size:]
            words.append(_field_word(word) if _is_field_key(word) else word.upper() if word in _ACRONYMS else word)
        if words[-1:] == ["s"] and not unit:  # a component clock name such as ``comsol_solver_s`` ends in its unit
            words, unit = words[:-1], "s"
        heading = " ".join(words)
        heading = heading[:1].upper() + heading[1:]
    if not unit and base.startswith(_DIMENSIONLESS_PREFIXES) and not base.endswith("_cases"):
        unit = field_labels.display_unit("1")
    return f"{heading} [{unit}]" if unit else heading


def _field_word(field: str) -> str:
    """Return a field key of a table key as its plain symbol, a water density with its basis (``w_surf (water density)``)."""
    symbol = field_labels.field_label(field, control=True)
    return f"{symbol} (water density)" if field_labels.field_display_metadata(field).wet_basis is not None else symbol


def _is_field_key(text: str) -> bool:
    """Return whether ``text`` is a field key with a declared presentation (``w_surf``, ``T``)."""
    return bool(text) and text == text.strip() and field_labels.has_declared_field_metadata(text)


def dataset_display_label(metadata: DatasetDisplayMetadata) -> str:
    """
    Return the ordinary task-qualified dataset label without collision evidence.

    Parameters
    ----------
    metadata : DatasetDisplayMetadata
        Canonical metadata retained separately from the returned text.

    Returns
    -------
    str
        Task, material or variation, abbreviated campaign, and physical domain.

    """
    if not isinstance(metadata, DatasetDisplayMetadata):
        message = "metadata must be DatasetDisplayMetadata."
        raise TypeError(message)
    parts = [
        task_display_label(metadata.task_id),
        material_display_label(metadata.material_family),
    ]
    if metadata.campaign_purpose is not None:
        parts.append(
            generation_inputs.labels.campaign_purpose_abbreviation(
                metadata.campaign_purpose,
            )
        )
    if metadata.domain_profile is not None:
        parts.append(humanize_identifier(metadata.domain_profile))
    return " · ".join(parts)


def wrapped_dataset_display_label(
    label: str,
    *,
    width: int = 32,
    max_lines: int | None = None,
) -> str:
    """
    Wrap one complete visible dataset label without removing identity text.

    Lines hold at most ``width`` characters and break after a space (the line break replaces the spaces before it), a ``+`` or a ``__`` (a
    dataset name has no spaces), never inside a word; ``max_lines`` widens the lines until the label fits that many. A single
    unbreakable word stays whole.
    """
    if not isinstance(label, str):
        message = "Dataset display labels must be text."
        raise TypeError(message)
    if not label or label != label.strip():
        message = "Dataset display labels must be non-empty without surrounding whitespace."
        raise ValueError(message)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        message = "Dataset display-label wrap width must be a positive integer."
        raise ValueError(message)
    if max_lines is not None and (isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1):
        message = "Dataset display-label line limit must be a positive integer."
        raise ValueError(message)
    pieces = _LABEL_BREAKS.split(label)
    while True:
        lines = [""]
        for piece in pieces:
            if lines[-1] and len(lines[-1]) + len(piece.rstrip()) > width:
                lines.append("")
            lines[-1] += piece
        if max_lines is None or len(lines) <= max_lines:
            return "\n".join(line.rstrip() for line in lines)
        width += 1


def dataset_display_labels(datasets: Iterable[DatasetDisplayMetadata]) -> tuple[str, ...]:
    """
    Build deterministic concise labels and disambiguate genuine collisions.

    Parameters
    ----------
    datasets : Iterable[DatasetDisplayMetadata]
        Dataset metadata in caller-selected order.

    Returns
    -------
    tuple[str, ...]
        One visible label per input dataset. A shortest unique canonical-identity
        prefix is appended only to members sharing the same ordinary label.

    Raises
    ------
    TypeError
        If an input is not DatasetDisplayMetadata.
    ValueError
        If canonical identities repeat or cannot distinguish a collision.

    Notes
    -----
    Canonical identities are never parsed. They remain separately available in
    each ``DatasetDisplayMetadata`` instance.

    """
    resolved_datasets = tuple(datasets)
    if any(not isinstance(dataset, DatasetDisplayMetadata) for dataset in resolved_datasets):
        message = "datasets must contain only DatasetDisplayMetadata instances."
        raise TypeError(message)
    identities = tuple(dataset.canonical_identity for dataset in resolved_datasets)
    if len(identities) != len(set(identities)):
        message = "Dataset display metadata must have unique canonical identities."
        raise ValueError(message)
    bases = tuple(dataset_display_label(dataset) for dataset in resolved_datasets)
    labels = list(bases)
    for base in sorted(set(bases)):
        positions = tuple(index for index, candidate in enumerate(bases) if candidate == base)
        if len(positions) < _COLLISION_MINIMUM_SIZE:
            continue
        prefixes = _unique_identity_prefixes(tuple(identities[index] for index in positions))
        for position, prefix in zip(positions, prefixes, strict=True):
            labels[position] = f"{base} · {prefix}"
    return tuple(labels)


def unique_display_labels(bases: Sequence[str], facets: Sequence[Sequence[str | None]]) -> tuple[str, ...]:
    """
    Make equal base labels unique by appending the first separating facet, again while a separated label meets another.

    Parameters
    ----------
    bases : Sequence[str]
        One short label per item.
    facets : Sequence[Sequence[str | None]]
        One tuple per item with its facet values in priority order; ``None``
        disqualifies the facet for the whole collision group.

    Returns
    -------
    tuple[str, ...]
        ``base`` alone when unique, otherwise ``base · facet`` with the facet
        reduced to the ``/``- and ``__``-separated tokens that not all items of
        its group share and, beyond 24 characters, shown as ``head…tail`` with
        the smallest unique length. An item without a distinguishing token keeps
        the bare ``base``.

    Raises
    ------
    ValueError
        If the inputs differ in length or no facet separates a collision group.

    """
    if len(bases) != len(facets):
        message = "bases and facets must have one entry per item."
        raise ValueError(message)
    labels = tuple(bases)
    # Each round separates every group of equal labels or raises; one round per item bounds rare chains of new meetings.
    for _round in range(len(labels) + 1):
        if len(set(labels)) == len(labels):
            return labels
        labels = _separated_once(labels, facets)
    message = "The facets could not make the labels unique."
    raise ValueError(message)


def _separated_once(bases: tuple[str, ...], facets: Sequence[Sequence[str | None]]) -> tuple[str, ...]:
    """Append the first separating facet to every group of equal ``bases``; raise if no facet separates one."""
    labels = list(bases)
    for base in dict.fromkeys(bases):
        group = tuple(index for index, candidate in enumerate(bases) if candidate == base)
        if len(group) < _COLLISION_MINIMUM_SIZE:
            continue
        for facet_index in range(max(len(facets[index]) for index in group)):
            values = tuple(facets[index][facet_index] if facet_index < len(facets[index]) else None for index in group)
            if None in values:
                continue
            shown_values = _shorten_facet_values(cast("tuple[str, ...]", values))
            if shown_values is not None:
                for index, shown in zip(group, shown_values, strict=True):
                    labels[index] = f"{base} · {shown}" if shown else base
                break
        else:
            message = f"No facet separates the items labelled {base!r}."
            raise ValueError(message)
    return tuple(labels)


def _shorten_facet_values(values: tuple[str, ...]) -> tuple[str, ...] | None:
    """Keep the tokens not shared by all values, ellipsize to the smallest unique length; ``None`` if not separating."""
    tokens = tuple(_FACET_TOKEN_SEPARATOR.split(value) for value in values)
    shared = set.intersection(*(set(item) for item in tokens))
    cores = tuple(" ".join(token for token in item if token not in shared) for item in tokens)
    if len(set(cores)) != len(cores):
        return None
    for width in range(_FACET_MINIMUM_WIDTH, max(len(core) for core in cores) + 1):
        shown = tuple(_ellipsized(core, width) for core in cores)
        if len(set(shown)) == len(shown):
            return shown
    return cores


def _ellipsized(text: str, width: int) -> str:
    """Return ``text`` whole up to ``width`` characters, else as ``head…tail`` of that width."""
    head = width // 2
    return text if len(text) <= width else f"{text[:head]}…{text[len(text) - (width - head - 1) :]}"


def _unique_identity_prefixes(identities: tuple[str, ...]) -> tuple[str, ...]:
    """Return the shortest deterministic unique prefix for each identity."""
    maximum_width = max(len(identity) for identity in identities)
    for width in range(1, maximum_width + 1):
        prefixes = tuple(identity[:width] for identity in identities)
        if len(prefixes) == len(set(prefixes)):
            return prefixes
    message = "Colliding dataset labels require distinct canonical identities."
    raise ValueError(message)


def _require_identifier(value: str, *, label: str) -> None:
    """Require one non-empty underscore-delimited presentation identifier."""
    if not isinstance(value, str):
        message = f"{label} must be text."
        raise TypeError(message)
    if not value or value != value.strip() or any(not component for component in value.split("_")):
        message = f"{label} must be non-empty underscore-delimited text."
        raise ValueError(message)


def _require_identity(value: str) -> None:
    """Require canonical identity text without interpreting its structure."""
    if not isinstance(value, str):
        message = "canonical_identity must be text."
        raise TypeError(message)
    if not value or value != value.strip():
        message = "canonical_identity must be non-empty text without surrounding whitespace."
        raise ValueError(message)
