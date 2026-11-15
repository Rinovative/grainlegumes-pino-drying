"""Resolve evaluation datasets as (material, role) items and the frames or cases each choice draws.

Responsibilities:
  - Build the dataset inventory of steady role frames, of comparison role frames per model, and of a transient session
  - Cut a steady or comparison frame to the materials of a choice and recompute its objective material-balanced like the artifact objective
  - Resolve transient choices to their frame names and case identities
  - Keep roles the workflow refused, and materials without shared cases, visible as unavailable with their reason
  - Name the steady views that read the recomputed objective and the reason the pooling switch stays off elsewhere

Design principles:
  - The role of a steady frame comes from its explicit saved split role, never from names or paths
  - The objective of a cut is finalized by the artifact contract (equal weight per material, roots after the means), never by a row mean
  - A comparison cut keeps exactly the cases every model shares, so the compared frames stay case-matched
  - A cut deep-copies the run-level attributes of its source frame and records the effective membership of its rows and the materials
    of its choice in the selection provenance the comparison admission compares, so cuts holding the same cases are comparable and
    cuts of different materials are separate datasets; the unchanged source provenance stays as ``attrs["source_artifact_provenance"]``
  - A cut of a frame bound to a live session is bound once in that session, so redraws reuse its caches

This module does NOT:
  - Load artifacts, admit runs, or decide which role the workflow refuses
  - Construct widgets, choose colors, or draw
  - Pool transient datasets, because no material-balanced transient view reduction exists
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np

from src.analysis.artifacts import contracts
from src.analysis.presentation import dataset_selection, display_labels

from . import evaluation_dataframe as dataframe
from . import evaluation_session

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import pandas as pd

    from .evaluation_transient_session import TransientCaseInventoryEntry, TransientEvaluationSession

# Steady views whose reduction is the artifact objective's material-balanced one (run summary table, relative scoreboard,
# accuracy-physics Pareto, capacity), so a pooled cut stays correct there.
POOLABLE_STEADY_VIEWS: Final = frozenset(
    {
        "overview_summary_table",
        "overview_global_comparison_summary",
        "overview_accuracy_physics_pareto",
        "architecture_capacity_accuracy",
    }
)
POOL_DISABLED_REASON: Final = f"Zusammenfassen hier noch nicht materialausgeglichen (folgt mit P2{chr(0x2013)}P4)"
EVALUATION_GROUP_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {role: display_labels.evaluation_role_display_label(role) for role in dataset_selection.EVALUATION_GROUP_ORDER}
)
_STEADY_ROLE_BY_SPLIT: Final = {"eval": "id", "material_ood": "material_ood", "parameter_ood": "parameter_ood"}
# The exact-grid admission cache of a frame lists the payloads of all its rows; a cut would fail it as changed membership.
_GRID_CACHE_ATTR: Final = "_exact_comparison_grid_cache"
# A cut is a new frame, so it must not inherit the live session reference of its source (a deepcopy of the session returns the
# session itself): the source's live session binds it explicitly, and an unbound cut gets an ephemeral session like any direct caller.
_UNSHARED_ATTRS: Final = frozenset({_GRID_CACHE_ATTR, evaluation_session.SESSION_ATTR, evaluation_session.SESSION_KEY_ATTR})
_NO_REFUSED: Final[Mapping[str, str]] = MappingProxyType({})
# A role refused as a whole has no known materials; its unavailable item carries this empty material.
_WHOLE_ROLE: Final = ""


@dataclass(frozen=True, slots=True)
class TransientDatasetContext:
    """Bind one transient dataset choice to the session frames and exact cases it draws."""

    label: str
    frame_names: tuple[str, ...]
    case_ids: tuple[str, ...]


def _role_label(role: str) -> str:
    """Return the display label of one evaluation role or name the known roles."""
    try:
        return EVALUATION_GROUP_LABELS[role]
    except KeyError:
        message = f"Unknown evaluation role {role!r}; expected one of {list(EVALUATION_GROUP_LABELS)}."
        raise ValueError(message) from None


def _item(material: str, role: str) -> dataset_selection.DatasetItem:
    """Return the dataset item of one material in one role, keyed ``(material, role)``."""
    return dataset_selection.DatasetItem((material, role), material, role, dataset_selection.dataset_label(material, _role_label(role)))


def _inventory(
    items: Iterable[dataset_selection.DatasetItem],
    unavailable: Iterable[tuple[dataset_selection.DatasetItem, str]],
    *,
    poolable: bool,
) -> dataset_selection.DatasetInventory:
    """Return the evaluation inventory over the fixed role order and labels."""
    return dataset_selection.DatasetInventory(
        tuple(items),
        group_order=dataset_selection.EVALUATION_GROUP_ORDER,
        group_labels=EVALUATION_GROUP_LABELS,
        poolable=poolable,
        unavailable=tuple(unavailable),
    )


def _refused_items(refused: Mapping[str, str]) -> list[tuple[dataset_selection.DatasetItem, str]]:
    """Return one unavailable role-level item per role the workflow refused, with its reason."""
    return [(dataset_selection.DatasetItem((_WHOLE_ROLE, role), _WHOLE_ROLE, role, _role_label(role)), reason) for role, reason in refused.items()]


def _steady_role(frame: pd.DataFrame) -> str:
    """Return the evaluation role of one steady frame from its explicit saved split role."""
    role = _STEADY_ROLE_BY_SPLIT.get(str(frame.attrs.get("dataset_role")))
    if role is None:
        message = f"Steady frame has no explicit split role (dataset_role={frame.attrs.get('dataset_role')!r})."
        raise ValueError(message)
    return role


def _frames_by_role(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return the steady frames keyed by their evaluation role, refusing two frames of one role."""
    by_role: dict[str, pd.DataFrame] = {}
    holders: dict[str, str] = {}
    for name, frame in frames.items():
        role = _steady_role(frame)
        if role in holders:
            message = f"Steady frames {holders[role]!r} and {name!r} both hold role {role!r}; expected one frame per role."
            raise ValueError(message)
        holders[role] = name
        by_role[role] = frame
    return by_role


def _cases_by_material(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Return the exact case identities of every material of one frame."""
    cases: dict[str, set[str]] = {}
    for case_id, material in contracts.artifact_case_membership(frame):
        cases.setdefault(material, set()).add(case_id)
    return cases


def _cut(frame: pd.DataFrame, keep: Sequence[bool], *, materials: Iterable[str]) -> pd.DataFrame:
    """
    Return the kept rows, the objective recomputed material-balanced on them, and their membership and materials as provenance.

    A cut of a frame bound to a live session is bound once in that session, so every redraw returns the same frame and hits its caches.
    """
    provenance = dataframe.require_complete_provenance(frame)
    subset = frame[np.asarray(keep, dtype=bool)].copy()
    aggregate = contracts.aggregate_normalized_group_macro_rmse(
        subset,
        output_groups=provenance["evaluator"]["output_groups"],
        train_scales=frame.attrs["train_scales"],
        reference_floors=frame.attrs["reference_floors"],
    )
    attrs = {key: deepcopy(value) for key, value in frame.attrs.items() if key not in _UNSHARED_ATTRS}
    selection = {
        **attrs["artifact_provenance"]["selection"],
        "effective_ordered_source_indices_sha256": contracts.ordered_indices_sha256(int(index) for index in subset["source_index"]),
        "effective_case_count": len(subset),
        "material_families": sorted(materials),
    }
    subset.attrs = {
        **attrs,
        "artifact_provenance": {**attrs["artifact_provenance"], "selection": selection},
        "source_artifact_provenance": deepcopy(dict(provenance)),
        dataframe.PRIMARY_OBJECTIVE_ID: dict(aggregate),
    }
    session = evaluation_session.bound_session(frame)
    return subset if session is None else session.bind_cut(frame, subset)


def _require_available(choice: dataset_selection.DatasetChoice) -> None:
    """Refuse to draw a choice that is listed as unavailable."""
    if not choice.available:
        message = f"Dataset choice {choice.label!r} is unavailable and cannot be drawn."
        raise ValueError(message)


def _role_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Return the frame after requiring that its saved split role is the expected one."""
    if _steady_role(frame) != role:
        message = f"Frame with split role {frame.attrs.get('dataset_role')!r} was given for role {role!r}."
        raise ValueError(message)
    return frame


def _entry_role(entry: TransientCaseInventoryEntry) -> str:
    """Return the declared assignment role of one transient case, else its artifact role."""
    assignment = entry.evaluation_assignment
    return str(assignment["role"]) if assignment is not None else str(entry.dataset_role)


def steady_inventory(
    frames: Mapping[str, pd.DataFrame],
    *,
    unavailable: Iterable[tuple[dataset_selection.DatasetItem, str]] = (),
) -> dataset_selection.DatasetInventory:
    """
    Build the poolable inventory of steady role frames: one item per material and role.

    Parameters
    ----------
    frames : Mapping[str, pandas.DataFrame]
        Loaded steady artifact frames, at most one per role; their ``attrs["dataset_role"]`` names the role.
    unavailable : Iterable[tuple[DatasetItem, str]], optional
        Items the caller knows to be unavailable, each with its reason.

    Returns
    -------
    DatasetInventory
        Items ``(material, role)`` with materials sorted inside each role.

    Raises
    ------
    ValueError
        If a frame has no explicit split role or two frames hold the same role.

    """
    items = [_item(material, role) for role, frame in _frames_by_role(frames).items() for material in sorted(_cases_by_material(frame))]
    return _inventory(items, unavailable, poolable=True)


def steady_frames(frames: Mapping[str, pd.DataFrame], choices: Iterable[dataset_selection.DatasetChoice]) -> dict[str, pd.DataFrame]:
    """
    Cut the steady role frames to the materials of each choice, keyed by the choice label.

    Each cut keeps the run-level attributes and carries ``attrs["normalized_group_macro_rmse"]`` recomputed
    on its rows through the artifact contract: the mean over cases per material, the equal-weight mean over
    the materials, then the roots. A pooled choice is therefore never a row mean.

    Raises
    ------
    ValueError
        If two frames hold one role, a choice is unavailable, names a role or material no frame holds, or two choices share a label.

    """
    by_role = _frames_by_role(frames)
    drawn: dict[str, pd.DataFrame] = {}
    for choice in choices:
        _require_available(choice)
        if choice.group not in by_role:
            message = f"No steady frame holds role {choice.group!r} for dataset choice {choice.label!r}."
            raise ValueError(message)
        frame = by_role[choice.group]
        materials = {item.material for item in choice.items}
        missing = materials - _cases_by_material(frame).keys()
        if missing:
            message = f"Steady frame of role {choice.group!r} holds no cases of materials {sorted(missing)}."
            raise ValueError(message)
        if choice.label in drawn:
            message = f"Dataset choice label {choice.label!r} is not unique."
            raise ValueError(message)
        keep = [material in materials for _case_id, material in contracts.artifact_case_membership(frame)]
        drawn[choice.label] = _cut(frame, keep, materials=materials)
    return drawn


def comparison_inventory(
    frames_by_role: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    refused: Mapping[str, str],
) -> dataset_selection.DatasetInventory:
    """
    Build the poolable inventory of a model comparison from the frames of every model per role.

    A ``(material, role)`` item is available when the material has cases that every model of that role shares.
    Otherwise it is unavailable with «keine gemeinsamen Fälle in <runs>», naming the models that lack the
    material or, when all hold it, every model. A role the workflow refused is one unavailable role-level item.

    Parameters
    ----------
    frames_by_role : Mapping[str, Mapping[str, pandas.DataFrame]]
        Role to model label to the loaded frame of that model and role.
    refused : Mapping[str, str]
        Role to the workflow's reason for refusing it.

    Raises
    ------
    ValueError
        If a role is unknown or refused and loaded at once, or a frame's split role contradicts its key.

    """
    items: list[dataset_selection.DatasetItem] = []
    unavailable = _refused_items(refused)
    for role, frames in frames_by_role.items():
        _role_label(role)
        if role in refused:
            message = f"Role {role!r} is both refused and loaded."
            raise ValueError(message)
        cases = {model: _cases_by_material(_role_frame(frame, role)) for model, frame in frames.items()}
        for material in sorted({material for held in cases.values() for material in held}):
            lacking = [model for model, held in cases.items() if material not in held]
            shared = set.intersection(*(held[material] for held in cases.values() if material in held))
            if not lacking and shared:
                items.append(_item(material, role))
            else:
                runs = ", ".join(lacking or cases)
                unavailable.append((_item(material, role), f"keine gemeinsamen Fälle in {runs}"))
    return _inventory(items, unavailable, poolable=True)


def comparison_frames(frames_by_model: Mapping[str, pd.DataFrame], choice: dataset_selection.DatasetChoice) -> dict[str, pd.DataFrame]:
    """
    Cut the frames of every model of one role to a choice, keyed by model label.

    Per material of the choice, only the cases every model holds stay, so the cut frames are case-matched;
    each carries the objective recomputed material-balanced on its rows. ``frames_by_model`` holds the frames
    of the role ``choice.group``, as in :func:`comparison_inventory`.

    Raises
    ------
    ValueError
        If the choice is unavailable, a frame belongs to another role, or a material has no shared case.

    """
    _require_available(choice)
    if not frames_by_model:
        message = "A comparison cut needs the frames of at least one model."
        raise ValueError(message)
    memberships = {model: contracts.artifact_case_membership(_role_frame(frame, choice.group)) for model, frame in frames_by_model.items()}
    shared: dict[str, set[str]] = {}
    for material in {item.material for item in choice.items}:
        shared[material] = set.intersection(*({case_id for case_id, held in rows if held == material} for rows in memberships.values()))
        if not shared[material]:
            message = f"Material {material!r} has no case shared by all models of role {choice.group!r}."
            raise ValueError(message)
    return {
        model: _cut(frames_by_model[model], [case_id in shared.get(material, ()) for case_id, material in rows], materials=shared)
        for model, rows in memberships.items()
    }


def transient_inventory(session: TransientEvaluationSession, *, refused: Mapping[str, str] = _NO_REFUSED) -> dataset_selection.DatasetInventory:
    """
    Build the non-poolable inventory of a transient session: one item per material and case role.

    The role of a case is its declared evaluation-assignment role, else the role of its artifact.
    A role the workflow refused is one unavailable role-level item with its reason.
    """
    pairs = {(_entry_role(entry), entry.material_family) for entry in session.case_inventory()}
    items = [_item(material, role) for role, material in sorted(pairs, key=lambda pair: (pair[1], pair[0]))]
    return _inventory(items, _refused_items(refused), poolable=False)


def transient_contexts(
    session: TransientEvaluationSession,
    choices: Iterable[dataset_selection.DatasetChoice],
) -> tuple[TransientDatasetContext, ...]:
    """
    Resolve transient choices to the session frames and exact cases they draw.

    Raises
    ------
    ValueError
        If a choice is unavailable or matches no case of the session.

    """
    entries = session.case_inventory()
    contexts: list[TransientDatasetContext] = []
    for choice in choices:
        _require_available(choice)
        wanted: set[tuple[str, str]] = {(item.material, item.group) for item in choice.items}
        matching = [entry for entry in entries if (entry.material_family, _entry_role(entry)) in wanted]
        if not matching:
            message = f"Dataset choice {choice.label!r} matches no case of the transient session."
            raise ValueError(message)
        frame_names = tuple(dict.fromkeys(entry.frame_name for entry in matching))
        case_ids = tuple(dict.fromkeys(entry.case_id for entry in matching))
        contexts.append(TransientDatasetContext(choice.label, frame_names, case_ids))
    return tuple(contexts)
