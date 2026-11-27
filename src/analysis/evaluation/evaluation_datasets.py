"""Resolve evaluation datasets as (material, role) items and the frames or cases each choice draws.

Responsibilities:
  - Build the dataset inventory of steady role frames, of comparison role frames per model, and of a transient session
  - Cut a steady or comparison frame to each material of a choice and recompute that cut's objective through the artifact contract
  - Return cuts as ``DisplayFrames`` that label each frame with the dataset panel and the model series it is drawn under
  - Resolve transient choices to their frame names and case identities
  - Keep roles the workflow refused, and materials without shared cases, visible as unavailable with their reason
  - Name the steady views that pool materials equally and the reason the pooling switch stays off elsewhere

Design principles:
  - The role of a steady frame comes from its explicit saved split role, never from names or paths
  - The objective of a cut is finalized by the artifact contract on that cut's rows alone
  - A pooled choice is never one cut of appended rows: it yields one cut per material under the choice's label, and the pooling views
    reduce them material-weighted (``evaluation_plot_series.material_mean``)
  - A comparison cut keeps exactly the cases every model shares, so the compared frames stay case-matched
  - A cut takes its attrs from the session's one cut derivation: the run-level attributes of its source, the source provenance as
    ``attrs["source_artifact_provenance"]``, and the membership, materials, aggregate and prediction validity of its own rows, so cuts
    holding the same cases are comparable and cuts of different materials are separate datasets
  - A cut of a frame bound to a live session is bound once in that session, so redraws reuse its caches; a cut of an unbound frame
    stays unbound and binds only beside its source

This module does NOT:
  - Load artifacts, admit runs, or decide which role the workflow refuses
  - Construct widgets, choose colors, or draw
  - Pool transient datasets, because no material-balanced transient view reduction exists
  - Reduce the cuts of a pooled choice to one value
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np

from src.analysis.artifacts import contracts
from src.analysis.presentation import dataset_selection, display_labels

from . import evaluation_session

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Sequence

    import pandas as pd

    from .evaluation_transient_session import TransientCaseInventoryEntry, TransientEvaluationSession

# Steady views that reduce one value per model from the objective of each material's cut and weight the materials equally
# (relative scoreboard, capacity). Every other view shows per-case quantiles or row-weighted reductions, which would weight a
# pooled role by its case counts, so it never pools.
POOLABLE_STEADY_VIEWS: Final = frozenset({"overview_global_comparison_summary", "architecture_capacity_accuracy"})
POOL_DISABLED_REASON: Final = "Only in the overview and capacity views, where every material counts equally"
EVALUATION_GROUP_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {role: display_labels.evaluation_role_display_label(role) for role in dataset_selection.EVALUATION_GROUP_ORDER}
)
_STEADY_ROLE_BY_SPLIT: Final = {"eval": "id", "material_ood": "material_ood", "parameter_ood": "parameter_ood"}
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
    """Return the dataset item of one known material in one known role, keyed ``(material, role)``."""
    _role_label(role)
    return dataset_selection.DatasetItem((material, role), material, role, dataset_selection.dataset_label(material, role))


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


def _cut(frame: pd.DataFrame, keep: Sequence[bool]) -> pd.DataFrame:
    """
    Return the kept rows with the attrs of a cut, whose objective is recomputed on them.

    A cut of a frame bound to a live session is bound once in that session, so every redraw returns the same frame and hits its caches.
    """
    subset = frame[np.asarray(keep, dtype=bool)].copy()
    subset.attrs = evaluation_session.cut_attrs(frame, subset)
    session = evaluation_session.bound_session(frame)
    return subset if session is None else session.bind_cut(frame, subset)


class DisplayFrames(Mapping[str, "pd.DataFrame"]):
    """
    Frames by key with the panel and series label each one is drawn under (rule 17: datasets are panels, models are series).

    The frames are the (possibly session-bound) cuts themselves; the labels belong to this call's result, never to a shared frame.
    ``labels`` maps each key to ``(dataset label, model label)``; the frames of a pooled choice, one per material, share one pair
    (``evaluation_plot_series.panel_series`` groups them). ``unnamed`` holds the model labels that are no explicit short
    label (a run's storage alias): the legend shows them, a title names none (``title_labels``).
    """

    def __init__(self, frames: Mapping[str, pd.DataFrame], labels: Mapping[str, tuple[str, str]], *, unnamed: Collection[str] = ()) -> None:
        """Hold ``frames`` and the ``labels`` of the same keys, copied, and the ``unnamed`` model labels."""
        if frames.keys() != labels.keys():
            message = f"Display labels name keys {sorted(labels)} but the frames hold {sorted(frames)}."
            raise ValueError(message)
        self._frames = dict(frames)
        self.labels: Mapping[str, tuple[str, str]] = MappingProxyType(dict(labels))
        self.unnamed = frozenset(unnamed)

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Return the frame of ``key``."""
        return self._frames[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate the keys in insertion order."""
        return iter(self._frames)

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self._frames)

    @property
    def title_labels(self) -> dict[str, str]:
        """Return each key's model label for a title (``layout.models_suffix``); the key of an ``unnamed`` model gets ``""``, which names none."""
        return {key: "" if model in self.unnamed else model for key, (_dataset, model) in self.labels.items()}

    def only(self, key: str) -> DisplayFrames:
        """Return the one frame ``key`` with its label."""
        return DisplayFrames({key: self._frames[key]}, {key: self.labels[key]}, unnamed=self.unnamed)


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


def steady_frames(frames: Mapping[str, pd.DataFrame], choices: Iterable[dataset_selection.DatasetChoice], *, model: str = "") -> DisplayFrames:
    """
    Cut the steady role frames to each material of each choice, keyed by the item label.

    The result labels every cut with its choice label and ``model``, the label of the one model the frames belong to. A pooled
    choice gives one cut per material under its one label, never one cut of appended rows.

    Each cut keeps the run-level attributes and carries ``attrs["normalized_group_macro_rmse"]`` recomputed
    on its rows through the artifact contract: the mean over cases, the roots after it.

    Raises
    ------
    ValueError
        If two frames hold one role, a choice is unavailable, names a role or material no frame holds, or two items share a label.

    """
    by_role = _frames_by_role(frames)
    drawn: dict[str, pd.DataFrame] = {}
    labels: dict[str, tuple[str, str]] = {}
    for choice in choices:
        _require_available(choice)
        if choice.group not in by_role:
            message = f"No steady frame holds role {choice.group!r} for dataset choice {choice.label!r}."
            raise ValueError(message)
        frame = by_role[choice.group]
        membership = contracts.artifact_case_membership(frame)
        missing = {item.material for item in choice.items} - {material for _case_id, material in membership}
        if missing:
            message = f"Steady frame of role {choice.group!r} holds no cases of materials {sorted(missing)}."
            raise ValueError(message)
        for item in choice.items:
            if item.label in drawn:
                message = f"Dataset item label {item.label!r} is not unique."
                raise ValueError(message)
            drawn[item.label] = _cut(frame, [material == item.material for _case_id, material in membership])
            labels[item.label] = (choice.label, model)
    return DisplayFrames(drawn, labels)


def comparison_inventory(
    frames_by_role: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    refused: Mapping[str, str],
) -> dataset_selection.DatasetInventory:
    """
    Build the poolable inventory of a model comparison from the frames of every model per role.

    A ``(material, role)`` item is available when the material has cases that every model of that role shares.
    Otherwise it is unavailable with «no shared cases in <runs>», naming the models that lack the
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
                unavailable.append((_item(material, role), f"no shared cases in {runs}"))
    return _inventory(items, unavailable, poolable=True)


def comparison_frames(frames_by_model: Mapping[str, pd.DataFrame], choice: dataset_selection.DatasetChoice) -> DisplayFrames:
    """
    Cut the frames of every model of one role to each material of a choice, keyed by model label.

    A pooled choice is one cut per material and model, keyed ``model · item label``, never one cut of appended rows.
    Per material of the choice, only the cases every model holds stay, so the cut frames are case-matched;
    each carries the objective recomputed on its rows. The result labels every frame with the choice label
    and its model label. ``frames_by_model`` holds the frames of the role ``choice.group``, as in :func:`comparison_inventory`.

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
    cuts: dict[str, pd.DataFrame] = {}
    labels: dict[str, tuple[str, str]] = {}
    for model, rows in memberships.items():
        for item in choice.items:
            key = model if len(choice.items) == 1 else f"{model} · {item.label}"
            cuts[key] = _cut(frames_by_model[model], [held == item.material and case_id in shared[held] for case_id, held in rows])
            labels[key] = (choice.label, model)
    return DisplayFrames(cuts, labels)


def transient_inventory(session: TransientEvaluationSession, *, refused: Mapping[str, str] = _NO_REFUSED) -> dataset_selection.DatasetInventory:
    """
    Build the non-poolable inventory of a transient session: one item per material and case role.

    The role of a case is its declared evaluation-assignment role, else the role of its artifact.
    A role the workflow refused is one unavailable role-level item with its reason.

    Raises
    ------
    ValueError
        If a refused role also holds loaded cases.

    """
    pairs = {(_entry_role(entry), entry.material_family) for entry in session.case_inventory()}
    loaded = sorted({role for role, _material in pairs}.intersection(refused))
    if loaded:
        message = f"Roles {loaded} are both refused and loaded."
        raise ValueError(message)
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
