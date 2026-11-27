"""Resolve the fixed style of every plotted series and group steady frames into panels.

Responsibilities:
  - Describe one plotted series by its model, role, materials and run (``SeriesEntity``)
  - Resolve its colour, line style, hatch and marker from the entity alone, for steady and transient evidence alike
  - Read the model family and physics flag of a frame's persisted provenance (``model_metadata``)
  - Build the entities of steady frames from their persisted provenance
  - Resolve the styles of a transient comparison once from the entity facts of all its runs (``comparison_styles``) and look a
    view's series up in them (``select_styles``)
  - Group the keys of steady comparison frames into datasets (panels) and models (lines or bars within a panel)
  - Reduce a pooled series, one frame per material, to the equal-weight mean of its materials' values (``material_mean``)

Design principles:
  - The colour identifies the model of several runs: the architecture hue, one shade per run of one architecture by sorted run key;
    else the single material, else the pooled role; a run-level view (``by_run``) takes the run colour also for a single run;
    the role is the line style or hatch
  - The marker is the material where one run spans several materials, so runs differ in every plot type and materials only in some
  - Styles are resolved once over all frames, so one model keeps one style in every panel
  - A pooled series weights its materials equally, never by their case counts; one unavailable material makes the pooled value unavailable

This module does NOT:
  - Draw axes, legends or figures
  - Choose which datasets or models a view compares
  - Define the colours themselves; ``analysis_visual_semantics`` owns them
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence

    import pandas as pd

# The entity facts of a transient series member: a record's identity or a session summary row's provenance columns of these names.
ENTITY_COLUMNS: Final = ("model_kind", "dataset_role", "material_family", "checkpoint_sha256")


@dataclass(frozen=True, slots=True)
class SeriesEntity:
    """Identify one plotted series: its model, evaluation role, materials (a tuple of names) and the run it comes from."""

    label: str
    family: str
    physics_enabled: bool
    role: str
    materials: Sequence[str]
    run_key: Hashable

    def __post_init__(self) -> None:
        """Normalise the materials to a tuple of names; a bare string would be read letter by letter."""
        if isinstance(self.materials, str):
            msg = f"Materials of {self.label!r} must be a sequence of names, not the string {self.materials!r}."
            raise TypeError(msg)
        object.__setattr__(self, "materials", tuple(str(material) for material in self.materials))


class SeriesStyles(dict[str, visual_semantics.SeriesStyle]):
    """The styles of a figure's series by label; ``run_colours`` is true when the colour is the run, not the architecture."""

    run_colours: bool = False


def _architecture(entity: SeriesEntity) -> str:
    """Return the architecture key of the entity's model: its family, ``pi-`` prefixed when physics-informed."""
    return f"pi-{entity.family}" if entity.physics_enabled else entity.family


def _single_run_style(entity: SeriesEntity) -> visual_semantics.SeriesStyle:
    """Return the style of one run's entity: its single material, else the pooled role."""
    if len(entity.materials) == 1:
        return visual_semantics.series_style(material=entity.materials[0], role=entity.role)
    return visual_semantics.series_style(role=entity.role, pooled=True)


def resolve_styles(entities: Sequence[SeriesEntity], *, by_run: bool = False) -> SeriesStyles:
    """
    Return the fixed colour, line style, hatch and marker of every entity, keyed by its label.

    The role gives the line style and the hatch. One run: the entity's single material gives the colour, else (several materials or a
    whole role) the pooled role colour; ``by_run`` colours one run as several are coloured, for a view whose series are runs. Several
    runs: the colour identifies the model, the hue of its architecture; runs of one
    architecture (physics flag included) get one shade each in sorted run-key order, so a run keeps its colour in every panel and
    runs differ in every plot type. When an architecture has more than ``visual_semantics.SHADE_LIMIT`` runs, the figure switches to
    run colours (``run_colours`` is true): every run takes its own color of ``visual_semantics.RUN_COLORS`` in sorted run-key order,
    and more runs than colors raise ``ValueError``. The marker is the material when the entities of the entity's run span several
    materials, else none.
    """
    styles = SeriesStyles()
    if not by_run and len({entity.run_key for entity in entities}) <= 1:
        styles.update((entity.label, _single_run_style(entity)) for entity in entities)
        return styles
    runs_by_architecture: dict[str, set[Hashable]] = {}
    materials_by_run: dict[Hashable, set[Sequence[str]]] = {}
    for entity in entities:
        runs_by_architecture.setdefault(_architecture(entity), set()).add(entity.run_key)
        materials_by_run.setdefault(entity.run_key, set()).add(entity.materials)
    all_runs = sorted(materials_by_run, key=repr)
    styles.run_colours = any(len(runs) > visual_semantics.SHADE_LIMIT for runs in runs_by_architecture.values())
    if styles.run_colours and len(all_runs) > len(visual_semantics.RUN_COLORS):
        msg = f"Plots tell at most {len(visual_semantics.RUN_COLORS)} runs apart by color; {len(all_runs)} were given."
        raise ValueError(msg)
    for entity in entities:
        architecture = _architecture(entity)
        runs = sorted(runs_by_architecture[architecture], key=repr)
        spans_materials = len(materials_by_run[entity.run_key]) > 1
        material = entity.materials[0] if spans_materials and len(entity.materials) == 1 else None
        style = visual_semantics.series_style(architecture=architecture, material=material, role=entity.role)
        if styles.run_colours:
            color = visual_semantics.RUN_COLORS[all_runs.index(entity.run_key)]
        else:
            color = visual_semantics.architecture_shade(architecture, runs.index(entity.run_key), len(runs))
        styles[entity.label] = replace(style, color=color)
    return styles


def model_metadata(frame: pd.DataFrame) -> tuple[str, int, bool]:
    """
    Read model family, exact trainable count, and PI flag.

    Values come only from complete provenance. Missing/non-typed parameter counts
    or physics enablement fail rather than being inferred from labels,
    configuration paths, or architecture names.
    """
    model = dataframe.require_complete_provenance(frame).get("model")
    if not isinstance(model, Mapping):
        msg = "Artifact provenance model must be a mapping."
        raise TypeError(msg)
    counts = model.get("parameter_counts")
    if not isinstance(counts, Mapping):
        msg = "Artifact provenance model.parameter_counts must be a mapping."
        raise TypeError(msg)
    family = model.get("kind")
    trainable = counts.get("trainable")
    if not isinstance(family, str) or not family or isinstance(trainable, bool) or not isinstance(trainable, int):
        msg = "Model metadata requires the model kind and an exact trainable parameter count."
        raise TypeError(msg)
    physics_enabled = model.get("physics_enabled")
    if type(physics_enabled) is not bool:
        msg = "Model metadata requires a boolean model.physics_enabled provenance value."
        raise TypeError(msg)
    return family, trainable, physics_enabled


def series_styles(datasets: Mapping[str, pd.DataFrame], pools: Iterable[Sequence[str]] = (), *, by_run: bool = False) -> SeriesStyles:
    """
    Resolve the styles of steady dataset frames, keyed by the datasets' keys.

    The inputs are the persisted provenance: run, role, selected materials and, only of several runs, the model. ``pools`` lists
    the keys of each pooled series (one frame per material, see ``panel_series``): these keys take the style of the series, that of
    all its materials together, instead of the style of their own material. ``by_run`` styles a view whose series are runs (the
    break-even): every frame takes its run's architecture colour by the shade rule, a single run too, read from its persisted model.
    """
    provenances = {label: dataframe.require_complete_provenance(frame) for label, frame in datasets.items()}
    materials = {label: tuple(provenance.get("selection", {}).get("material_families") or ()) for label, provenance in provenances.items()}
    for keys in pools:
        materials.update(dict.fromkeys(keys, tuple(dict.fromkeys(material for key in keys for material in materials[key]))))
    run_keys = {label: (provenance["run"].get("name"), provenance["run"].get("effective_config_digest")) for label, provenance in provenances.items()}
    several_runs = by_run or len(set(run_keys.values())) > 1
    entities = []
    for label, frame in datasets.items():
        family, _trainable, physics_enabled = model_metadata(frame) if several_runs else ("", 0, False)
        split = str(frame.attrs.get("dataset_role"))
        entities.append(
            SeriesEntity(
                label=label,
                family=family,
                physics_enabled=physics_enabled,
                role="id" if split == "eval" else split,
                materials=materials[label],
                run_key=run_keys[label],
            )
        )
    return resolve_styles(entities, by_run=by_run)


def _entity_facts(member: Sequence[Any], *, label: str) -> tuple[str, str, str, str]:
    """Admit one member's entity facts, ordered as ``ENTITY_COLUMNS``; each must be non-empty text."""
    facts = tuple(member)
    if len(facts) != len(ENTITY_COLUMNS) or not all(isinstance(value, str) and value for value in facts):
        msg = f"Series {label!r} requires non-empty text {', '.join(ENTITY_COLUMNS)} for every member, not {facts}."
        raise ValueError(msg)
    kind, role, material, checkpoint = facts
    return kind, role, material, checkpoint


def _style_key(kind: str, role: str, checkpoint: str, material: str | None) -> str:
    """Return the key of a run's series of one role in ``comparison_styles``: one material, or ``None`` for several."""
    return json.dumps([kind, role, checkpoint, material])


def comparison_styles(members: Iterable[Sequence[Any]]) -> SeriesStyles:
    """
    Resolve once the fixed style of every series a transient comparison can draw, from the entity facts of all its loaded runs.

    ``members`` holds the entity facts ordered as ``ENTITY_COLUMNS`` (model kind, Dataset role, material, selected checkpoint) of
    every run, role and material of the comparison. Each gets a style, and each run and role one for a series pooling several of
    its materials, which has the same style whatever materials it pools. The run is the model kind with the checkpoint. Every view
    looks its series up here (``select_styles``) instead of resolving the series it happens to hold, so a run keeps its colour,
    and the comparison its single-run or several-run rule, in a view with evidence for only some runs.
    """
    pooled: dict[tuple[str, str, str], dict[str, None]] = {}
    for member in members:
        kind, role, material, checkpoint = _entity_facts(member, label="comparison")
        pooled.setdefault((kind, role, checkpoint), {})[material] = None
    entities = []
    for (kind, role, checkpoint), materials in pooled.items():
        for shown in dict.fromkeys((tuple(materials), *((material,) for material in materials))):  # one material: one entity
            key = _style_key(kind, role, checkpoint, shown[0] if len(shown) == 1 else None)
            entities.append(SeriesEntity(key, kind, False, role, shown, (kind, checkpoint)))
    return resolve_styles(entities)


def select_styles(members: Mapping[str, Sequence[Sequence[Any]]], styles: SeriesStyles) -> SeriesStyles:
    """
    Return the comparison style of every labelled series with members, looked up by its members' entity facts (``ENTITY_COLUMNS``).

    A series is one model's: one model kind, role and checkpoint. With one material it takes that material's style, else the
    style of its run and role for several materials. A series that ``styles`` (``comparison_styles``) does not hold raises
    ``ValueError``. The result keeps the comparison's ``run_colours``.
    """
    result = SeriesStyles()
    result.run_colours = styles.run_colours
    for label, facts in members.items():
        if not facts:
            continue
        admitted = [_entity_facts(item, label=label) for item in facts]
        kinds, roles, materials, checkpoints = (tuple(dict.fromkeys(values)) for values in zip(*admitted, strict=True))
        if len(kinds) != 1 or len(roles) != 1 or len(checkpoints) != 1:
            msg = f"Series {label!r} must hold one model kind, role and checkpoint, not {kinds}, {roles} and {checkpoints}."
            raise ValueError(msg)
        key = _style_key(kinds[0], roles[0], checkpoints[0], materials[0] if len(materials) == 1 else None)
        if key not in styles:
            msg = f"Series {label!r} ({kinds[0]}, {roles[0]}, checkpoint {checkpoints[0]}, {materials}) is not part of the comparison's styles."
            raise ValueError(msg)
        result[label] = styles[key]
    return result


def panel_groups(labels: Mapping[str, tuple[str, str]]) -> dict[str, dict[str, str]]:
    """
    Group the keys of steady frames into panels: dataset label, then model label, then the key, in order of first appearance.

    ``labels`` maps every key to its (dataset label, model label); a repeated pair raises ``ValueError`` naming both keys. Look the
    frame up with ``frames[key]`` and the style with ``series_styles(frames)[key]``, so one model keeps one style in every panel.
    """
    groups: dict[str, dict[str, str]] = {}
    for key, (dataset, model) in labels.items():
        if (earlier := groups.get(dataset, {}).get(model)) is not None:
            msg = f"Frames {earlier!r} and {key!r} are both model {model!r} on dataset {dataset!r}."
            raise ValueError(msg)
        groups.setdefault(dataset, {})[model] = key
    return groups


def panel_series(labels: Mapping[str, tuple[str, str]]) -> dict[str, dict[str, tuple[str, ...]]]:
    """
    Group the keys of steady frames into panels as ``panel_groups`` does, with the keys of each series in order.

    A series is one key, or, for a pooled choice, one key per material under the same (dataset label, model label) pair. Only the
    views that pool read this; every other view takes ``panel_groups``, which refuses a repeated pair.
    """
    groups: dict[str, dict[str, list[str]]] = {}
    for key, (dataset, model) in labels.items():
        groups.setdefault(dataset, {}).setdefault(model, []).append(key)
    return {dataset: {model: tuple(keys) for model, keys in models.items()} for dataset, models in groups.items()}


def series_heads(groups: Mapping[str, Mapping[str, tuple[str, ...]]]) -> dict[str, dict[str, str]]:
    """Return the first key of every series of ``panel_series`` in the shape of ``panel_groups``, for a legend and the series style."""
    return {dataset: {model: keys[0] for model, keys in models.items()} for dataset, models in groups.items()}


def material_name(frame: pd.DataFrame) -> str:
    """Return the display name of the one material a pooled series' frame holds (``display_labels.material_display_label``)."""
    (material,) = dataframe.require_complete_provenance(frame)["selection"]["material_families"]
    return display_labels.material_display_label(str(material))


def material_mean(keys: Sequence[str], value: Callable[[str], Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the value of one series and, per key, the value of its material: the one place where materials are pooled.

    ``value`` reduces the frame of one key to its components (one number, or one per diagnostic). A series of one key is its own
    value. Several keys, one per material, give the unweighted mean over the materials, whatever their case counts; a component
    that is NaN (unavailable) in any material is NaN in the series, never a mean of the rest.
    """
    per_key = np.array([value(key) for key in keys], dtype=float)
    return per_key.mean(axis=0), per_key
