"""Derive config sources, model labels, pair keys, trial leaves and run names.

Responsibilities:
  - Derive the (task, family, name) source of an experiment or Optuna YAML from its location
  - Derive the model label from the resolved architecture and physics-loss settings
  - Require experiment config names to equal or extend their model label
  - Derive B/A1 pair keys, Optuna trial leaves and run and parent names from a saved layout

Design principles:
  - Every name is a pure function of the resolved config, its source and its layout
  - A config YAML outside the canonical tree is an explicit ``external`` family, never guessed
  - Readable names locate runs; exact scientific identity stays in saved metadata

This module does NOT:
  - Compose directories or check what already exists. ``common.paths`` owns layout paths
  - Resolve configs, finalize layouts or validate scientific settings. The loader owns those
  - Choose which parameters a study samples. The search-space owner does
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src import common, learning

from . import experiments_config_loader as loader

CONFIG_TREE = ("configs", "03_learning")
EXPERIMENTS_DIRNAME = "experiments"
EXTERNAL_FAMILY = "external"
_PHYSICS_PREFIX = "pi-"
_DERIVATIVE_SUFFIXES = {"physical": "fd", "spectral": "spec"}
_CONFIG_SUFFIX = ".yaml"
_EXPERIMENT_SHAPE_PARTS = 4
_OPTUNA_SHAPE_PARTS = 3
_TRIAL_NUMBER_DIGITS = 4
_VALUE_SIGNIFICANT_FORMAT = ".3g"


@dataclass(frozen=True, slots=True)
class ConfigSource:
    """Identify where a config comes from: its task, family folder and file stem."""

    task: str
    family: str
    name: str


def _external_source(path: Path) -> ConfigSource:
    """Admit a YAML outside the canonical config tree as the explicit external family."""
    if not path.is_file():
        message = f"Config file not found: {path}"
        raise FileNotFoundError(message)
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    mapping = payload if isinstance(payload, Mapping) else {}
    nested = mapping.get("experiment")
    task = mapping.get("task", nested.get("task") if isinstance(nested, Mapping) else None)
    if not isinstance(task, str):
        message = f"Config {path} lies outside configs/03_learning and must declare a task."
        raise loader.ConfigError(message)
    try:
        return ConfigSource(
            common.paths.validate_logical_name(task, label="config task"),
            EXTERNAL_FAMILY,
            common.paths.validate_logical_name(path.stem, label="config file stem"),
        )
    except ValueError as error:
        raise loader.ConfigError(str(error)) from error


def config_source_from_path(path: Path | str) -> ConfigSource:
    """Derive a config's (task, family, name) from its location.

    Parameters
    ----------
    path : Path | str
        YAML path. Canonical shapes below ``configs/03_learning`` are
        ``<task>/experiments/<family>/<name>.yaml`` and ``<task>/optuna/<name>.yaml``.

    Returns
    -------
    ConfigSource
        The canonical task, family (``optuna`` for studies) and file stem. A YAML
        outside ``configs/03_learning`` is the ``external`` family with the task
        declared inside the YAML, so test-owned or ad-hoc configs keep working.

    Raises
    ------
    ConfigError
        If the path lies inside the canonical tree but has any other shape (for
        example ``base.yaml``, nested families or a reserved family name), or an
        external YAML declares no task.

    """
    source_path = Path(path).expanduser()
    parts = source_path.parts
    markers = [index for index in range(len(parts)) if tuple(parts[index : index + len(CONFIG_TREE)]) == CONFIG_TREE]
    if not markers:
        return _external_source(source_path)
    if len(markers) != 1:
        message = f"Config path contains an ambiguous repeated configs/03_learning marker: {path}"
        raise loader.ConfigError(message)
    relative = parts[markers[0] + len(CONFIG_TREE) :]
    reserved = False
    if len(relative) == _EXPERIMENT_SHAPE_PARTS and relative[1] == EXPERIMENTS_DIRNAME:
        task, family, filename = relative[0], relative[2], relative[3]
        reserved = family in {common.paths.OPTUNA_FAMILY, EXTERNAL_FAMILY}
    elif len(relative) == _OPTUNA_SHAPE_PARTS and relative[1] == common.paths.OPTUNA_FAMILY:
        task, family, filename = relative[0], common.paths.OPTUNA_FAMILY, relative[2]
    else:
        task = family = filename = ""
    if not filename.endswith(_CONFIG_SUFFIX) or reserved:
        message = (
            f"Config {path} must be configs/03_learning/<task>/{EXPERIMENTS_DIRNAME}/<family>/<name>.yaml "
            f"(family not {common.paths.OPTUNA_FAMILY!r} or {EXTERNAL_FAMILY!r}) or "
            f"configs/03_learning/<task>/{common.paths.OPTUNA_FAMILY}/<name>.yaml."
        )
        raise loader.ConfigError(message)
    try:
        return ConfigSource(
            common.paths.validate_logical_name(task, label="config directory task"),
            common.paths.validate_logical_name(family, label="config family"),
            common.paths.validate_logical_name(filename.removesuffix(_CONFIG_SUFFIX), label="config file stem"),
        )
    except ValueError as error:
        raise loader.ConfigError(str(error)) from error


def model_family(config: Mapping[str, Any]) -> str:
    """Return the network and physics-loss family used as the W&B model tag: ``pi-<kind>`` or ``<kind>``."""
    kind = config["model"]["kind"]
    return f"{_PHYSICS_PREFIX}{kind}" if config["loss"]["physics"]["enabled"] else str(kind)


def model_label(config: Mapping[str, Any]) -> str:
    """Return the run-navigation label of the resolved architecture and physics loss.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved experiment with ``model`` and ``loss.physics``.

    Returns
    -------
    str
        The core architecture label, prefixed ``pi-`` with a ``__fd`` (central finite
        differences) or ``__spec`` (spectral) suffix when the physics loss is enabled.
        Non-default model switches are not part of it.

    Raises
    ------
    ConfigError
        If an enabled physics loss declares an unnamed derivative kind.

    """
    core = learning.models.naming.core_architecture_label(config)
    physics = config["loss"]["physics"]
    if not physics["enabled"]:
        return core
    kind = physics["derivatives"]["kind"]
    suffix = _DERIVATIVE_SUFFIXES.get(kind)
    if suffix is None:
        message = f"Physics loss derivative kind {kind!r} has no model-label suffix; expected one of {sorted(_DERIVATIVE_SUFFIXES)}."
        raise loader.ConfigError(message)
    return f"{_PHYSICS_PREFIX}{core}__{suffix}"


def validate_config_name(source: ConfigSource, config: Mapping[str, Any]) -> None:
    """Require an experiment config name to equal or extend its model label.

    Optuna studies are exempt: their trials sample the architecture, so the study
    name cannot encode it.

    Raises
    ------
    ConfigError
        If the name is neither the model label nor starts with ``<model label>__``.

    """
    if source.family == common.paths.OPTUNA_FAMILY:
        return
    label = model_label(config)
    if source.name != label and not source.name.startswith(f"{label}__"):
        message = (
            f"Config name {source.name!r} ({source.family}/{source.name}{_CONFIG_SUFFIX}) must equal its model label {label!r} "
            f"or start with {label + '__'!r}. Rename the file to match the architecture and physics loss it configures."
        )
        raise loader.ConfigError(message)


def pair_key(b_config: Mapping[str, Any], a1_config: Mapping[str, Any]) -> str:
    """Return the directory key of one matched B/A1 pair.

    Parameters
    ----------
    b_config, a1_config : Mapping[str, Any]
        Resolved B and A1 children of one matched plan. B carries the measured
        ``train_seconds`` budget and the rollout window; both carry their ``lr_scale``.

    Returns
    -------
    str
        ``b-a1__budget-<seconds>s__rollout-<k>__lr-<x>``; unequal learning-rate scales
        read ``lr-b-<x>__lr-a1-<y>``.

    """
    protocol = {"b": b_config["training"]["matched_a1_b"], "a1": a1_config["training"]["matched_a1_b"]}
    if protocol["b"]["role"] != "b" or protocol["a1"]["role"] != "a1":
        message = "A pair key requires the resolved B child first and the resolved A1 child second."
        raise ValueError(message)
    number = common.names.number_token
    scale_b, scale_a1 = protocol["b"]["lr_scale"], protocol["a1"]["lr_scale"]
    scales = number(scale_b) if scale_b == scale_a1 else f"b-{number(scale_b)}__lr-a1-{number(scale_a1)}"
    rollout = b_config.get("temporal", {}).get("sampling", {}).get("rollout_length", 1)
    return f"b-a1__budget-{number(protocol['b']['train_seconds'])}s__rollout-{number(rollout)}__lr-{scales}"


def _value_token(value: object) -> str:
    """Render a sampled search value; floats keep three significant digits."""
    if isinstance(value, float):
        return common.names.number_token(float(format(value, _VALUE_SIGNIFICANT_FORMAT)))
    return common.names.text_token(value)


def trial_leaf(trial_number: int, config: Mapping[str, Any], sampled_params: Mapping[str, Any]) -> str:
    """Return an Optuna trial directory name from its number, model label and sampled values.

    Parameters
    ----------
    trial_number : int
        Native zero-based Optuna trial number, zero-padded to four digits.
    config : Mapping[str, Any]
        The trial's resolved config, whose model label names the sampled architecture.
    sampled_params : Mapping[str, Any]
        Sampled search-space parameters that are not ``model.params`` values, by
        parameter name; the model label already carries the architecture.

    Returns
    -------
    str
        ``<NNNN>__<model label>__<name>-<value>...`` with tokens sorted by parameter name.

    """
    if type(trial_number) is not int or trial_number < 0:
        message = f"Trial number must be a non-negative integer, got {trial_number!r}."
        raise ValueError(message)
    values = [f"{name}-{_value_token(value)}" for name, value in sorted(sampled_params.items())]
    return "__".join([f"{trial_number:0{_TRIAL_NUMBER_DIGITS}d}", model_label(config), *values])


def run_name(layout: Mapping[str, Any]) -> str:
    """Return the run name of a saved layout.

    Steady runs are ``<config>__seed-<n>[__rev-<r>]``; transient stages are
    ``<config>__seed-<n>__a0ep-<epochs>[__rev-<r>]`` and append the pair (B/A1) and
    the stage; trials are ``<study>__trial-<NNNN>`` (their directory keeps the
    descriptive trial leaf).
    """
    if layout["trial"] is not None:
        number = str(layout["trial"]).split("__", 1)[0]
        return f"{layout['config_name']}__trial-{number}"
    parts = [layout["config_name"], common.paths.seed_group_name(layout["seed"], layout["revision"], a0_epochs=layout["budget_epochs"])]
    if layout["pair"] is not None:
        parts.append(layout["pair"])
    if layout["stage"] != "single":
        parts.append(layout["stage"])
    return "__".join(parts)


def parent_label(layout: Mapping[str, Any]) -> str:
    """Return the label shared by every stage of one seed group: family, config and seed group.

    A transient seed group carries its A0 epoch budget, so another budget is another parent.
    A trial's group is its study directory name followed by the training seed.
    """
    if layout["trial"] is not None:
        group = common.paths.study_group_name(layout["config_name"], layout["budget_epochs"], layout["revision"])
        return f"{layout['family']}__{group}__seed-{layout['seed']}"
    group = common.paths.seed_group_name(layout["seed"], layout["revision"], a0_epochs=layout["budget_epochs"])
    return f"{layout['family']}__{layout['config_name']}__{group}"
