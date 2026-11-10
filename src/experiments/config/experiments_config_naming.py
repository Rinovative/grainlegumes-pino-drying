"""Derive config sources, model labels, pair keys, trial leaves and run names.

Responsibilities:
  - Derive the (task, family, name) source of an experiment or Optuna YAML from its location
  - Derive the model label from the resolved architecture and physics-loss settings
  - Derive the config name: the model label plus a token for every shared, single-stage and A0 leaf the config file authors
  - Require every config file's name to equal the derived name and refuse an authored leaf no token shows
  - Derive B/A1 pair keys (budget, rollout, learning rates, then a token per authored B/A1 leaf), Optuna trial leaves and run and parent names

Design principles:
  - Every name is a pure function of the resolved config, its source and its layout; a config name reads the file's own leaves plus the model label
  - The model label reads the merged model parameters and the physics switch, so a base change of a model parameter renames every config
  - B/A1 settings name the pair folder, never the config folder that holds A0, so editing them keeps A0 findable
  - A config YAML outside the canonical tree is an explicit ``external`` family, never guessed, and is name-checked like a maintained file
  - Readable names locate runs; exact scientific identity stays in saved metadata

This module does NOT:
  - Compose directories or check what already exists. ``common.paths`` owns layout paths
  - Resolve configs, finalize layouts or validate scientific settings. The loader owns those
  - Choose which parameters a study samples. The search-space owner does
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    authored: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)  # the file's own leaves, without ``task`` and ``base``


def _authored_leaves(path: Path) -> dict[str, Any]:
    """Return a config file's own mapping before the task base is merged, without ``task`` and ``base``."""
    return {key: value for key, value in loader.load_yaml(path).items() if key not in {"task", loader.AUTHORED_BASE_KEY}}


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
            None if isinstance(nested, Mapping) else _authored_leaves(path),  # a study file has no experiment leaves of its own
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
        declared inside the YAML; its name is checked like a maintained file's.

    Raises
    ------
    ConfigError
        If the path lies inside the canonical tree but has any other shape (for
        example ``base.yaml``, nested families or a reserved family name), a file
        in the optuna directory has no ``experiment`` block, or an external YAML declares no task.

    """
    source_path = Path(path).expanduser().resolve()
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
    if family == common.paths.OPTUNA_FAMILY and source_path.is_file() and not isinstance(loader.load_yaml(source_path).get("experiment"), Mapping):
        message = (
            f"Config {path} is no study (no experiment block); experiments live under {EXPERIMENTS_DIRNAME}/<family>/, not in the optuna directory."
        )
        raise loader.ConfigError(message)
    try:
        return ConfigSource(
            common.paths.validate_logical_name(task, label="config directory task"),
            common.paths.validate_logical_name(family, label="config family"),
            common.paths.validate_logical_name(filename.removesuffix(_CONFIG_SUFFIX), label="config file stem"),
            _authored_leaves(source_path) if family != common.paths.OPTUNA_FAMILY and source_path.is_file() else None,
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
        The architecture label (including domain padding and non-default model
        switches), prefixed ``pi-`` with a ``__fd`` (central finite differences) or
        ``__spec`` (spectral) suffix when the physics loss is enabled.

    Raises
    ------
    ConfigError
        If an enabled physics loss declares an unnamed derivative kind.

    """
    try:
        label = learning.models.naming.architecture_label(config)
    except ValueError as error:  # a boolean or non-numeric domain_padding, mode or width has no label
        message = f"model: {error}"
        raise loader.ConfigError(message) from error
    physics = config["loss"]["physics"]
    if not physics["enabled"]:
        return label
    kind = physics["derivatives"]["kind"]
    suffix = _DERIVATIVE_SUFFIXES.get(kind)
    if suffix is None:
        message = f"Physics loss derivative kind {kind!r} has no model-label suffix; expected one of {sorted(_DERIVATIVE_SUFFIXES)}."
        raise loader.ConfigError(message)
    return f"{_PHYSICS_PREFIX}{label}__{suffix}"


_PHYSICS_TERMS = ("momentum_weight", "continuity_weight", "boundary_weight")
# Leaves of one matched-plan stage by token suffix.
_STAGE_LEAVES = {
    "ep": ("max_epochs",),
    "ev": ("evaluation_interval",),
    "oev": ("ood_evaluation_interval",),
    "acc": ("gradient_accumulation_steps",),
    "cur": ("curriculum.lengths", "curriculum.milestone_fractions"),
    "smp": ("sampling.mode",),
    "rl": ("sampling.rollout_length",),
    "ws": ("sampling.window_stride",),
    "wo": ("sampling.window_offset",),
    "ts": ("train_seconds",),
    "lrs": ("lr_scale",),
}


def _stage_tokens(*stages: str) -> dict[str, tuple[str, ...]]:
    """Return the token of every stage leaf, prefixed by its stage."""
    return {f"{stage}{token}": tuple(f"training.{stage}.{leaf}" for leaf in leaves) for stage in stages for token, leaves in _STAGE_LEAVES.items()}


# The pair key's fixed part already shows these B/A1 leaves.
_PAIR_SHOWN = ("training.b.train_seconds", "training.b.sampling.rollout_length", "training.b.lr_scale", "training.a1.lr_scale")
# The tables of name tokens, in name order: token -> the authored leaves whose values it shows. The config name takes the
# shared, single-stage and A0 tokens; the pair folder takes the B/A1 tokens, so editing a B/A1 setting never renames the config that holds A0.
_PAIR_TOKENS: dict[str, tuple[str, ...]] = {
    token: paths
    for token, paths in {"cev": ("training.continuation_evaluations",), **_stage_tokens("b", "a1")}.items()
    if not set(paths) & set(_PAIR_SHOWN)
}
_TOKENS: dict[str, tuple[str, ...]] = {
    "crop": ("loss.physics.interior_crop",),
    "w": tuple(f"loss.physics.{term}.target" for term in _PHYSICS_TERMS),
    "wu": tuple(f"loss.physics.{term}.warmup.epochs" for term in _PHYSICS_TERMS),
    "loss": ("loss.data.kind",),
    "space": ("loss.data.space",),
    "lw": ("loss.data.weight",),
    "huber": ("loss.data.beta",),
    "cw": ("loss.data.channel_weights",),
    "aux": ("loss.data.state_aux_weight",),
    "stride": ("data.spatial_stride",),
    "bs": ("data.batch_size",),
    "sub": ("data.train_subset",),
    "lr": ("optimizer.lr",),
    "wd": ("optimizer.weight_decay",),
    "betas": ("optimizer.betas",),
    "eps": ("optimizer.second_moment_floor",),
    "sch": ("scheduler",),
    "schf": ("scheduler.factor",),
    "schp": ("scheduler.patience",),
    "schmin": ("scheduler.min_lr",),
    "det": ("run.deterministic",),
    "profile": ("input_profile",),
    "scale": ("scaling.mode",),
    "tcond": ("temporal.temporal_conditioning.kind",),
    "obj": ("evaluation.objective.id",),
    "ep": ("training.epochs",),
    "ev": ("training.evaluation_interval",),
    "oev": ("training.ood_evaluation_interval",),
    "acc": ("training.gradient_accumulation_steps",),
    **_stage_tokens("a0"),
}
# Authored leaves no token shows; each reason is why the experiment's content does not depend on it.
_UNNAMED = (
    "model",  # the model label shows the architecture and every deviating parameter
    "loss.physics.enabled",  # the pi- prefix
    "loss.physics.derivatives",  # the __fd/__spec suffix
    *(f"loss.physics.{term}.warmup.kind" for term in _PHYSICS_TERMS),  # one admitted kind
    "optimizer.kind",  # one admitted kind
    "scheduler.kind",  # one admitted kind
    "training.mode",  # the matched plan's one admitted protocol
    "training.a1.match_to",  # one admitted value
    "training.stage_schedule",  # stage markers derived from the plan or the study mode
    "training.stage",
    "training.comparison_arm",
    "training.teacher_handoff",
    *_PAIR_SHOWN,  # the pair key's fixed part
    "run.seed",  # the run directory carries seed-N
    "run.revision",  # the run directory carries rev-<label>
    "run.device",  # hardware
    "tracking",  # observation only
    "data.train_dataset",  # the run path carries the Dataset
    "data.material_ood_datasets",  # evaluation roles of the same Dataset selection
    "data.num_workers",  # loader throughput only
    "data.pin_memory",
    "data.persistent_workers",
)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Return the dotted leaf paths of a mapping; lists and null are leaves, an empty mapping states nothing."""
    if not isinstance(node, Mapping):
        return {prefix: node}
    return {path: value for key, child in node.items() for path, value in _flatten(child, f"{prefix}.{key}" if prefix else str(key)).items()}


def _token_text(token: str, values: list[Any]) -> str:
    """Render leaf values as one name value: a curriculum shows its fractions only when not equally spaced, equal values show once."""
    if token.endswith("cur"):
        lengths, fractions = values
        text = "-".join(_value_token(length) for length in lengths)
        return text + ("" if list(fractions) == [i / len(lengths) for i in range(len(lengths))] else "-at-" + "-".join(map(_value_token, fractions)))
    parts = [_value_token(item) for value in values for item in (value if isinstance(value, list) else [value])]
    return parts[0] if len(values) > 1 and len(set(parts)) == 1 else "-".join(parts)


def _token_texts(tokens: Mapping[str, tuple[str, ...]], leaves: Mapping[str, Any]) -> list[str]:
    """Return ``<token>-<value>`` for every token one of whose leaves the file authors; a multi-leaf token needs all of its leaves."""
    texts = []
    for token, paths in tokens.items():
        if any(path in leaves for path in paths):
            missing = next((path for path in paths if path not in leaves), None)
            if missing is not None:
                message = f"Config sets part of name token {token!r} but not {missing!r}: set all of {', '.join(paths)} together."
                raise loader.ConfigError(message)
            texts.append(f"{token}-{_token_text(token, [leaves[path] for path in paths])}")
    return texts


def complete_token_groups(authored: Mapping[str, Any], effective: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``authored`` plus every leaf of each multi-leaf name token it partly sets, taken from the ``effective`` config."""
    completed = copy.deepcopy(dict(authored))
    leaves = _flatten(authored)
    for paths in _TOKENS.values():
        if len(paths) > 1 and any(path in leaves for path in paths):
            for path in paths:
                *parents, leaf = path.split(".")
                node, source = completed, effective
                for key in parents:
                    node, source = node.setdefault(key, {}), source[key]
                node[leaf] = copy.deepcopy(source[leaf])
    return completed


def config_name(config: Mapping[str, Any], authored: Mapping[str, Any]) -> str:
    """Return the model label followed by ``__<token>-<value>`` for every leaf the config file authors.

    ``config`` supplies the model label; ``authored`` is the file's own mapping (before the task base is
    merged, without ``task`` and ``base``). The tokens are the keys of ``_TOKENS`` and ``configs/README.md``
    lists them. B/A1 leaves of a matched plan are not part of the name: ``pair_key`` shows them.

    Raises
    ------
    ConfigError
        If the file authors a leaf that no token shows and that ``_UNNAMED`` does not exclude, or part of a multi-leaf token.

    """
    leaves = _flatten(authored)
    covered = {path for paths in (*_TOKENS.values(), *_PAIR_TOKENS.values()) for path in paths}
    for path in leaves:
        if path not in covered and not any(path == name or path.startswith(f"{name}.") for name in _UNNAMED):
            message = f"Config sets {path!r}, which no config-name token shows: add a name token for {path} in experiments_config_naming."
            raise loader.ConfigError(message)
    return "__".join([model_label(config), *_token_texts(_TOKENS, leaves)])


def validate_config_name(source: ConfigSource, config: Mapping[str, Any]) -> None:
    """Require a config file's name to equal the name its authored leaves derive.

    Sources without their file's leaves (Optuna studies, which sample the architecture, and in-memory
    sources) are exempt. An external YAML is checked like a maintained file.

    Raises
    ------
    ConfigError
        If the name differs from :func:`config_name`.

    """
    if source.authored is None:
        return
    expected = config_name(config, source.authored)
    if source.name != expected:
        message = (
            f"Config name {source.name!r} ({source.family}/{source.name}{_CONFIG_SUFFIX}) must be {expected!r}: the model label, "
            "then one token per setting the file sets (configs/README.md). Rename the file."
        )
        raise loader.ConfigError(message)


def pair_key(b_config: Mapping[str, Any], a1_config: Mapping[str, Any], authored: Mapping[str, Any] | None) -> str:
    """Return the directory key of one matched B/A1 pair.

    Parameters
    ----------
    b_config, a1_config : Mapping[str, Any]
        Resolved B and A1 children of one matched plan. B carries the measured
        ``train_seconds`` budget and the rollout window; both carry their ``lr_scale``.
    authored : Mapping[str, Any] | None
        The plan file's own mapping (``ConfigSource.authored``); ``None`` for a source without a file.

    Returns
    -------
    str
        ``b-a1__budget-<seconds>s__rollout-<k>__lr-<x>``; unequal learning-rate scales
        read ``lr-b-<x>__lr-a1-<y>``. One ``__<token>-<value>`` follows for every other B/A1 leaf
        the file authors (``_PAIR_TOKENS``), so the pair folder, not the config folder holding A0, shows them.

    Raises
    ------
    ConfigError
        If the file sets part of a multi-leaf token or the key exceeds one path component.

    """
    protocol = {"b": b_config["training"]["matched_a1_b"], "a1": a1_config["training"]["matched_a1_b"]}
    if protocol["b"]["role"] != "b" or protocol["a1"]["role"] != "a1":
        message = "A pair key requires the resolved B child first and the resolved A1 child second."
        raise ValueError(message)
    number = common.names.number_token
    scale_b, scale_a1 = protocol["b"]["lr_scale"], protocol["a1"]["lr_scale"]
    scales = number(scale_b) if scale_b == scale_a1 else f"b-{number(scale_b)}__lr-a1-{number(scale_a1)}"
    rollout = b_config.get("temporal", {}).get("sampling", {}).get("rollout_length", 1)
    fixed = f"b-a1__budget-{number(protocol['b']['train_seconds'])}s__rollout-{number(rollout)}__lr-{scales}"
    key = "__".join([fixed, *_token_texts(_PAIR_TOKENS, _flatten(authored or {}))])
    try:
        return common.paths.validate_logical_name(key, label="pair key")
    except ValueError as error:
        raise loader.ConfigError(str(error)) from error


def _value_token(value: object) -> str:
    """Render a name value; a non-integral float keeps three significant digits, an integral one stays exact (the digest decides identity)."""
    if isinstance(value, float) and not value.is_integer():
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
