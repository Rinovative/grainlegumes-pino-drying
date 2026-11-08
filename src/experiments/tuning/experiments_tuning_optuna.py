"""Run reusable Optuna studies and trial objectives.

Responsibilities:
  - Load strict study YAML with an inline resolved experiment and search space
  - Bind reusable studies to scientific semantic signatures before optimization
  - Admit transient studies only as the single-stage ``stage_a_only`` A0 protocol and reject matched A0/B/A1 plans
  - Run fresh numbered trials with actual-completed-epoch reporting and pruning
  - Bind the same manifest identity and payload digests of either task, read by the one manifest reader, before the dry-run
    description or the study signature
  - Supply the trial status, reporting progress, failure context and best evidence of every terminal trial summary
  - Record the Slurm source binding of every invocation that creates or reopens a study, before its first trial
  - Classify non-finite, OOM, recoverable, interrupted, duplicate and unexpected outcomes explicitly
  - Declare the per-task HPO code projections: its own modules plus the training projection of the study's task

Design principles:
  - Reopening preserves history but each invocation allocates only additional fresh trials
  - Both tasks run their trials through the one driver and the shared prepared-run lifecycle
  - Embedded requests use the generic experiment structure. Wrappers own study/search policy
  - Wrapper objective and direction derive from the embedded resolved experiment objective
  - Device, output, tracking, invocation count and Dataset locations remain outside scientific identity
  - Local trial state is authoritative. W&B is an optional post-publication observer
  - Optuna imports stay lazy for help and dry-run validation

This module does NOT:
  - Parse search-space schemas. ``experiments.tuning.search_space`` owns admission
  - Execute training epochs or publish run state. ``experiments.run`` owns the run lifecycle
  - Clean repository or run state
"""

from __future__ import annotations

import copy
import functools
import gc
import importlib
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from src import common, datasets, domain, experiments, learning
from src.experiments.config import experiments_config_naming as config_naming
from src.experiments.config import experiments_config_transient_plan as transient_plan

from . import experiments_tuning_search_space as search_space

STUDY_SIGNATURE_SCHEMA_VERSION = 1
TRIAL_LIFECYCLE_SCHEMA_VERSION = 1
_STUDY_SIGNATURE_ATTR = "semantic_signature"
_STUDY_SIGNATURE_PAYLOAD_ATTR = "semantic_signature_payload"
_STUDY_SIGNATURE_SCHEMA_ATTR = "semantic_signature_schema_version"
_RESOLVED_OBJECTIVE_ATTR = "resolved_objective"
_SAMPLER_METADATA_ATTR = "sampler_metadata"
_NUMERICAL_ENVIRONMENT_ATTR = "numerical_environment"
_STUDY_SUMMARY_FILENAME = "study_summary.json"
_PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS = 20
_STUDY_ROLES = frozenset({"production"})
_MODEL_PARAMS_PREFIX = "model.params."
_OOM_PRUNED_MESSAGE = "Trial pruned after allocator out-of-memory failure"
OPTUNA_ROOT_KEYS = frozenset({"study", "experiment", "search_space"})
_TRIAL_OUTCOMES = (
    "completed",
    "pruned",
    "nonfinite_pruned",
    "oom_pruned",
    "recoverable_failed",
    "failed",
    "interrupted",
    "duplicate",
    "duplicate_pruned",
)
_HPO_FILES = (
    "experiments/tuning/experiments_tuning_optuna.py",
    "experiments/tuning/experiments_tuning_search_space.py",
)
_HPO_EXEMPTIONS = {"datasets/dataset_packages.py": "CLI module; Dataset administration does not act on study science"}
HPO_PROJECTIONS = {
    "steady_flow": common.code_identity.CodeProjection(
        name="hpo_steady_flow",
        files=_HPO_FILES,
        exemptions={
            **_HPO_EXEMPTIONS,
            "datasets/packages/dataset_packages_trajectory.py": "transient package index summary, reached through the transient study branch",
        },
        includes=(learning.training.checkpoint.TRAINING_PROJECTIONS["steady_flow"],),
    ),
    "transient_drying": common.code_identity.CodeProjection(
        name="hpo_transient_drying",
        files=_HPO_FILES,
        exemptions=_HPO_EXEMPTIONS,
        includes=(learning.training.checkpoint.TRAINING_PROJECTIONS["transient_drying"],),
    ),
}


class RecoverableTrialError(RuntimeError):
    """
    Represent an explicitly recoverable trial-local execution failure.

    Only this project exception is supplied to ``study.optimize(catch=...)``.
    Programming, identity, storage, tracking, and arbitrary runtime errors remain
    fatal to the invocation.
    """


class NonFiniteTrialError(FloatingPointError):
    """
    Represent a non-finite objective produced after trial execution starts.

    The trial lifecycle classifies this separately from generic failure and
    converts it to pruning only after publishing truthful local terminal state.
    """


class TrialProtocol(search_space.TrialLike, Protocol):
    """Minimal Optuna trial surface used by the objective: the search-space suggestions plus lifecycle calls."""

    number: int

    def report(self, value: float, step: int) -> None:
        """Report an intermediate metric value."""

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a serializable trial user attribute."""

    def should_prune(self) -> bool:
        """Return whether Optuna wants to prune this trial."""
        ...


@dataclass(frozen=True)
class OptunaStudyConfig:
    """
    Resolved Optuna study configuration.

    Parameters
    ----------
    path : Path
        Source Optuna YAML path
    study : dict[str, Any]
        Study-level settings such as name, objective, derived direction, and n_trials
    base_experiment : dict[str, Any]
        Inline base experiment block from the Optuna YAML
    base_config : dict[str, Any]
        Base experiment resolved through config defaults
    search_space : tuple[search_space.SearchSpaceParameter, ...]
        Parsed search-space parameters.
    dataset_identities : dict[str, Any] | None
        Validated physical Dataset-package evidence bound before description or
        execution. Loading remains storage-independent and leaves this unset.

    Notes
    -----
    The dataclass is frozen, but nested mappings are treated as owned resolved
    values and are defensively copied by runtime-override and trial preparation
    boundaries before mutation.

    """

    path: Path
    study: dict[str, Any]
    base_experiment: dict[str, Any]
    base_config: dict[str, Any]
    search_space: tuple[search_space.SearchSpaceParameter, ...]
    dataset_identities: dict[str, Any] | None = None


@dataclass(frozen=True)
class _OptunaStudyPaths:
    """Pure resolved study paths shared by dry-run planning and execution."""

    study_dir: Path
    trial_root: Path
    storage: str
    local_storage_path: Path | None


@dataclass
class OptunaEpochReporter:
    """Report the exact held-out objective only at genuine completed-epoch events."""

    trial: TrialProtocol
    objective_id: str
    direction: str
    evaluation_interval: int = 1
    target_epoch: int | None = None
    best_value: float | None = None
    best_epoch: int | None = None
    last_reported_epoch: int | None = None
    last_reported_objective: float | None = None
    last_global_step: int | None = None
    last_metrics: dict[str, float] | None = None

    def _expected_epoch(self) -> int:
        """Return the next interval-or-terminal completed epoch."""
        interval = _require_exact_int(
            self.evaluation_interval,
            label="Optuna reporter evaluation_interval",
            minimum=1,
        )
        if self.target_epoch is not None:
            target = _require_exact_int(
                self.target_epoch,
                label="Optuna reporter target_epoch",
                minimum=1,
            )
            if self.last_reported_epoch is None:
                return min(interval, target)
            return min(self.last_reported_epoch + interval, target)
        if self.last_reported_epoch is None:
            return interval
        return self.last_reported_epoch + interval

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Publish one actual ID evaluation at its completed epoch and ask the pruner once."""
        if type(epoch) is not int or epoch <= 0:
            msg = f"Completed Optuna epoch must be a positive integer, got {epoch!r}."
            raise TypeError(msg)
        expected_epoch = self._expected_epoch()
        if epoch != expected_epoch:
            msg = f"Optuna reports must follow interval-or-terminal completed epochs. Expected {expected_epoch}, received {epoch}."
            raise ValueError(msg)
        metric_key = f"id/{self.objective_id}"
        if metric_key not in metrics:
            msg = f"Held-out Optuna objective {metric_key!r} is missing at completed epoch {epoch}."
            raise KeyError(msg)
        value = float(metrics[metric_key])
        if not math.isfinite(value):
            self.trial.set_user_attr("nonfinite_epoch", epoch)
            msg = f"Non-finite Optuna objective {self.objective_id} at epoch {epoch}: {value}"
            raise NonFiniteTrialError(msg)

        if self.best_value is None or _is_better(value, self.best_value, self.direction):
            self.best_value = value
            self.best_epoch = epoch

        raw_global_step = metrics.get("global_step")
        if raw_global_step is not None and math.isfinite(float(raw_global_step)):
            self.last_global_step = int(raw_global_step)
        self.trial.report(value, step=epoch)
        self.trial.set_user_attr("last_reported_epoch", epoch)
        self.trial.set_user_attr("last_reported_objective", value)
        if self.last_global_step is not None:
            self.trial.set_user_attr("last_global_step", self.last_global_step)
        self.last_reported_epoch = epoch
        self.last_reported_objective = value
        self.last_metrics = dict(metrics)
        if self.trial.should_prune():
            msg = f"Pruned by Optuna at completed epoch {epoch}"
            raise _trial_pruned_error()(msg)


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Validate and return a mapping value."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping, got: {type(value).__name__}"
        raise TypeError(msg)
    return cast("Mapping[str, Any]", value)


def _trial_pruned_error() -> type[Exception]:
    """Return Optuna's TrialPruned exception class using a lazy import."""
    exceptions = importlib.import_module("optuna.exceptions")
    return cast("type[Exception]", exceptions.TrialPruned)


def _optuna_module() -> Any:
    """Import Optuna lazily for study creation and CLI help friendliness."""
    return importlib.import_module("optuna")


def _is_better(value: float, best: float, direction: str) -> bool:
    """Return whether value improves best under the resolved direction."""
    if direction == "maximize":
        return value > best
    if direction == "minimize":
        return value < best
    msg = f"Unknown objective direction {direction!r}."
    raise ValueError(msg)


def _require_nonempty_string(value: Any, *, label: str) -> str:
    """Return a non-empty string without coercing another scalar type."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{label} must be a non-empty string, got: {value!r}"
        raise TypeError(msg)
    return value


def _normalise_study_role(value: Any) -> str | None:
    """Return the canonical optional study role after strict allowlist validation."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"study.role must be null or a string, got: {value!r}"
        raise TypeError(msg)
    if not value.strip():
        return None
    if value not in _STUDY_ROLES:
        msg = f"study.role must be null or one of {sorted(_STUDY_ROLES)}, got {value!r}."
        raise ValueError(msg)
    return value


def _require_exact_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    """Return one exact integer with an optional inclusive minimum."""
    if type(value) is not int:
        msg = f"{label} must be an integer, got: {value!r}"
        raise TypeError(msg)
    if minimum is not None and value < minimum:
        msg = f"{label} must be at least {minimum}, got: {value}"
        raise ValueError(msg)
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    """Return one exact boolean without truthiness coercion."""
    if type(value) is not bool:
        msg = f"{label} must be a boolean, got: {value!r}"
        raise TypeError(msg)
    return value


def _normalise_sampler_config(value: Any) -> dict[str, Any]:
    """
    Validate and isolate one exact semantic sampler mapping.

    Only ``random`` and ``tpe`` are accepted. Multivariate mode belongs only to
    TPE and must be an exact boolean. String shorthand, unknown keys, and scalar
    coercion are deliberately rejected.
    """
    if isinstance(value, str):
        msg = "study.sampler must be a mapping with a semantic kind. String shorthand is unsupported."
        raise TypeError(msg)
    sampler = dict(copy.deepcopy(_as_mapping(value, label="study.sampler")))
    allowed = {"kind", "multivariate"}
    unknown = sorted(set(sampler).difference(allowed))
    if unknown:
        msg = f"study.sampler contains unknown key(s): {unknown}."
        raise ValueError(msg)
    if "kind" not in sampler:
        msg = "study.sampler.kind is required"
        raise KeyError(msg)
    kind = _require_nonempty_string(sampler["kind"], label="study.sampler.kind")
    if kind not in {"random", "tpe"}:
        msg = f"Unsupported Optuna sampler: {kind!r}"
        raise ValueError(msg)
    if "multivariate" in sampler:
        sampler["multivariate"] = _require_bool(
            sampler["multivariate"],
            label="study.sampler.multivariate",
        )
        if kind != "tpe":
            msg = "study.sampler.multivariate is only valid for the tpe sampler"
            raise ValueError(msg)
    sampler["kind"] = kind
    return sampler


def _normalise_pruner_config(value: Any) -> dict[str, Any]:
    """
    Validate and isolate one exact semantic pruner mapping.

    Median-pruner cadence receives explicit defaults and exact non-negative or
    positive integers. The ``none`` pruner rejects tuning-only keys rather than
    silently ignoring them.
    """
    if isinstance(value, str):
        msg = "study.pruner must be a mapping with a semantic kind. String shorthand is unsupported."
        raise TypeError(msg)
    pruner = dict(copy.deepcopy(_as_mapping(value, label="study.pruner")))
    allowed = {"kind", "n_startup_trials", "n_warmup_steps", "interval_steps"}
    unknown = sorted(set(pruner).difference(allowed))
    if unknown:
        msg = f"study.pruner contains unknown key(s): {unknown}."
        raise ValueError(msg)
    if "kind" not in pruner:
        msg = "study.pruner.kind is required"
        raise KeyError(msg)
    kind = _require_nonempty_string(pruner["kind"], label="study.pruner.kind")
    if kind not in {"median", "none"}:
        msg = f"Unsupported Optuna pruner: {kind!r}"
        raise ValueError(msg)
    tuning_keys = {"n_startup_trials", "n_warmup_steps", "interval_steps"}
    invalid_keys = sorted(tuning_keys.intersection(pruner)) if kind == "none" else []
    if invalid_keys:
        msg = f"study.pruner contains key(s) invalid for none: {invalid_keys}."
        raise ValueError(msg)
    if kind == "median":
        pruner.setdefault("n_startup_trials", 5)
        pruner.setdefault("n_warmup_steps", 25)
        pruner.setdefault("interval_steps", 5)
    for key in ("n_startup_trials", "n_warmup_steps"):
        if key in pruner:
            pruner[key] = _require_exact_int(
                pruner[key],
                label=f"study.pruner.{key}",
                minimum=0,
            )
    if "interval_steps" in pruner:
        pruner["interval_steps"] = _require_exact_int(
            pruner["interval_steps"],
            label="study.pruner.interval_steps",
            minimum=1,
        )
    pruner["kind"] = kind
    return pruner


def _validate_study_settings(study: Mapping[str, Any]) -> None:
    """
    Revalidate the complete normalized study mapping without scalar coercion.

    Name, objective, direction, seed, additional-trial count, sampler, pruner,
    and optional storage are checked with an exact closed schema. No file,
    database, Optuna import, or runtime allocation occurs.
    """
    allowed = {
        "name",
        "mode",
        "role",
        "objective",
        "direction",
        "seed",
        "n_trials",
        "sampler",
        "pruner",
        "storage",
    }
    unknown = sorted(set(study).difference(allowed))
    if unknown:
        msg = f"Resolved study contains unknown key(s): {unknown}."
        raise ValueError(msg)
    common.paths.validate_logical_name(
        _require_nonempty_string(study.get("name"), label="study.name"),
        label="study.name",
    )
    _normalise_study_role(study.get("role"))
    if study.get("mode") not in {"standard", "stage_a_only"}:
        msg = "study.mode must be 'standard' or 'stage_a_only'."
        raise ValueError(msg)
    _require_nonempty_string(study.get("objective"), label="study.objective")
    direction = _require_nonempty_string(study.get("direction"), label="study.direction")
    if direction not in {"minimize", "maximize"}:
        msg = f"study.direction must be minimize or maximize, got {direction!r}."
        raise ValueError(msg)
    _require_exact_int(study.get("seed"), label="study.seed")
    _require_exact_int(study.get("n_trials"), label="study.n_trials", minimum=1)
    _normalise_sampler_config(study.get("sampler"))
    pruner = _normalise_pruner_config(study.get("pruner"))
    if pruner["kind"] == "median" and int(pruner["n_startup_trials"]) >= int(study["n_trials"]):
        msg = "study.pruner.n_startup_trials must be smaller than study.n_trials."
        raise ValueError(msg)
    _sampler_seed(study)
    if "storage" in study:
        storage = study["storage"]
        if storage is not None:
            _require_nonempty_string(storage, label="study.storage")


def _normalise_study(raw_study: Mapping[str, Any], base_config: dict[str, Any], study_name: str) -> dict[str, Any]:
    """
    Normalize study policy and derive objective identity from the base config.

    Defaults are applied to an isolated copy, direction cannot be supplied
    independently from the resolved experiment objective, and all names and
    scalar types are validated before an ``OptunaStudyConfig`` is constructed.
    The study name is the study config's file stem and cannot be authored.
    """
    allowed = {"mode", "role", "seed", "n_trials", "sampler", "pruner", "storage"}
    unknown = sorted(set(raw_study).difference(allowed))
    if unknown:
        msg = f"study contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ValueError(msg)

    study = dict(copy.deepcopy(raw_study))
    study["name"] = study_name
    study.setdefault("role", None)
    study.setdefault("mode", "standard")
    study.setdefault("n_trials", 30)
    study.setdefault(
        "pruner",
        {"kind": "median", "n_startup_trials": 5, "n_warmup_steps": 25, "interval_steps": 5},
    )
    study.setdefault("sampler", {"kind": "tpe"})

    objective_config = _as_mapping(base_config["evaluation"]["objective"], label="evaluation.objective")
    study["objective"] = _require_nonempty_string(objective_config["id"], label="evaluation.objective.id")
    study["direction"] = _require_nonempty_string(
        objective_config["direction"],
        label="evaluation.objective.direction",
    )
    study["name"] = common.paths.validate_logical_name(
        _require_nonempty_string(study["name"], label="study.name"),
        label="study.name",
    )
    study["role"] = _normalise_study_role(study["role"])
    if study["mode"] not in {"standard", "stage_a_only"}:
        msg = "study.mode must be 'standard' or 'stage_a_only'."
        raise ValueError(msg)
    study["seed"] = _require_exact_int(study.get("seed"), label="study.seed")
    study["n_trials"] = _require_exact_int(study["n_trials"], label="study.n_trials", minimum=1)
    study["sampler"] = _normalise_sampler_config(study["sampler"])
    study["pruner"] = _normalise_pruner_config(study["pruner"])
    if "storage" in study and study["storage"] is not None:
        study["storage"] = _require_nonempty_string(study["storage"], label="study.storage")
    _validate_study_settings(study)
    return study


def _reject_matched_transient_plan(experiment: Mapping[str, Any]) -> None:
    """Reject a matched A0/B/A1 plan: Optuna tunes only the single-stage A0 protocol."""
    if transient_plan.is_matched_a1_b_config(experiment):
        message = "Transient Optuna studies tune A0 only through stage_a_only; matched A0/B/A1 plans are trained, not searched."
        raise ValueError(message)


def _validate_study_mode(base_config: Mapping[str, Any], *, mode: Any) -> None:
    """Require ``standard`` for steady studies and a direct Stage-A-only A0 schedule for transient studies."""
    if base_config.get("task") != "transient_drying":
        if mode != "standard":
            message = "stage_a_only is a transient_drying study mode; other tasks use study.mode standard."
            raise ValueError(message)
        return
    if mode != "stage_a_only":
        message = "Transient Optuna supports only the stage_a_only study mode."
        raise ValueError(message)
    schedule = _as_mapping(
        _as_mapping(base_config.get("training"), label="experiment.training").get("stage_schedule"),
        label="experiment.training.stage_schedule",
    )
    if schedule.get("mode") != "stage_a_only":
        message = "stage_a_only transient Optuna requires a direct Stage-A-only schedule."
        raise ValueError(message)


def load_optuna_study_config(
    path: Path | str,
    *,
    epochs: int | None = None,
    train_dataset: str | None = None,
    revision: str | None = None,
) -> OptunaStudyConfig:
    """
    Load and semantically validate one complete Optuna study recipe.

    The inline experiment is deep-copied, receives its wrapper-owned study
    tracking identity, and is resolved first. Study objective and direction are
    then derived from that resolved config before exact parameter schemas,
    approved paths, supported choices, and base-value containment are validated.

    Parameters
    ----------
    path : pathlib.Path | str
        Optuna YAML containing only ``study``, ``experiment``, and
        ``search_space`` top-level blocks.
    epochs : int | None, optional
        Replace the experiment's total epoch budget before semantic resolution.
    train_dataset : str | None, optional
        Replace the experiment's exact training Dataset ID before resolution.
    revision : str | None, optional
        Label of a deliberately separate study location (``run.revision``).

    Returns
    -------
    OptunaStudyConfig
        Source path, normalized study policy, untouched raw experiment copy,
        resolved base config, and parsed search parameters.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not identify a readable YAML file.
    KeyError
        If ``experiment`` or ``search_space`` is absent, or a required nested
        semantic field is missing.
    TypeError
        If a mapping, scalar, or search parameter has the wrong exact type.
    ValueError
        If keys, study policy, resolved experiment semantics, search schemas,
        approved paths, supported choices, or base-value containment are invalid.

    Notes
    -----
    Loading is side-effect free: it does not import Optuna, create output paths,
    open study storage, allocate trial directories, or mutate the caller's YAML
    mappings. Every failure occurs before study or trial lifecycle state exists.

    """
    source_path = Path(path)
    file_source = config_naming.config_source_from_path(source_path)
    if file_source.family not in {common.paths.OPTUNA_FAMILY, config_naming.EXTERNAL_FAMILY}:
        msg = f"Optuna YAML must live in configs/03_learning/<task>/optuna/ or outside the config tree, got {source_path}."
        raise ValueError(msg)
    study_source = config_naming.ConfigSource(file_source.task, common.paths.OPTUNA_FAMILY, file_source.name)
    raw = experiments.config.loader.load_yaml(source_path)
    raw_mapping = _as_mapping(raw, label="Optuna YAML")
    allowed_root = OPTUNA_ROOT_KEYS
    unknown_root = sorted(set(raw_mapping).difference(allowed_root))
    if unknown_root:
        msg = f"Optuna YAML contains unknown top-level key(s): {unknown_root}."
        raise ValueError(msg)

    if "experiment" not in raw_mapping:
        msg = "Optuna YAML must contain an experiment block"
        raise KeyError(msg)
    if "search_space" not in raw_mapping:
        msg = "Optuna YAML must contain a search_space block"
        raise KeyError(msg)

    base_experiment = experiments.config.loader.apply_training_overrides(
        experiments.config.loader.apply_authored_base(_as_mapping(raw_mapping["experiment"], label="experiment"), origin=source_path),
        epochs=epochs,
        train_dataset=train_dataset,
        revision=revision,
    )
    raw_run = _as_mapping(base_experiment.get("run"), label="experiment.run")
    if "seed" not in raw_run:
        msg = "experiment.run.seed must be explicit for an Optuna study."
        raise ValueError(msg)
    raw_study = _as_mapping(raw_mapping.get("study", {}), label="study")
    if "seed" not in raw_study:
        msg = "study.seed must be explicit for deterministic sampler reconstruction."
        raise ValueError(msg)
    study_name = study_source.name

    effective_experiment = copy.deepcopy(base_experiment)
    tracking = dict(_as_mapping(effective_experiment.get("tracking"), label="experiment.tracking"))
    wandb_settings = dict(_as_mapping(tracking.get("wandb"), label="experiment.tracking.wandb"))
    workflow = wandb_settings.get("workflow")
    if workflow != "optuna_trial":
        msg = "experiment.tracking.wandb.workflow must be 'optuna_trial' for an Optuna study."
        raise ValueError(msg)
    tracking["wandb"] = wandb_settings
    effective_experiment["tracking"] = tracking

    _reject_matched_transient_plan(effective_experiment)
    base_config = experiments.config.loader.resolve_config(effective_experiment, source=study_source)
    study = _normalise_study(raw_study, base_config, study_name)
    base_config = experiments.config.loader.validate_resolved_config(base_config)
    search_parameters = search_space.parse_search_space(raw_mapping["search_space"])
    search_space.validate_search_space_paths(base_config, search_parameters)
    _validate_search_space_choices(base_config, search_parameters)
    _validate_transient_study_policy(base_config, search_parameters)
    _validate_study_mode(base_config, mode=study["mode"])

    return OptunaStudyConfig(
        path=source_path,
        study=study,
        base_experiment=base_experiment,
        base_config=base_config,
        search_space=search_parameters,
    )


def _validate_reporting_contract(base_config: Mapping[str, Any]) -> int:
    """Require one shared positive validation, diagnostic, and physics cadence."""
    training = _as_mapping(base_config.get("training"), label="experiment.training")
    wandb = _as_mapping(_as_mapping(base_config.get("tracking"), label="experiment.tracking").get("wandb"), label="tracking.wandb")
    monitor = _as_mapping(wandb.get("monitor"), label="tracking.wandb.monitor")
    diagnostic_key = "additional_evaluation_interval"
    intervals = {
        "ID": _require_exact_int(training.get("evaluation_interval"), label="training.evaluation_interval", minimum=1),
        "diagnostic": _require_exact_int(training.get(diagnostic_key), label=f"training.{diagnostic_key}", minimum=1),
        "physics": _require_exact_int(monitor.get("interval"), label="tracking.wandb.monitor.interval", minimum=1),
    }
    if len(set(intervals.values())) != 1:
        msg = f"Optuna validation/diagnostic/physics intervals must share one completed-epoch cadence, got {intervals}."
        raise ValueError(msg)
    return intervals["ID"]


def _trial_name_values(
    parameters: Sequence[search_space.SearchSpaceParameter],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the sampled values a trial directory name shows; model parameters live in the model label."""
    return {
        parameter.name: overrides[parameter.path]
        for parameter in parameters
        if parameter.kind != "fixed" and not parameter.path.startswith(_MODEL_PARAMS_PREFIX)
    }


def _validate_search_space_choices(
    base_config: dict[str, Any],
    parameters: Sequence[search_space.SearchSpaceParameter],
) -> None:
    """
    Revalidate every declared endpoint, choice, or fixed value as a full config.

    Candidate values are applied one path at a time to the resolved base config.
    Any model, loss, device-policy, or task-contract violation fails before
    Optuna storage or a trial run is created.
    """
    for parameter in parameters:
        if parameter.kind == "categorical":
            candidates = parameter.values
        elif parameter.kind in {"float", "int"}:
            candidates = (parameter.low, parameter.high)
        else:
            candidates = (parameter.value,)
        for candidate in candidates:
            overrides = {parameter.path: candidate}
            candidate_config = search_space.apply_trial_overrides(base_config, overrides)
            candidate_config = experiments.config.loader.finalize_run_layout(
                candidate_config,
                trial=config_naming.trial_leaf(0, candidate_config, _trial_name_values((parameter,), overrides)),
            )
            try:
                experiments.config.loader.validate_resolved_config(candidate_config)
            except (KeyError, TypeError, ValueError) as error:
                msg = f"Search-space candidate {candidate!r} for {parameter.path!r} violates the resolved experiment contract."
                raise ValueError(msg) from error


def _validate_study_contract(config: OptunaStudyConfig) -> tuple[OptunaStudyConfig, dict[str, Any]]:
    """
    Revalidate a study and return its exact full objective before side effects.

    Study policy, base config, per-epoch reporting, search-path admission, and
    every declared candidate must agree. A defensive replacement containing the
    revalidated base config is returned with the canonical objective.
    """
    normalized_study = copy.deepcopy(config.study)
    normalized_study["role"] = _normalise_study_role(normalized_study.get("role"))
    _validate_study_settings(normalized_study)
    config = replace(config, study=normalized_study)
    base_config = experiments.config.loader.validate_resolved_config(config.base_config)
    _reject_matched_transient_plan(config.base_experiment)
    _validate_study_mode(base_config, mode=config.study["mode"])
    reporting_interval = _validate_reporting_contract(base_config)
    pruner = _normalise_pruner_config(config.study["pruner"])
    if pruner["kind"] == "median":
        if int(pruner["interval_steps"]) != reporting_interval:
            msg = "study.pruner.interval_steps must equal the genuine ID evaluation interval."
            raise ValueError(msg)
        warmup = int(pruner["n_warmup_steps"])
        if warmup < reporting_interval or warmup % reporting_interval != 0:
            msg = "study.pruner.n_warmup_steps must be a positive whole number of ID evaluation intervals."
            raise ValueError(msg)
    objective = experiments.config.loader.get_resolved_objective(base_config)
    if config.study.get("objective") != objective["id"]:
        msg = "Resolved study objective id does not match its experiment objective."
        raise ValueError(msg)
    if config.study.get("direction") != objective["direction"]:
        msg = "Resolved study direction does not match its experiment objective."
        raise ValueError(msg)
    search_space.validate_search_space_paths(base_config, config.search_space)
    _validate_search_space_choices(base_config, config.search_space)
    _validate_transient_study_policy(base_config, config.search_space)
    return replace(config, base_config=base_config), objective


def _scientific_base_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the positive completed-training dependency payload for study identity."""
    return learning.training.checkpoint.effective_config_identity_payload(base_config)


def _hpo_implementation_sha256(task_id: str) -> str:
    """Hash the Optuna policy, search admission and the study task's training implementation."""
    projection = HPO_PROJECTIONS.get(task_id)
    if projection is None:
        msg = f"No HPO implementation projection is declared for task {task_id!r}."
        raise ValueError(msg)
    return projection.sha256()


def build_study_signature(config: OptunaStudyConfig) -> dict[str, Any]:
    """
    Build the canonical invocation-independent scientific study signature.

    Parameters
    ----------
    config : OptunaStudyConfig
        Loaded study with resolved base experiment and parsed search parameters.

    Returns
    -------
    dict[str, Any]
        Schema version, SHA-256 digest, and canonical payload covering task,
        scientific config, search space, objective, sampler/pruner, reporting, and
        trial lifecycle policy.

    Raises
    ------
    TypeError
        If an in-memory study component has an invalid exact type.
    ValueError
        If study, objective, reporting cadence, search policy, or resolved base
        config has drifted outside the admitted semantic contract, or the
        configured Datasets are not bound to validated physical identities.

    Notes
    -----
    Device/output location and W&B settings are operational and excluded. Every
    model-selection or training semantic participates in the digest. Construction
    is deterministic and does not mutate ``config`` or touch study storage.

    """
    validated, objective = _validate_study_contract(config)
    sampler = _normalise_sampler_config(validated.study["sampler"])
    pruner = _normalise_pruner_config(validated.study["pruner"])
    sampler_seed = _sampler_seed(validated.study)
    reporting_interval = _validate_reporting_contract(validated.base_config)
    if validated.dataset_identities is None:
        msg = "A study signature binds validated physical Dataset identities; bind the configured Datasets first."
        raise ValueError(msg)
    dataset_roles = {"binding": "validated_physical_packages", "roles": copy.deepcopy(validated.dataset_identities)}
    payload = {
        "schema_version": STUDY_SIGNATURE_SCHEMA_VERSION,
        "task": {
            "id": validated.base_config["task"],
            "schema_version": validated.base_config["task_contract"]["schema_version"],
            "contract_digest": validated.base_config["task_contract"]["digest"],
        },
        "scientific_base_config": _scientific_base_config(validated.base_config),
        "dataset_roles": dataset_roles,
        "search_space": sorted(
            search_space.search_space_summary(validated.search_space),
            key=lambda item: (str(item["path"]), str(item["name"])),
        ),
        "study_role": validated.study["role"],
        "study_mode": validated.study["mode"],
        "objective": objective,
        "direction": objective["direction"],
        "sampler": {**sampler, "seed": sampler_seed},
        "pruner": pruner,
        "reporting": {
            "metric_source": "held_out_evaluation",
            "evaluation_interval": reporting_interval,
            "event_model": "completed_epoch_interval_or_terminal",
            "step": "actual_completed_epoch",
            "pruning_subset": "id_only",
        },
        "trial_lifecycle": {
            "schema_version": TRIAL_LIFECYCLE_SCHEMA_VERSION,
            "run_summary_schema_version": common.paths.RUN_SUMMARY_SCHEMA_VERSION,
            "outcomes": list(_TRIAL_OUTCOMES),
            "resume_policy": "new_trials_only",
            "trial_count_policy": "additional_fresh_trials_per_invocation",
            "training_seed_policy": "fixed_configured_run_seed_across_trials",
            "trial_identity_policy": "native_zero_based_optuna_number",
        },
        "hpo_implementation_sha256": _hpo_implementation_sha256(validated.base_config["task"]),
    }
    return {
        "schema_version": STUDY_SIGNATURE_SCHEMA_VERSION,
        "digest": common.serialization.canonical_json_sha256(payload),
        "payload": payload,
    }


def describe_optuna_study_config(config: OptunaStudyConfig) -> dict[str, Any]:
    """
    Validate and describe a study without creating runtime state.

    Parameters
    ----------
    config : OptunaStudyConfig
        Loaded or programmatically modified study configuration.

    Returns
    -------
    dict[str, Any]
        Serializable source, study, paths, dataset/task/model identities, device
        policy, search space, objective, and complete semantic signature.

    Raises
    ------
    TypeError
        If an in-memory study or compact dataset-metadata component is invalid.
    ValueError
        If study policy, resolved science, dataset identity, reporting, or search
        semantics have drifted since loading.

    Notes
    -----
    Description reads compact model-training metadata but does not load dataset
    tensors, resolve hardware, create paths, open storage, allocate trials,
    initialize W&B, or import the Optuna SDK.

    """
    validated, objective = _validate_study_contract(config)
    validated = _bind_configured_dataset_identities(validated)
    signature = build_study_signature(validated)
    study_paths = _resolve_study_paths(validated)
    return {
        "path": str(validated.path),
        "study": validated.study,
        "base_run_name": validated.base_config["run"]["name"],
        "device_policy": validated.base_config["run"]["device"],
        "storage": study_paths.storage,
        "study_dir": str(study_paths.study_dir),
        "trial_root": str(study_paths.trial_root),
        "task": validated.base_config["task"],
        "model_kind": validated.base_config["model"]["kind"],
        "dataset_roles": copy.deepcopy(validated.dataset_identities),
        "objective": objective,
        "search_space": search_space.search_space_summary(validated.search_space),
        "semantic_signature": signature,
    }


def _study_dir(config: OptunaStudyConfig) -> Path:
    """Return the task-, dataset- and epoch-budget-owned study directory under the experiments root."""
    return common.paths.resolve_study_dir(
        str(config.base_config["task"]),
        str(config.base_config["data"]["train_dataset"]),
        str(config.study["name"]),
        config.base_config["training"]["epochs"],
        config.base_config["run"].get("revision"),
        output_root=Path(config.base_config["paths"]["output_root"]),
    )


def _database_study_id(config: OptunaStudyConfig) -> str:
    """Return the storage-wide study identifier, unique across tasks, Datasets, epoch budgets and revisions."""
    return f"{config.base_config['task']}/{config.base_config['data']['train_dataset']}/{_study_dir(config).name}"


def _resolve_study_paths(config: OptunaStudyConfig) -> _OptunaStudyPaths:
    """Resolve storage, study, and trial roots without creating any path."""
    study_name = common.paths.validate_logical_name(config.study["name"], label="study.name")
    study_dir = _study_dir(config)
    configured_storage = config.study.get("storage")
    local_storage_path = None if configured_storage is not None else study_dir / f"{study_name}.db"
    storage = str(configured_storage) if configured_storage is not None else f"sqlite:///{local_storage_path}"
    return _OptunaStudyPaths(study_dir, study_dir / common.paths.STUDY_TRIALS_DIRNAME, storage, local_storage_path)


def _configured_dataset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact configured Dataset selections without assigning package roles."""
    data = _as_mapping(config.get("data"), label="experiment.data")
    train_dataset = _require_nonempty_string(data.get("train_dataset"), label="data.train_dataset")
    diagnostic_key = "additional_evaluation_datasets"
    raw_diagnostic = data.get(diagnostic_key)
    if not isinstance(raw_diagnostic, list):
        msg = f"data.{diagnostic_key} must be a list of configured Dataset ids."
        raise TypeError(msg)
    diagnostic_datasets = [
        _require_nonempty_string(dataset_id, label=f"data.{diagnostic_key}[{index}]") for index, dataset_id in enumerate(raw_diagnostic)
    ]
    if len(diagnostic_datasets) != len(set(diagnostic_datasets)):
        msg = f"data.{diagnostic_key} must not contain duplicates."
        raise ValueError(msg)

    task = experiments.config.loader.validate_resolved_task_contract(config)
    paths = _as_mapping(config.get("paths"), label="experiment.paths")
    storage_root = common.paths.storage_root_for_datasets_root(_require_nonempty_string(paths.get("dataset_root"), label="paths.dataset_root"))

    def summarize(dataset_id: str) -> dict[str, Any]:
        # The one manifest reader admits the manifest and rehashes the payload digests it binds, for either task.
        manifest = datasets.packages.manifest.load_package_manifest(task.id, dataset_id, storage_root=storage_root)
        # Locations stay outside the signature, as device and output root do: the digests identify the Dataset.
        return {
            "dataset_id": manifest["dataset_id"],
            "dataset_view": manifest["dataset_view"],
            "dataset_digest": manifest["dataset_digest"],
            "channel_contract_digest": manifest["channel_contract_digest"],
            "manifest_schema": {"kind": manifest["schema_kind"], "version": manifest["schema_version"]},
            "payload_sha256": manifest["payload_sha256"],
            "derived_payload_sha256": manifest["derived_payload_sha256"],
            "sample_count": manifest["sample_count"],
            "source_case_count": manifest["source_case_count"],
            "materials": copy.deepcopy(manifest["materials"]),
            "source_case_identity_digest": common.serialization.canonical_json_sha256(manifest["source_case_identities"]),
            "validation": "validated_package_manifest_and_payload_digests",
        }

    return {
        "train": summarize(train_dataset),
        "additional_evaluation": [summarize(dataset_id) for dataset_id in diagnostic_datasets],
    }


def _bind_configured_dataset_identities(config: OptunaStudyConfig) -> OptunaStudyConfig:
    """Bind current validated physical Dataset evidence to one study invocation."""
    identities = _configured_dataset_identities(config.base_config)
    return replace(config, dataset_identities=copy.deepcopy(identities))


def _build_pruner(study: Mapping[str, Any]) -> Any:
    """
    Build the exact Optuna pruner from validated semantic policy.

    Optuna is imported lazily only at construction. ``none`` maps to ``NopPruner``
    and ``median`` forwards the already normalized startup, warmup, and interval values.
    """
    pruner_cfg = _normalise_pruner_config(study.get("pruner"))
    optuna = _optuna_module()
    pruner_type = pruner_cfg["kind"]

    if pruner_type == "none":
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 25),
            interval_steps=pruner_cfg.get("interval_steps", 5),
        )
    msg = f"Unsupported Optuna pruner: {pruner_type!r}"
    raise ValueError(msg)


def _sampler_seed(study: Mapping[str, Any]) -> int:
    """Return the explicit persisted Optuna sampler seed without derivation."""
    seed = _require_exact_int(study.get("seed"), label="study.seed", minimum=0)
    if seed >= 2**32:
        msg = f"study.seed must fit Optuna/NumPy's 32-bit sampler domain, got {seed}."
        raise ValueError(msg)
    return seed


def _build_sampler(study: Mapping[str, Any]) -> Any:
    """
    Build a seeded Optuna sampler from validated semantic policy.

    Both random and TPE receive the explicit persisted 32-bit study seed. Only
    TPE accepts the normalized multivariate flag. Optuna remains lazily imported.
    """
    sampler_cfg = _normalise_sampler_config(study.get("sampler"))
    sampler_type = sampler_cfg["kind"]
    seed = _sampler_seed(study)
    optuna = _optuna_module()

    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=sampler_cfg.get("multivariate", False),
        )
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    msg = f"Unsupported Optuna sampler: {sampler_type!r}"
    raise ValueError(msg)


def _prepare_trial_config(study_config: OptunaStudyConfig, trial: TrialProtocol) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one trial while preserving the configured fixed training seed."""
    trial_number = _require_exact_int(trial.number, label="Optuna trial.number", minimum=0)
    overrides = search_space.suggest_trial_overrides(trial, study_config.search_space)
    config = search_space.apply_trial_overrides(study_config.base_config, overrides)
    training_seed = _require_exact_int(config["run"]["seed"], label="experiment.run.seed", minimum=0)
    sampler_seed = _sampler_seed(study_config.study)
    wandb_settings = config["tracking"]["wandb"]
    wandb_settings["workflow"] = "optuna_trial"
    wandb_settings.update(experiments.config.loader.derive_wandb_organization(config))
    leaf = config_naming.trial_leaf(trial_number, config, _trial_name_values(study_config.search_space, overrides))
    config = experiments.config.loader.finalize_run_layout(config, trial=leaf)
    config = experiments.config.loader.validate_resolved_config(config)

    base_objective = experiments.config.loader.get_resolved_objective(study_config.base_config)
    trial_objective = experiments.config.loader.get_resolved_objective(config)
    if trial_objective != base_objective:
        msg = "Sampled trial objective does not match the resolved study objective."
        raise ValueError(msg)

    sampled_parameters = {
        parameter.name: copy.deepcopy(overrides[parameter.path]) for parameter in study_config.search_space if parameter.kind != "fixed"
    }
    context = {
        "study_name": str(study_config.study["name"]),
        "study_role": study_config.study["role"],
        "study_mode": study_config.study["mode"],
        "trial_number": trial_number,
        "training_seed": training_seed,
        "sampler_seed": sampler_seed,
        "overrides": overrides,
        "sampled_parameters": sampled_parameters,
        "search_signature": build_study_signature(study_config)["digest"],
    }
    trial.set_user_attr("run_name", config["run"]["name"])
    trial.set_user_attr("run_seed", training_seed)
    trial.set_user_attr("sampler_seed", sampler_seed)
    trial.set_user_attr("study_role", context["study_role"])
    trial.set_user_attr("study_mode", context["study_mode"])
    trial.set_user_attr("overrides", overrides)
    return config, context


def _trial_run_dir(config: Mapping[str, Any]) -> Path:
    """Return the trial directory below its study, as the saved layout declares it."""
    return common.paths.resolve_config_run_dir(config)


def _finite_objective_value(result: Mapping[str, Any], objective: Mapping[str, Any]) -> float:
    """
    Admit the training result's objective identity and finite selected value.

    A mismatched full objective is a configuration/lifecycle error. A matching
    but NaN or infinite best metric becomes ``NonFiniteTrialError`` for distinct
    trial-outcome classification.
    """
    if result.get("objective") != objective:
        msg = "Training result objective does not match the resolved trial objective."
        raise ValueError(msg)
    objective_value = float(result["best_metric"])
    if not math.isfinite(objective_value):
        msg = "Training completed without a finite held-out objective."
        raise NonFiniteTrialError(msg)
    return objective_value


def _runtime_error_is_allocation_oom(error: RuntimeError) -> bool:
    """
    Classify only established CUDA or CPU allocator OOM ``RuntimeError`` text.

    Matching is intentionally narrow so arbitrary runtime, driver, shape, and
    programming errors remain fatal rather than being mislabeled as prunable
    resource exhaustion.
    """
    message = str(error).lower()
    return (
        message.startswith("cuda out of memory")
        or "cuda error: out of memory" in message
        or "defaultcpuallocator: can't allocate memory" in message
        or "defaultcpuallocator: not enough memory" in message
        or "mmap failed: cannot allocate memory" in message
    )


def _failure_epoch(error: BaseException, reporter: OptunaEpochReporter) -> int | None:
    """
    Recover an explicit failure epoch without inventing progress.

    A maintained ``" at epoch N:"`` message marker takes precedence. Otherwise
    only the reporter's last successfully published completed epoch is returned.
    """
    marker = " at epoch "
    message = str(error)
    if marker in message:
        suffix = message.split(marker, 1)[1]
        digits = suffix.split(":", 1)[0].strip()
        if digits.isdigit():
            return int(digits)
    return reporter.last_reported_epoch


def _failure_context(
    *,
    error: BaseException,
    device_resolution: learning.device.DeviceResolution,
    reporter: OptunaEpochReporter,
) -> dict[str, Any]:
    """
    Build bounded trial-failure context without initializing a CUDA context.

    Requested/concrete device, explicit epoch, latest Optuna report, and global
    step are always safe. CUDA allocation counters are queried only when the
    resolved device is CUDA and Torch already initialized that runtime.
    """
    context: dict[str, Any] = {
        "error_type": type(error).__name__,
        "requested_device_policy": device_resolution.requested_policy,
        "resolved_device": str(device_resolution.device),
        "epoch": _failure_epoch(error, reporter),
        "last_reported_epoch": reporter.last_reported_epoch,
        "last_reported_objective": reporter.last_reported_objective,
        "last_global_step": reporter.last_global_step,
    }
    device = device_resolution.device
    if device.type == "cuda" and torch.cuda.is_initialized():
        with suppress(RuntimeError):
            context["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            context["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
            context["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            context["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return context


def _checkpoint_progress(
    run_dir: Path,
    reporter: OptunaEpochReporter,
) -> dict[str, Any]:
    """
    Bind an unfinished trial's progress to its durable best checkpoint.

    Trials never resume and write no ``last_checkpoint.pt``, so no trial summary
    records one. The loop publishes a best before it reports that epoch, so best
    epoch, metric and digest come from ``best_checkpoint.pt`` whenever it exists,
    with its progress when that is later than the last report. A finite reported
    best without the file is a lifecycle violation rather than guessed or
    backfilled progress. Every key is returned, ``None`` when nothing was reached.
    """
    best_path = common.paths.resolve_best_checkpoint_file(run_dir)
    if reporter.best_epoch is not None and not best_path.is_file():
        msg = "A finite best Optuna objective must already have a durable best checkpoint."
        raise experiments.run.RunLifecycleError(msg)
    progress: dict[str, Any] = {
        **dict.fromkeys(("best_epoch", "best_metric", "best_checkpoint", "best_checkpoint_sha256")),
        "completed_epoch": reporter.last_reported_epoch,
        "global_step": reporter.last_global_step if reporter.last_reported_epoch is not None else None,
    }
    if best_path.is_file():
        # Memory-mapped, because this runs in failure handlers (host memory exhaustion included) while the
        # model is still alive; training has stopped, so the digest below reads the same file.
        payload = torch.load(best_path, map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(payload, Mapping) or not {"best_epoch", "best_metric", "completed_epoch", "global_step"} <= payload.keys():
            msg = f"Optuna best checkpoint does not record its best and progress: {best_path}"
            raise experiments.run.RunLifecycleError(msg)
        progress.update(
            {
                "best_epoch": payload["best_epoch"],
                "best_metric": payload["best_metric"],
                "best_checkpoint": best_path.name,
                "best_checkpoint_sha256": common.serialization.file_sha256(best_path),
            }
        )
        if reporter.last_reported_epoch is None or payload["completed_epoch"] > reporter.last_reported_epoch:
            progress.update({"completed_epoch": payload["completed_epoch"], "global_step": payload["global_step"]})
    return progress


def _trial_terminal_evidence(
    status: str,
    outcome: Mapping[str, Any] | BaseException,
    *,
    run_dir: Path,
    reporter: OptunaEpochReporter,
    objective: Mapping[str, Any],
    device_resolution: learning.device.DeviceResolution,
) -> dict[str, Any]:
    """
    Return the trial facts the shared run lifecycle merges into a terminal summary.

    A completed result must have reported its terminal epoch and carry a finite
    objective; a raise fails the run. A failure binds its bounded context and the
    best evidence of :func:`_checkpoint_progress` before the best file goes.
    """
    evidence: dict[str, Any] = {
        "last_reported_epoch": reporter.last_reported_epoch,
        "last_reported_objective": reporter.last_reported_objective,
        "pruning_epoch": reporter.last_reported_epoch if status == "pruned" else None,
    }
    if isinstance(outcome, BaseException):
        failure_context = _failure_context(error=outcome, device_resolution=device_resolution, reporter=reporter)
        return {**evidence, "failure_context": failure_context, **_checkpoint_progress(run_dir, reporter)}
    _validate_completed_reporting(reporter, outcome)
    _finite_objective_value(outcome, objective)
    return {**evidence, "failure_context": None}


def _validate_completed_reporting(
    reporter: OptunaEpochReporter,
    result: Mapping[str, Any],
) -> None:
    """
    Require Optuna's latest durable report to equal training's terminal epoch.

    This closes the lifecycle only when actual completed-epoch pruning evidence
    is continuous through the same terminal checkpoint represented by the result.
    """
    if reporter.last_reported_epoch != result.get("completed_epoch"):
        msg = "Completed training did not report its actual terminal epoch to Optuna."
        raise RuntimeError(msg)


def _validate_transient_study_policy(
    config: Mapping[str, Any],
    parameters: Sequence[search_space.SearchSpaceParameter],
) -> None:
    """Enforce transient Optuna search ownership: the A0 arm, the TaskSpec objective, and admitted architectures."""
    if config.get("task") != "transient_drying":
        return
    objective = experiments.config.loader.get_resolved_objective(config)
    transient_task = domain.tasks.registry.get_task("transient_drying")
    expected_metric = next(metric for metric in transient_task.default_metrics if metric.id == "normalized_drying_group_macro_rmse")
    expected_objective = expected_metric.as_dict(all_fields=transient_task.output_names)
    if objective != expected_objective:
        message = "Transient Optuna requires the TaskSpec-owned normalized drying group-macro objective."
        raise ValueError(message)
    training = _as_mapping(config["training"], label="experiment.training")
    if training["stage"] != "a" or training["comparison_arm"] != "a0":
        message = "Transient Optuna must use the immutable Stage-A A0 comparison arm."
        raise ValueError(message)
    model_paths = tuple(parameter.path for parameter in parameters if parameter.path.startswith("model."))
    if model_paths and config["model"]["kind"] not in {"fno", "uno", "rno"}:
        message = "Transient Optuna architecture search admits only FNO, UNO, and RNO."
        raise ValueError(message)


def _trial_terminal_status(error: BaseException, trial_pruned: type[BaseException]) -> str:
    """Map a trial's exception to its terminal status; explicit classes take precedence over allocator text."""
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    if isinstance(error, trial_pruned):
        return "pruned"
    if isinstance(error, FloatingPointError):
        return "nonfinite_pruned"
    if isinstance(error, RecoverableTrialError):
        return "recoverable_failed"
    if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)) or (isinstance(error, RuntimeError) and _runtime_error_is_allocation_oom(error)):
        return "oom_pruned"
    return "failed"


def _duplicate_trial_outcome(config: Mapping[str, Any], trial: TrialProtocol) -> tuple[float | None, str] | None:
    """Return the stored value and owner of a trial whose exact experiment already ran.

    A completed trial of the same study (which fixes the epoch budget through its
    directory name) with the same objective supplies its stored value. Any other
    existing owner of the identity (another study of the same epoch budget, an
    unfinished or pruned trial) supplies no value, and the trial is pruned. The
    trial's digest includes its epoch budget, so a trial of another budget and an
    ordinary steady run (whose digest is epoch-free) own other identities;
    ``None`` means the identity is new.
    """
    entry = experiments.registry.lookup(config)
    if entry is None:
        return None
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve(strict=False)
    relative = entry.target.resolve(strict=False).relative_to(output_root).as_posix()
    trial.set_user_attr("duplicate_of", relative)
    same_study = entry.kind == "trial" and entry.target.resolve(strict=False).parent == _trial_run_dir(config).resolve(strict=False).parent
    if same_study:
        summary = experiments.run.read_run_summary(entry.target)
        value = summary.get("best_metric")
        if (
            summary.get("status") == "completed"
            and summary.get("objective") == dict(config["evaluation"]["objective"])
            # The trained budget must match the study's: a later extension changes the stored best value.
            and summary.get("completed_epoch") == config["training"]["epochs"]
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value), relative
    return None, relative


def _settle_close_failures(failures: Sequence[tuple[str, BaseException]], *, propagating: BaseException | None) -> None:
    """Note closing-step failures on the trial's propagating error, or raise the first one when the trial returned normally."""
    if not failures:
        return
    notes = [f"{label}: {type(error).__name__}: {error}" for label, error in failures]
    if propagating is not None:
        for note in notes:
            propagating.add_note(note)
        return
    first = failures[0][1]
    for note in notes[1:]:
        first.add_note(note)
    raise first


def run_trial(
    study_config: OptunaStudyConfig,
    trial: TrialProtocol,
) -> float:
    """
    Execute one trial of either task through the shared prepared-run lifecycle.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Validated study contract and runtime overrides.
    trial : TrialProtocol
        Optuna-compatible trial supplying suggestions, reports, pruning, and attrs.

    Returns
    -------
    float
        Finite terminal held-out value of the resolved primary objective.

    Raises
    ------
    optuna.TrialPruned
        For explicit pruning, non-finite science, narrowly classified OOM, or a
        duplicate whose stored value cannot be reused.
    RecoverableTrialError
        For an explicitly recoverable trial-local failure.
    BaseException
        Interrupts and programming, identity, storage or tracking errors
        propagate unchanged after the terminal publication.

    Notes
    -----
    A duplicate is answered from stored evidence before any device is resolved
    or run directory allocated. Every other trial gets a fresh study-qualified
    run leaf and never resumes. The shared lifecycle publishes the terminal
    summary with this driver's status and evidence before the best checkpoint is
    deleted; this driver sets ``terminal_status`` and prints the one ``trial``
    line of every outcome. Setting the attribute, printing the line and the
    cleanup each run even when another of them fails; their failures become
    notes on the trial's own propagating error, or the first is raised after a
    normal return.

    """
    started = time.monotonic()
    trial_pruned = _trial_pruned_error()
    returned = False
    status = "failed"
    value: float | None = None
    device: torch.device | None = None
    # Setup runs inside the guarded attempt, so a refused sample still ends with a status and one trial line.
    context: Mapping[str, Any] = {"trial_number": trial.number, "sampled_parameters": {}}
    reporter: OptunaEpochReporter | None = None

    def attempt() -> float:
        """Answer a duplicate from stored evidence, or train the trial through the shared run lifecycle."""
        nonlocal status, device, context, reporter
        validated_config, _ = _validate_study_contract(study_config)
        config, context = _prepare_trial_config(validated_config, trial)
        objective = experiments.config.loader.get_resolved_objective(config)
        epoch_reporter = reporter = OptunaEpochReporter(
            trial=trial,
            objective_id=str(objective["id"]),
            direction=str(objective["direction"]),
            evaluation_interval=int(config["training"]["evaluation_interval"]),
            target_epoch=int(config["training"]["epochs"]),
        )

        def report(epoch: int, metrics: dict[str, float]) -> None:
            """Report each evaluated epoch to Optuna, after the transient one-step guardrail, and expose it to W&B."""
            key = f"id/{objective['id']}"
            if key not in metrics:
                return
            guardrail = metrics.get(f"id/guardrail/one_step/{objective['id']}", math.nan)
            if config["task"] == "transient_drying" and not math.isfinite(float(guardrail)):
                message = "Transient one-step guardrail must be finite at each reported evaluation."
                raise NonFiniteTrialError(message)
            epoch_reporter(epoch, metrics)
            if epoch_reporter.best_value is not None:
                metrics.update({"optuna/objective": float(metrics[key]), "optuna/best_objective": epoch_reporter.best_value})

        _validate_transient_study_policy(config, validated_config.search_space)
        duplicate = _duplicate_trial_outcome(config, trial)
        if duplicate is not None:
            stored, owner = duplicate
            if stored is None:
                status = "duplicate_pruned"
                message = f"Trial duplicates the existing experiment at {owner}; it was not trained again."
                raise trial_pruned(message)
            status = "duplicate"
            return stored
        device_resolution = learning.device.resolve_device(config["run"]["device"], path="run.device")
        device = device_resolution.device
        experiments.run.validate_deterministic_model_device_policy(config, device_resolution)
        # Preparation and execution classify a failure of the trial's run like the trial itself.
        resolve_status = functools.partial(_trial_terminal_status, trial_pruned=trial_pruned)
        run_dir = experiments.run.prepare_fresh_run(
            config, run_dir=_trial_run_dir(config), summary_extra=context, terminal_status_resolver=resolve_status
        )
        result = experiments.run.execute_prepared_run(
            config,
            run_dir=run_dir,
            persisted_config=config,
            epoch_end_callback=report,
            summary_extra=context,
            device_resolution=device_resolution,
            terminal_status_resolver=resolve_status,
            terminal_evidence=lambda terminal, outcome: _trial_terminal_evidence(
                terminal, outcome, run_dir=run_dir, reporter=epoch_reporter, objective=objective, device_resolution=device_resolution
            ),
        )
        status = "completed"
        # The completed evidence admitted this value as finite before the run was published.
        return float(result["best_metric"])

    try:
        value = attempt()
    except BaseException as error:
        if status != "duplicate_pruned":
            status = _trial_terminal_status(error, trial_pruned)
        if status == "failed":
            experiments.console.print_sanitized_traceback(error)
        else:
            # Pruned outcomes print no traceback; a lifecycle step that failed while recording them left notes.
            experiments.console.print_error_notes(error)
        if status == "nonfinite_pruned":
            message = f"Trial pruned after non-finite value: {error}"
            raise trial_pruned(message) from None
        if status == "oom_pruned":
            raise trial_pruned(_OOM_PRUNED_MESSAGE) from error
        raise
    else:
        returned = True
        return value
    finally:
        # Each closing step runs even when an earlier one fails; their failures are settled after all three.
        close_failures: list[tuple[str, BaseException]] = []
        try:
            trial.set_user_attr("terminal_status", status)
        except BaseException as error:  # noqa: BLE001 -- settled below, after the trial line and cleanup
            close_failures.append((f"Optuna attribute terminal_status={status!r} was not recorded", error))
        try:
            experiments.console.optuna_trial_event(
                status,
                trial=int(context["trial_number"]),
                sampled=context["sampled_parameters"],
                duration_seconds=time.monotonic() - started,
                objective=value if value is not None or reporter is None else reporter.last_reported_objective,
                step=None if reporter is None else reporter.last_reported_epoch,
            )
        except BaseException as error:  # noqa: BLE001 -- settled below, after the cleanup
            close_failures.append(("The trial line was not printed", error))
        try:
            gc.collect()
            if device is not None and device.type == "cuda" and torch.cuda.is_initialized():
                with suppress(RuntimeError):
                    torch.cuda.empty_cache()
                with suppress(RuntimeError):
                    torch.cuda.reset_peak_memory_stats(device)
        except BaseException as error:  # noqa: BLE001 -- settled below
            close_failures.append(("The trial cleanup did not finish", error))
        _settle_close_failures(close_failures, propagating=None if returned else sys.exception())


def create_objective(study_config: OptunaStudyConfig) -> Callable[[TrialProtocol], float]:
    """
    Create an Optuna objective callable for a resolved study config.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Resolved Optuna study configuration

    Returns
    -------
    Callable[[TrialProtocol], float]
        Objective function suitable for study.optimize

    """

    def objective(trial: TrialProtocol) -> float:
        """Run one trial for Optuna's optimize loop."""
        return run_trial(study_config, trial)

    return objective


def with_runtime_overrides(
    config: OptunaStudyConfig,
    *,
    device: str | None = None,
    output_root: Path | str | None = None,
) -> OptunaStudyConfig:
    """
    Return a validated study copy with invocation-only runtime overrides.

    Parameters
    ----------
    config : OptunaStudyConfig
        Resolved study configuration to copy.
    device : str | None, optional
        Exact runtime policy applied to every trial for this invocation.
    output_root : pathlib.Path | str | None, optional
        Invocation-only study and trial output root.

    Returns
    -------
    OptunaStudyConfig
        Revalidated copy retaining unchanged scientific study semantics.

    Raises
    ------
    experiments.config.loader.ConfigError
        If the device override or resulting resolved config is invalid.

    Notes
    -----
    The input object is not mutated. This function validates requested policy but
    does not resolve hardware, create the output root, or alter the semantic study
    signature because both overrides are invocation-only.

    """
    base_config = copy.deepcopy(config.base_config)
    if device is not None:
        base_config["run"]["device"] = device
    if output_root is not None:
        base_config["paths"]["output_root"] = str(Path(output_root).expanduser())
    base_config = experiments.config.loader.validate_resolved_config(base_config)
    return replace(config, base_config=base_config, dataset_identities=None)


def _publish_study_signature(
    study: Any,
    signature: Mapping[str, Any],
    objective: Mapping[str, Any],
    *,
    numerical_environment: Mapping[str, Any] | None = None,
) -> None:
    """
    Attach the complete semantic signature and objective to a new study.

    Schema version, digest, canonical payload, and full objective are published
    together before optimization so later reopen admission can fail closed.
    """
    study.set_user_attr(_STUDY_SIGNATURE_SCHEMA_ATTR, signature["schema_version"])
    study.set_user_attr(_STUDY_SIGNATURE_ATTR, signature["digest"])
    study.set_user_attr(_STUDY_SIGNATURE_PAYLOAD_ATTR, signature["payload"])
    study.set_user_attr(_RESOLVED_OBJECTIVE_ATTR, dict(objective))
    study.set_user_attr(_SAMPLER_METADATA_ATTR, copy.deepcopy(signature["payload"]["sampler"]))
    environment = numerical_environment
    if environment is None:
        environment = learning.training.checkpoint.numerical_environment_identity()
    study.set_user_attr(_NUMERICAL_ENVIRONMENT_ATTR, copy.deepcopy(dict(environment)))


def _validate_existing_study(
    study: Any,
    *,
    signature: Mapping[str, Any],
    objective: Mapping[str, Any],
    numerical_environment: Mapping[str, Any] | None = None,
) -> None:
    """
    Admit an existing study only when direction and semantic metadata match.

    Missing signature fields, payload drift, or objective drift are rejected.
    Name and storage identity alone never authorize adding fresh trials.
    """
    actual_direction = str(study.direction.name).lower()
    if actual_direction != objective["direction"]:
        msg = f"Existing Optuna study direction {actual_direction!r} does not match resolved objective direction {objective['direction']!r}."
        raise ValueError(msg)
    environment = numerical_environment
    if environment is None:
        environment = learning.training.checkpoint.numerical_environment_identity()
    required = {
        _STUDY_SIGNATURE_SCHEMA_ATTR: signature["schema_version"],
        _STUDY_SIGNATURE_ATTR: signature["digest"],
        _STUDY_SIGNATURE_PAYLOAD_ATTR: signature["payload"],
        _RESOLVED_OBJECTIVE_ATTR: dict(objective),
        _SAMPLER_METADATA_ATTR: copy.deepcopy(signature["payload"]["sampler"]),
        _NUMERICAL_ENVIRONMENT_ATTR: copy.deepcopy(dict(environment)),
    }
    missing = sorted(key for key in required if key not in study.user_attrs)
    if missing:
        msg = f"Existing Optuna study is missing required semantic metadata: {missing}."
        raise ValueError(msg)

    def require_schema_version(value: Any, expected: int, *, label: str) -> None:
        if not common.serialization.is_schema_version(value, expected):
            msg = f"Existing Optuna study {label} must be schema version {expected}."
            raise ValueError(msg)

    require_schema_version(
        study.user_attrs[_STUDY_SIGNATURE_SCHEMA_ATTR],
        STUDY_SIGNATURE_SCHEMA_VERSION,
        label=_STUDY_SIGNATURE_SCHEMA_ATTR,
    )
    actual_payload = study.user_attrs[_STUDY_SIGNATURE_PAYLOAD_ATTR]
    if not isinstance(actual_payload, Mapping):
        msg = "Existing Optuna study semantic signature payload must be a mapping."
        raise TypeError(msg)
    actual_task = actual_payload.get("task")
    actual_lifecycle = actual_payload.get("trial_lifecycle")
    if not isinstance(actual_task, Mapping) or not isinstance(actual_lifecycle, Mapping):
        msg = "Existing Optuna study semantic signature has invalid task or trial lifecycle metadata."
        raise TypeError(msg)
    require_schema_version(
        actual_payload.get("schema_version"),
        STUDY_SIGNATURE_SCHEMA_VERSION,
        label="signature payload",
    )
    expected_payload = signature["payload"]
    require_schema_version(
        actual_task.get("schema_version"),
        expected_payload["task"]["schema_version"],
        label="task contract",
    )
    require_schema_version(
        actual_lifecycle.get("schema_version"),
        TRIAL_LIFECYCLE_SCHEMA_VERSION,
        label="trial lifecycle",
    )
    require_schema_version(
        actual_lifecycle.get("run_summary_schema_version"),
        common.paths.RUN_SUMMARY_SCHEMA_VERSION,
        label="run summary",
    )

    mismatched = sorted(key for key, value in required.items() if study.user_attrs.get(key) != value)
    if mismatched:
        msg = f"Existing Optuna study semantic signature mismatch in: {mismatched}."
        raise ValueError(msg)


def _create_or_load_study(
    *,
    optuna: Any,
    study_name: str,
    direction: str,
    pruner: Any,
    sampler: Any,
    storage: str,
) -> tuple[Any, bool]:
    """
    Create a study or explicitly load the exact duplicate-name study.

    Creation never uses ``load_if_exists``. Only Optuna's dedicated duplicate
    exception selects the load path, and the returned boolean requires callers
    to perform semantic-signature admission before optimization.
    """
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            storage=storage,
            load_if_exists=False,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            pruner=pruner,
            sampler=sampler,
        )
        return study, True
    return study, False


def _storage_console_location(paths: _OptunaStudyPaths) -> str:
    """Return a useful storage label without credentials or query parameters."""
    if paths.local_storage_path is not None:
        return str(paths.local_storage_path)
    scheme = paths.storage.split(":", 1)[0].strip().lower()
    return f"{scheme or 'external'}://<configured>"


def _trial_state_name(trial: Any) -> str:
    """Return one stable lowercase Optuna trial state name."""
    return str(getattr(getattr(trial, "state", None), "name", "unknown")).lower()


def _fail_running_trials(study: Any, numbers: Sequence[int], error: BaseException) -> None:
    """Record FAIL for this invocation's trials that an exception outside Optuna's own handling (``SystemExit``) left RUNNING."""
    states = _optuna_module().trial.TrialState
    try:
        for frozen in study.get_trials(deepcopy=False, states=(states.RUNNING,)):
            if frozen.number in numbers:
                study.tell(frozen.number, state=states.FAIL)
    except Exception as cleanup_error:  # noqa: BLE001 -- the error that ended the study stays the one the caller sees
        error.add_note(f"Unfinished Optuna trials could not be marked failed: {cleanup_error}")


def _study_lifecycle_summary(
    study: Any,
    *,
    config: OptunaStudyConfig,
    objective: Mapping[str, Any],
    requested_trials: int,
    reopened: bool,
    status: str,
    elapsed_seconds: float,
    include_importance: bool,
) -> dict[str, Any]:
    """Build one bounded study inventory from persisted Optuna trial facts."""
    trials = list(study.trials)
    counts = {
        "completed": sum(_trial_state_name(trial) == "complete" for trial in trials),
        "pruned": sum(_trial_state_name(trial) == "pruned" for trial in trials),
        "failed": sum(_trial_state_name(trial) == "fail" for trial in trials),
        "running": sum(_trial_state_name(trial) == "running" for trial in trials),
        "waiting": sum(_trial_state_name(trial) == "waiting" for trial in trials),
    }
    completed = [trial for trial in trials if _trial_state_name(trial) == "complete" and trial.value is not None]
    reverse = objective["direction"] == "maximize"
    best = sorted(completed, key=lambda item: float(item.value), reverse=reverse)[0] if completed else None
    inventory: list[dict[str, Any]] = []
    for trial in trials:
        started = getattr(trial, "datetime_start", None)
        finished = getattr(trial, "datetime_complete", None)
        duration = (finished - started).total_seconds() if started is not None and finished is not None else None
        attrs = getattr(trial, "user_attrs", {})
        inventory.append(
            {
                "number": int(trial.number),
                "state": _trial_state_name(trial),
                "value": trial.value,
                "params": copy.deepcopy(dict(getattr(trial, "params", {}))),
                "run_name": attrs.get("run_name") if isinstance(attrs, Mapping) else None,
                "training_seed": attrs.get("run_seed") if isinstance(attrs, Mapping) else None,
                "sampler_seed": attrs.get("sampler_seed") if isinstance(attrs, Mapping) else None,
                "duration_seconds": duration,
            }
        )

    importance: dict[str, Any]
    if include_importance and len(completed) >= _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS and any(trial.params for trial in completed):
        try:
            values = _optuna_module().importance.get_param_importances(study)
        except (ImportError, RuntimeError, TypeError, ValueError) as error:
            importance = {"status": "unavailable", "reason": type(error).__name__, "values": {}}
        else:
            importance = {"status": "computed", "minimum_completed_trials": _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS, "values": values}
    else:
        importance = {
            "status": "not_computed",
            "reason": "insufficient_completed_trials",
            "minimum_completed_trials": _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS,
            "values": {},
        }

    return {
        "status": status,
        "study_name": config.study["name"],
        "study_role": config.study["role"],
        "task": config.base_config["task"],
        "model_kind": config.base_config["model"]["kind"],
        "objective": copy.deepcopy(dict(objective)),
        "sampler": {**copy.deepcopy(config.study["sampler"]), "seed": _sampler_seed(config.study)},
        "pruner": copy.deepcopy(config.study["pruner"]),
        "training_seed": int(config.base_config["run"]["seed"]),
        "configured_trials": int(config.study["n_trials"]),
        "invocation_trials_requested": requested_trials,
        "continuation": "reopened" if reopened else "new",
        "attempted_trials": len(trials),
        **counts,
        "best_trial_number": int(best.number) if best is not None else None,
        "best_objective": float(best.value) if best is not None else None,
        "best_parameters": copy.deepcopy(dict(best.params)) if best is not None else {},
        "elapsed_seconds": elapsed_seconds,
        "parameter_importance": importance,
        "trials": inventory,
    }


def _publish_study_lifecycle_summary(paths: _OptunaStudyPaths, summary: Mapping[str, Any]) -> None:
    """Persist the bounded inventory only beside repository-owned local storage."""
    if paths.local_storage_path is not None:
        common.serialization.atomic_write_json(paths.study_dir / _STUDY_SUMMARY_FILENAME, dict(summary))


def _emit_study_summary(summary: Mapping[str, Any], *, storage: str, summary_file: str | None) -> None:
    """Emit one concise W&B-independent study terminal event."""
    experiments.console.optuna_study_event(
        str(summary["status"]),
        study=summary["study_name"],
        attempted_trials=summary["attempted_trials"],
        completed_trials=summary["completed"],
        pruned_trials=summary["pruned"],
        failed_trials=summary["failed"],
        best_trial=summary["best_trial_number"],
        best_objective=summary["best_objective"],
        best_parameters=summary["best_parameters"],
        duration_seconds=summary["elapsed_seconds"],
        storage=storage,
        summary_file=summary_file,
    )


def run_optuna_study(
    config: OptunaStudyConfig | Path | str,
    *,
    n_trials: int | None = None,
    device: str | None = None,
    output_root: Path | str | None = None,
) -> Any:
    """
    Run additional fresh trials in a new or semantically identical study.

    Parameters
    ----------
    config : OptunaStudyConfig | pathlib.Path | str
        Loaded study or YAML path.
    n_trials : int | None, optional
        Positive number of additional trials for this invocation.
    device : str | None, optional
        Runtime-only ``auto``, ``cpu``, or strict ``cuda`` override.
    output_root : pathlib.Path | str | None, optional
        Invocation-only study database and trial-output root.

    Returns
    -------
    Any
        Optimized Optuna study object.

    Raises
    ------
    ValueError
        If an existing study's direction, objective, or semantic signature differs.
    TypeError
        If the requested additional trial count is not an exact integer.

    Notes
    -----
    Reopening adds fresh trial numbers. It never resumes partial run directories.

    """
    study_config = load_optuna_study_config(config) if isinstance(config, (str, Path)) else config
    study_config = with_runtime_overrides(study_config, device=device, output_root=output_root)
    study_config, objective = _validate_study_contract(study_config)
    study_config = _bind_configured_dataset_identities(study_config)
    signature = build_study_signature(study_config)
    numerical_environment = learning.training.checkpoint.numerical_environment_identity()

    raw_trial_count = n_trials if n_trials is not None else study_config.study["n_trials"]
    if type(raw_trial_count) is not int:
        msg = f"Optuna n_trials must be an integer, got: {raw_trial_count!r}"
        raise TypeError(msg)
    trial_count = raw_trial_count
    if trial_count <= 0:
        msg = f"Optuna n_trials must be positive, got {trial_count}."
        raise ValueError(msg)

    study_paths = _resolve_study_paths(study_config)
    optuna = _optuna_module()
    pruner = _build_pruner(study_config.study)
    sampler = _build_sampler(study_config.study)
    study_name = common.paths.validate_logical_name(study_config.study["name"], label="study.name")
    if study_paths.local_storage_path is not None:
        study_paths.study_dir.mkdir(parents=True, exist_ok=True)
        common.paths.record_job_log_owner(study_paths.study_dir)

    study, reopened = _create_or_load_study(
        optuna=optuna,
        study_name=_database_study_id(study_config),
        direction=str(objective["direction"]),
        pruner=pruner,
        sampler=sampler,
        storage=study_paths.storage,
    )
    if reopened:
        # An existing study is continued in the tree it is bound to unless the job switched explicitly.
        experiments.source_binding.admit_continuation(
            experiments.source_binding.existing_study_run_dirs(study_config.base_config),
            study=study_paths.study_dir,
        )
        _validate_existing_study(
            study,
            signature=signature,
            objective=objective,
            numerical_environment=numerical_environment,
        )
    else:
        _publish_study_signature(
            study,
            signature,
            objective,
            numerical_environment=numerical_environment,
        )
    if study_paths.local_storage_path is not None:
        # Recorded before any trial, so a study that never reached a trial session keeps its tree.
        experiments.source_binding.record_study_session(study_paths.study_dir)

    _validate_reporting_contract(study_config.base_config)
    storage_label = _storage_console_location(study_paths)
    invocation_started = time.perf_counter()
    experiments.console.optuna_study_event(
        "started",
        study=study_name,
        task=study_config.base_config["task"],
        model_family=study_config.base_config["model"]["kind"],
        storage=storage_label,
        sampler=study_config.study["sampler"]["kind"],
        pruner=study_config.study["pruner"]["kind"],
        configured_trials=int(study_config.study["n_trials"]),
        invocation_trials=trial_count,
        maximum_epochs=int(study_config.base_config["training"]["epochs"]),
        continuation="reopened" if reopened else "new",
    )
    run_objective = create_objective(study_config)
    started_trials: list[int] = []

    def objective_of_invocation(trial: TrialProtocol) -> float:
        """Record which trials this invocation started before running one."""
        started_trials.append(trial.number)
        return run_objective(trial)

    try:
        study.optimize(
            objective_of_invocation,
            n_trials=trial_count,
            catch=(RecoverableTrialError,),
        )
    except BaseException as error:
        _fail_running_trials(study, started_trials, error)
        summary = _study_lifecycle_summary(
            study,
            config=study_config,
            objective=objective,
            requested_trials=trial_count,
            reopened=reopened,
            status="failed",
            elapsed_seconds=time.perf_counter() - invocation_started,
            include_importance=False,
        )
        _publish_study_lifecycle_summary(study_paths, summary)
        _emit_study_summary(
            summary,
            storage=storage_label,
            summary_file=_STUDY_SUMMARY_FILENAME if study_paths.local_storage_path is not None else None,
        )
        raise
    summary = _study_lifecycle_summary(
        study,
        config=study_config,
        objective=objective,
        requested_trials=trial_count,
        reopened=reopened,
        status="completed",
        elapsed_seconds=time.perf_counter() - invocation_started,
        include_importance=True,
    )
    _publish_study_lifecycle_summary(study_paths, summary)
    _emit_study_summary(
        summary,
        storage=storage_label,
        summary_file=_STUDY_SUMMARY_FILENAME if study_paths.local_storage_path is not None else None,
    )
    return study
