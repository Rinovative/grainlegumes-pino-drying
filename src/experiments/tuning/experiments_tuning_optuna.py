"""Run reusable Optuna studies and trial objectives.

Responsibilities:
  - Load strict study YAML with an inline experiment on the task base and a search space
  - Derive the trial budget as the study's fraction of the experiment's budget K; a transient study tunes its plan's A0 alone
  - Bind reusable studies to scientific semantic signatures, trial budget, effective sampler settings and Optuna version
    included, before optimization, and refuse to reopen one on another numerical environment or device model
  - Require a study that searches only physics values to equal its named supervised twin outside physics, k included
  - Initialize a study in repeatable steps under one initialization record, its configured base enqueued as its first
    trial, so a retry completes an interrupted creation; fail the trials an aborted invocation left RUNNING when the
    study is next opened
  - Admit one concrete device per invocation and refuse a trial that resolves another before its run exists
  - Run fresh numbered trials that report and prune at the eligible budget validation points, keyed by the point index
  - End a diverged trial pruned with a violated constraint TPE ranks last, and a trial without an eligible point pruned
  - Bind the same manifest identity and payload digests of either task, read by the one manifest reader, before the dry-run
    description or the study signature
  - Supply the trial status, reporting progress, failure context and best evidence of every terminal trial summary
  - Record the Slurm source binding of every invocation that creates or reopens a study, before its first trial
  - Classify non-finite, OOM, recoverable, interrupted, duplicate and unexpected outcomes explicitly
  - Report the study: its allocation's finished, pruned, failed and remaining trials, counts per trial status, the best
    value over completed trainings, the parameter importance over trained trials, the work of every trial and the
    seconds of all invocations
  - Declare the per-task HPO code projections: its own modules plus the training projection of the study's task

Design principles:
  - ``study.n_trials`` is the study's total allocation: reopening preserves history and each invocation runs only the
    fresh trials left, with a sampler seed derived from the study seed and the stored trial count, so a reopened study
    continues reproducibly instead of repeating; every trial records that seed and every invocation the count
  - Trials run one at a time, so TPE runs without ``constant_liar``; the code, not a config switch, fixes it
  - One invocation runs a study at a time, holding the study's lock from opening it until its summary is published
  - A repeated trial completes with its stored value but never counts as a training towards the pruner's startup or the
    parameter importance; its value, read from a completed training of the study, may be the study's best
  - Both tasks run their trials through the one driver and the shared prepared-run lifecycle
  - Embedded requests use the experiment structure of their task; wrappers own study/search policy
  - Wrapper objective and direction derive from the embedded resolved experiment objective
  - A non-eligible validation point gets neither a report nor a pruning decision, and no value is ever invented
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
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import yaml

from src import common, datasets, domain, experiments, learning
from src.experiments.config import experiments_config_naming as config_naming
from src.experiments.config import experiments_config_transient_plan as transient_plan

from . import experiments_tuning_search_space as search_space

_STUDY_SIGNATURE_ATTR = "semantic_signature"
_STUDY_SIGNATURE_PAYLOAD_ATTR = "semantic_signature_payload"
_RESOLVED_OBJECTIVE_ATTR = "resolved_objective"
_SAMPLER_METADATA_ATTR = "sampler_metadata"
_NUMERICAL_ENVIRONMENT_ATTR = "numerical_environment"
_STUDY_SUMMARY_FILENAME = "study_summary.json"
_BEST_CONFIG_FILENAME = "best_config.yaml"
# Tracking workflow, device and seeds belong to the study, not to the config its best trial yields.
_STUDY_ONLY_PATHS = frozenset({"tracking", "run"})
_PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS = 20
_STUDY_ROLES = frozenset({"production"})
_MODEL_PARAMS_PREFIX = "model.params."
_OOM_PRUNED_MESSAGE = "Trial pruned after allocator out-of-memory failure"
OPTUNA_ROOT_KEYS = frozenset({"study", "experiment", "search_space"})
_TRIAL_OUTCOMES = (
    "completed",
    "pruned",
    "diverged",
    "no_eligible_checkpoint",
    "nonfinite_pruned",
    "oom_pruned",
    "recoverable_failed",
    "failed",
    "interrupted",
    "duplicate",
    "duplicate_pruned",
)
# The violated constraint of a diverged trial; TPE ranks infeasible trials behind every feasible complete or pruned one.
_DIVERGENCE_CONSTRAINT = "diverged"
# Study user attribute prefix of every invocation's record: the stored trial count its sampler seed derives from and its
# wall seconds, ``None`` until that invocation ends.
_INVOCATIONS_ATTR = "invocation"
# Lock file in the study directory that one invocation holds from opening the study until its summary is published.
_STUDY_LOCK_FILENAME = "study.lock"
# The trial statuses whose value is the result of a completed training of the study: a duplicate carries its owner's stored value.
_BEST_STATUSES = frozenset({"completed", "duplicate"})
# Median-pruner settings the study policy fixes: pruning after 5 completed trials and only with 5 trials reported at the same point.
_MEDIAN_PRUNER_DEFAULTS = {"n_startup_trials": 5, "n_min_trials": 5, "interval_steps": 1}
# TPE settings the code fixes for every study: trials run one at a time, so no proposal steers away from a trial an aborted
# invocation left running, and the random startup states the library default instead of following a later release.
_TPE_FIXED_SETTINGS = {"constant_liar": False, "n_startup_trials": 10}
# Study user attribute recording the study's initialization: ``started`` with its signature digest, then ``complete``.
_INITIALIZATION_ATTR = "initialization"
# Study user attribute prefix of the record of a trial an aborted invocation left RUNNING and a later one failed.
_ABORTED_TRIAL_ATTR = "aborted_trial"
# Study key naming, relative to the study file, the supervised experiment a physics-weight study is the twin of.
_TWIN_KEY = "twin_of"
# Marks a study without the twin key, so an explicit null still counts as naming a twin.
_NO_TWIN = object()
_PHYSICS_PATH_PREFIX = "loss.physics."
# Trial user attribute naming the existing run a repeated trial answers from; a complete trial with it trained nothing.
_DUPLICATE_ATTR = "duplicate_of"
# The status a study report counts for a trial without a terminal status: running, or ended outside the trial driver.
_OPTUNA_STATE_STATUSES = {"complete": "completed", "pruned": "pruned", "fail": "failed", "running": "running", "waiting": "waiting"}
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
        exemptions={
            **_HPO_EXEMPTIONS,
            "domain/physics/domain_physics_brinkman.py": "steady mean-inlet-pressure reference of the shared scaling, only for steady encoding",
        },
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


class TrialProtocol(search_space.TrialLike, Protocol):
    """Minimal Optuna trial surface used by the objective: the search-space suggestions plus lifecycle calls."""

    number: int

    def report(self, value: float, step: int) -> None:
        """Report an intermediate metric value."""

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a serializable trial user attribute."""

    def set_constraint(self, key: str, value: float) -> None:
        """Set one constraint value; a positive value makes the trial infeasible."""

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
        Study-level settings such as name, objective, derived direction, n_trials, the
        budget fraction and the full budget K it applies to
    base_config : dict[str, Any]
        Trial experiment at the trial budget, resolved through config defaults
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
    base_config: dict[str, Any]
    search_space: tuple[search_space.SearchSpaceParameter, ...]
    dataset_identities: dict[str, Any] | None = None


@dataclass(frozen=True)
class _OptunaStudyPaths:
    """Pure resolved study paths shared by dry-run planning and execution; the study's SQLite database lives in its directory."""

    study_dir: Path
    trial_root: Path
    storage: str
    local_storage_path: Path


@dataclass
class OptunaPointReporter:
    """
    Report the held-out objective at the trial's eligible budget validation points, keyed by the fixed point index.

    Points may be missing (a group that crosses several validates once) but must increase.
    A non-eligible point (``validation/eligible`` 0) is recorded and gets neither a report
    nor a pruning decision, so the pruner never compares an invented value; an eligible
    point must carry its finite objective.
    """

    trial: TrialProtocol
    objective_id: str
    direction: str
    best_value: float | None = None
    best_point: int | None = None
    last_validation_point: int | None = None
    last_validation_epoch: int | None = None
    last_global_step: int | None = None
    last_reported_point: int | None = None
    last_reported_objective: float | None = None
    ineligible_points: list[int] = field(default_factory=list)

    def __call__(self, step: int, metrics: Mapping[str, float]) -> None:
        """Report one eligible validation point at its index and ask the pruner once; record a non-eligible point only."""
        if type(step) is not int or step <= 0:
            msg = f"Optuna reports require a positive integer global step, got {step!r}."
            raise TypeError(msg)
        raw_point = metrics.get("validation/point")
        point = int(raw_point) if raw_point is not None and float(raw_point).is_integer() else None
        if point is None or not 1 <= point <= learning.training.budget.VALIDATION_POINTS:
            msg = f"Optuna reports require a validation point between 1 and {learning.training.budget.VALIDATION_POINTS}, got {raw_point!r}."
            raise ValueError(msg)
        if self.last_validation_point is not None and point <= self.last_validation_point:
            msg = f"Optuna validation points must increase. Last {self.last_validation_point}, received {point}."
            raise ValueError(msg)
        eligible = metrics.get("validation/eligible")
        if eligible not in {0.0, 1.0}:
            msg = f"Validation point {point} must state validation/eligible as 0 or 1, got {eligible!r}."
            raise ValueError(msg)
        epoch = int(metrics["epoch"])
        self.last_validation_point, self.last_validation_epoch, self.last_global_step = point, epoch, step
        self.trial.set_user_attr("last_validation_point", point)
        self.trial.set_user_attr("last_global_step", step)
        if not eligible:
            self.ineligible_points.append(point)
            self.trial.set_user_attr("ineligible_points", list(self.ineligible_points))
            return
        value = metrics.get(f"id/{self.objective_id}")
        if value is None or not math.isfinite(float(value)):
            msg = f"Eligible validation point {point} (epoch {epoch}) must carry a finite id/{self.objective_id}, got {value!r}."
            raise RuntimeError(msg)
        value = float(value)
        if self.best_value is None or _is_better(value, self.best_value, self.direction):
            self.best_value = value
            self.best_point = point
        self.trial.report(value, step=point)
        self.trial.set_user_attr("last_reported_point", point)
        self.trial.set_user_attr("last_reported_objective", value)
        self.last_reported_point, self.last_reported_objective = point, value
        if self.trial.should_prune():
            msg = f"Pruned by Optuna at validation point {point} (epoch {epoch})"
            raise _trial_pruned_error()(msg)


@dataclass(frozen=True)
class _TrainedStartupPruner:
    """
    Start an Optuna pruner only after ``n_startup_trials`` complete trials that trained.

    A repeated trial completes with its owner's stored value, so TPE does not sample
    the point again, but it trained nothing and must not count as a training towards
    the pruner's startup. It reports no point, so the pruner's comparisons at a point
    never see it either.
    """

    n_startup_trials: int
    pruner: Any

    def prune(self, study: Any, trial: Any) -> bool:
        """Return the wrapped decision once enough complete trials trained, never before."""
        complete = study.get_trials(deepcopy=False, states=(_optuna_module().trial.TrialState.COMPLETE,))
        trained = sum(1 for completed in complete if _DUPLICATE_ATTR not in completed.user_attrs)
        return trained >= self.n_startup_trials and bool(self.pruner.prune(study, trial))


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


def _effective_sampler_settings(value: Any) -> dict[str, Any]:
    """Return every setting the sampler is built with: a TPE sampler states its multivariate flag and the fixed TPE settings."""
    sampler = _normalise_sampler_config(value)
    if sampler["kind"] != "tpe":
        return sampler
    return {"kind": "tpe", "multivariate": sampler.get("multivariate", False), **_TPE_FIXED_SETTINGS}


def _normalise_pruner_config(value: Any) -> dict[str, Any]:
    """
    Validate and isolate one exact semantic pruner mapping.

    Steps are validation points. A median pruner states its start point
    ``n_warmup_steps``, the first of the 20 points at which a trial can be pruned, and
    receives the policy defaults of 5 completed trials (``n_startup_trials``), 5 trials
    reported at the same point (``n_min_trials``) and a one-point interval; every value
    is an exact integer. The ``none`` pruner rejects tuning-only keys rather than
    silently ignoring them.
    """
    if isinstance(value, str):
        msg = "study.pruner must be a mapping with a semantic kind. String shorthand is unsupported."
        raise TypeError(msg)
    pruner = dict(copy.deepcopy(_as_mapping(value, label="study.pruner")))
    minimums = {"n_startup_trials": 0, "n_min_trials": 1, "n_warmup_steps": 0, "interval_steps": 1}
    unknown = sorted(set(pruner).difference({"kind", *minimums}))
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
    if kind == "none":
        invalid_keys = sorted(set(minimums).intersection(pruner))
        if invalid_keys:
            msg = f"study.pruner contains key(s) invalid for none: {invalid_keys}."
            raise ValueError(msg)
        return {"kind": kind}
    if "n_warmup_steps" not in pruner:
        msg = "study.pruner.n_warmup_steps, the validation point from which the median pruner may prune, is required."
        raise KeyError(msg)
    pruner = {**_MEDIAN_PRUNER_DEFAULTS, **pruner, "kind": kind}
    for key, minimum in minimums.items():
        pruner[key] = _require_exact_int(pruner[key], label=f"study.pruner.{key}", minimum=minimum)
    if pruner["n_warmup_steps"] >= learning.training.budget.VALIDATION_POINTS:
        msg = f"study.pruner.n_warmup_steps counts validation points and must be below {learning.training.budget.VALIDATION_POINTS}."
        raise ValueError(msg)
    return pruner


def _trial_budget_epochs(full_budget_epochs: Any, budget_fraction: Any) -> int:
    """
    Return a trial's budget in epochs: the study's fraction of the experiment's budget K.

    The fraction is an exact real in (0, 1]; the product must be a whole number of
    epochs, so no trial trains a silently rounded budget.
    """
    full = _require_exact_int(full_budget_epochs, label="the experiment's budget K in epochs", minimum=1)
    if isinstance(budget_fraction, bool) or not isinstance(budget_fraction, (int, float)) or not 0 < budget_fraction <= 1:
        msg = f"study.budget_fraction must be a number in (0, 1], got {budget_fraction!r}."
        raise ValueError(msg)
    epochs = full * budget_fraction
    if not float(epochs).is_integer():
        msg = f"study.budget_fraction {budget_fraction} of the budget K = {full} epochs is {epochs}; choose K so that the trial budget is whole."
        raise ValueError(msg)
    return int(epochs)


def _validate_study_settings(study: Mapping[str, Any]) -> None:
    """
    Revalidate the complete normalized study mapping without scalar coercion.

    Name, objective, direction, seed, additional-trial count, trial budget, sampler
    and pruner are checked with an exact closed schema. No file, database, Optuna
    import, or runtime allocation occurs.
    """
    allowed = {
        "name",
        "mode",
        "role",
        "objective",
        "direction",
        "seed",
        "n_trials",
        "budget_fraction",
        "full_budget_epochs",
        "sampler",
        "pruner",
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
    _trial_budget_epochs(study.get("full_budget_epochs"), study.get("budget_fraction"))
    _normalise_sampler_config(study.get("sampler"))
    pruner = _normalise_pruner_config(study.get("pruner"))
    if pruner["kind"] == "median" and int(pruner["n_startup_trials"]) >= int(study["n_trials"]):
        msg = "study.pruner.n_startup_trials must be smaller than study.n_trials."
        raise ValueError(msg)
    _sampler_seed(study)


def _normalise_study(raw_study: Mapping[str, Any], base_config: dict[str, Any], study_name: str, *, full_budget_epochs: int) -> dict[str, Any]:
    """
    Normalize study policy and derive objective identity from the base config.

    Defaults are applied to an isolated copy, direction cannot be supplied
    independently from the resolved experiment objective, and all names and
    scalar types are validated before an ``OptunaStudyConfig`` is constructed.
    The study name is the study config's file stem and cannot be authored; the
    budget K the authored ``budget_fraction`` applies to comes from the experiment.
    Storage cannot be authored: every study keeps its SQLite database in its own
    directory, where the study lock binds every invocation.
    """
    allowed = {"mode", "role", "seed", "n_trials", "budget_fraction", "sampler", "pruner"}
    unknown = sorted(set(raw_study).difference(allowed))
    if unknown:
        msg = f"study contains unknown key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        raise ValueError(msg)

    study = dict(copy.deepcopy(raw_study))
    study["name"] = study_name
    study["full_budget_epochs"] = full_budget_epochs
    study.setdefault("role", None)
    study.setdefault("mode", "standard")
    study.setdefault("n_trials", 30)
    study.setdefault("pruner", {"kind": "median"})
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
    _validate_study_settings(study)
    return study


def _trial_experiment(experiment: Mapping[str, Any], budget_fraction: Any) -> tuple[dict[str, Any], int]:
    """
    Return the authored experiment a trial trains and the experiment's budget K.

    A steady experiment trains ``budget_fraction`` of its ``training.budget_epochs``.
    A transient study's experiment is a matched plan on the task base; its trial is the
    plan's stage-A-only A0 over that fraction of ``training.a0.budget_epochs``, without B's budget.
    """
    transient = experiment.get("task") == "transient_drying"
    if transient and not transient_plan.is_matched_a1_b_config(experiment):
        message = "A transient study tunes the A0 of a matched plan: its experiment inherits the task base.yaml (base: ../base.yaml)."
        raise ValueError(message)
    training = _as_mapping(experiment.get("training"), label="experiment.training")
    budget_owner = _as_mapping(training.get("a0"), label="experiment.training.a0") if transient else training
    full_budget = _require_exact_int(budget_owner.get("budget_epochs"), label="the experiment's budget K in epochs", minimum=1)
    trial_budget = _trial_budget_epochs(full_budget, budget_fraction)
    if transient:
        return transient_plan.stage_a_only_experiment(experiment, budget_epochs=trial_budget), full_budget
    return experiments.config.loader.apply_epoch_budget(experiment, trial_budget), full_budget


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


def _validate_physics_twin(
    twin_of: Any,
    base_config: Mapping[str, Any],
    parameters: Sequence[search_space.SearchSpaceParameter],
    *,
    origin: Path,
    budget_fraction: Any,
    apply_overrides: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> None:
    """
    Require a study that searches only ``loss.physics`` values to equal its named supervised experiment outside physics.

    Such a study tunes physics on a fixed model, so it is the twin of a supervised experiment (``study.twin_of``,
    relative to the study file). That experiment, with the same overrides and trial budget, must be supervised and
    equal the study's experiment in every scientific value outside ``loss.physics``, k included. Any other study
    has no twin key, not even a null one (``twin_of`` is ``_NO_TWIN`` when the key is absent).
    """
    is_twin = all(parameter.path.startswith(_PHYSICS_PATH_PREFIX) for parameter in parameters)
    if twin_of is _NO_TWIN or not is_twin:
        if twin_of is not _NO_TWIN or is_twin:
            message = (
                "A study that searches only loss.physics values is a physics twin and must name its supervised experiment "
                f"in study.{_TWIN_KEY}; no other study names one."
            )
            raise ValueError(message)
        return
    path = origin.parent / _require_nonempty_string(twin_of, label=f"study.{_TWIN_KEY}")
    loader = experiments.config.loader
    supervised, _ = _trial_experiment(apply_overrides(loader.apply_authored_base(loader.load_yaml(path), origin=path)), budget_fraction)
    supervised = loader.resolve_config(supervised, source=config_naming.config_source_from_path(path))
    if supervised["loss"]["physics"]["enabled"]:
        message = f"study.{_TWIN_KEY} must name a supervised experiment, but {path} enables physics."
        raise ValueError(message)
    twin, reference = (learning.training.checkpoint.effective_config_identity_payload(config) for config in (base_config, supervised))
    twin["loss"], reference["loss"] = ({key: value for key, value in payload["loss"].items() if key != "physics"} for payload in (twin, reference))
    differing = sorted(key for key in twin.keys() | reference.keys() if twin.get(key) != reference.get(key))
    if differing:
        message = f"The physics twin differs from its supervised experiment {path} outside loss.physics in: {differing}."
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

    The inline experiment is deep-copied onto its task base, set to the trial budget
    (``study.budget_fraction`` of its budget K; a transient plan's A0 alone), and
    resolved first. Study objective and direction are
    then derived from that resolved config before exact parameter schemas,
    approved paths, supported choices, and base-value containment are validated.

    Parameters
    ----------
    path : pathlib.Path | str
        Optuna YAML containing only ``study``, ``experiment``, and
        ``search_space`` top-level blocks.
    epochs : int | None, optional
        Replace the experiment's budget K (steady ``training.budget_epochs``, transient
        ``training.a0.budget_epochs``) before the trial budget derives from it.
    train_dataset : str | None, optional
        Replace the experiment's exact training Dataset ID before resolution.
    revision : str | None, optional
        Label of a deliberately separate study location (``run.revision``).

    Returns
    -------
    OptunaStudyConfig
        Source path, normalized study policy with its budget K, resolved trial
        experiment, and parsed search parameters.

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

    # A physics twin's supervised experiment gets the same overrides, so both derive the same trial.
    apply_overrides = functools.partial(
        experiments.config.loader.apply_training_overrides, epochs=epochs, train_dataset=train_dataset, revision=revision
    )
    authored_experiment = apply_overrides(
        experiments.config.loader.apply_authored_base(_as_mapping(raw_mapping["experiment"], label="experiment"), origin=source_path)
    )
    raw_run = _as_mapping(authored_experiment.get("run"), label="experiment.run")
    if "seed" not in raw_run:
        msg = "experiment.run.seed must be explicit for an Optuna study."
        raise ValueError(msg)
    raw_study = dict(_as_mapping(raw_mapping.get("study", {}), label="study"))
    twin_of = raw_study.pop(_TWIN_KEY, _NO_TWIN)
    if "seed" not in raw_study:
        msg = "study.seed must be explicit for deterministic sampler reconstruction."
        raise ValueError(msg)
    study_name = study_source.name

    effective_experiment, full_budget_epochs = _trial_experiment(authored_experiment, raw_study.get("budget_fraction"))
    tracking = dict(_as_mapping(effective_experiment.get("tracking"), label="experiment.tracking"))
    wandb_settings = dict(_as_mapping(tracking.get("wandb"), label="experiment.tracking.wandb"))
    workflow = wandb_settings.get("workflow")
    if workflow != "optuna_trial":
        msg = "experiment.tracking.wandb.workflow must be 'optuna_trial' for an Optuna study."
        raise ValueError(msg)
    tracking["wandb"] = wandb_settings
    effective_experiment["tracking"] = tracking

    base_config = experiments.config.loader.resolve_config(effective_experiment, source=study_source)
    study = _normalise_study(raw_study, base_config, study_name, full_budget_epochs=full_budget_epochs)
    base_config = experiments.config.loader.validate_resolved_config(base_config)
    search_parameters = search_space.parse_search_space(raw_mapping["search_space"])
    search_space.validate_search_space_paths(base_config, search_parameters)
    _validate_search_space_choices(base_config, search_parameters)
    _validate_transient_study_policy(base_config, search_parameters)
    _validate_study_mode(base_config, mode=study["mode"])
    _validate_physics_twin(
        twin_of,
        base_config,
        search_parameters,
        origin=source_path,
        budget_fraction=study["budget_fraction"],
        apply_overrides=apply_overrides,
    )
    if not _task_base_path(base_config["task"]).is_file():
        msg = f"The task base {_task_base_path(base_config['task'])} that the best config is exported against is missing."
        raise ValueError(msg)

    return OptunaStudyConfig(
        path=source_path,
        study=study,
        base_config=base_config,
        search_space=search_parameters,
    )


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

    Study policy, base config, trial budget, search-path admission, and
    every declared candidate must agree. A defensive replacement containing the
    revalidated base config is returned with the canonical objective.
    """
    normalized_study = copy.deepcopy(config.study)
    normalized_study["role"] = _normalise_study_role(normalized_study.get("role"))
    _validate_study_settings(normalized_study)
    config = replace(config, study=normalized_study)
    base_config = experiments.config.loader.validate_resolved_config(config.base_config)
    _validate_study_mode(base_config, mode=config.study["mode"])
    trial_budget = _trial_budget_epochs(config.study["full_budget_epochs"], config.study["budget_fraction"])
    if base_config["training"]["budget_epochs"] != trial_budget:
        msg = f"The trial experiment trains {base_config['training']['budget_epochs']} epochs, not the study's trial budget of {trial_budget}."
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
        SHA-256 digest and canonical payload covering task,
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
    pruner = _normalise_pruner_config(validated.study["pruner"])
    sampler_seed = _sampler_seed(validated.study)
    if validated.dataset_identities is None:
        msg = "A study signature binds validated physical Dataset identities; bind the configured Datasets first."
        raise ValueError(msg)
    dataset_roles = {"binding": "validated_physical_packages", "roles": copy.deepcopy(validated.dataset_identities)}
    payload = {
        "task": {
            "id": validated.base_config["task"],
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
        "trial_budget": {
            "full_budget_epochs": validated.study["full_budget_epochs"],
            "budget_fraction": validated.study["budget_fraction"],
            "budget_epochs": validated.base_config["training"]["budget_epochs"],
        },
        "objective": objective,
        "direction": objective["direction"],
        # The Optuna release defines what the sampler and pruner settings do, so another release is another study.
        "sampler": {
            **_effective_sampler_settings(validated.study["sampler"]),
            "seed": sampler_seed,
            "invocation_seed": "derived_from_seed_and_stored_trial_count",
            "optuna_version": metadata.version("optuna"),
        },
        "pruner": pruner,
        "reporting": {
            "metric_source": "held_out_evaluation",
            "validation_points": learning.training.budget.VALIDATION_POINTS,
            "event_model": "budget_validation_points",
            "step": "validation_point_index",
            "non_eligible_point": "neither_reported_nor_pruned",
            "pruning_subset": "id_only",
        },
        "trial_lifecycle": {
            "outcomes": list(_TRIAL_OUTCOMES),
            "diverged": "pruned_with_violated_constraint",
            "no_eligible_checkpoint": "pruned_without_value",
            "duplicate": "complete_with_stored_value_outside_pruner_startup",
            "resume_policy": "new_trials_only",
            "orphaned_running_trial": "failed_on_reopen",
            "first_trial": "configured_base",
            "trial_count_policy": "n_trials_is_total_allocation",
            "training_seed_policy": "fixed_configured_run_seed_across_trials",
            "trial_identity_policy": "native_zero_based_optuna_number",
        },
        "hpo_implementation_sha256": _hpo_implementation_sha256(validated.base_config["task"]),
    }
    return {
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
    """Return the task-, dataset- (with its Train subset) and epoch-budget-owned study directory under the experiments root."""
    return common.paths.resolve_study_dir(
        str(config.base_config["task"]),
        experiments.config.loader.layout_dataset(config.base_config),
        str(config.study["name"]),
        config.base_config["training"]["budget_epochs"],
        config.base_config["run"].get("revision"),
        output_root=Path(config.base_config["paths"]["output_root"]),
    )


def _database_study_id(config: OptunaStudyConfig) -> str:
    """Return the storage-wide study identifier, unique across tasks, Datasets, epoch budgets and revisions."""
    return f"{config.base_config['task']}/{experiments.config.loader.layout_dataset(config.base_config)}/{_study_dir(config).name}"


def _resolve_study_paths(config: OptunaStudyConfig) -> _OptunaStudyPaths:
    """Resolve the study's SQLite database, study and trial roots without creating any path."""
    study_name = common.paths.validate_logical_name(config.study["name"], label="study.name")
    study_dir = _study_dir(config)
    local_storage_path = study_dir / f"{study_name}.db"
    return _OptunaStudyPaths(study_dir, study_dir / common.paths.STUDY_TRIALS_DIRNAME, f"sqlite:///{local_storage_path}", local_storage_path)


def _configured_dataset_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact configured Dataset selections without assigning package roles."""
    data = _as_mapping(config.get("data"), label="experiment.data")
    train_dataset = _require_nonempty_string(data.get("train_dataset"), label="data.train_dataset")
    diagnostic_key = "material_ood_datasets"
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
            "manifest_schema": {"kind": manifest["schema_kind"]},
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
        "material_ood": [summarize(dataset_id) for dataset_id in diagnostic_datasets],
    }


def _bind_configured_dataset_identities(config: OptunaStudyConfig) -> OptunaStudyConfig:
    """Bind current validated physical Dataset evidence to one study invocation."""
    identities = _configured_dataset_identities(config.base_config)
    return replace(config, dataset_identities=copy.deepcopy(identities))


def _build_pruner(study: Mapping[str, Any]) -> Any:
    """
    Build the exact Optuna pruner from validated semantic policy.

    Optuna is imported lazily only at construction. ``none`` maps to ``NopPruner``
    and ``median`` forwards the already normalized startup, minimum-comparison,
    start-point, and interval values; its startup counts only trials that trained.
    """
    pruner_cfg = _normalise_pruner_config(study.get("pruner"))
    optuna = _optuna_module()
    pruner_type = pruner_cfg["kind"]

    if pruner_type == "none":
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        median = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg["n_startup_trials"],
            n_warmup_steps=pruner_cfg["n_warmup_steps"],
            interval_steps=pruner_cfg["interval_steps"],
            n_min_trials=pruner_cfg["n_min_trials"],
        )
        return _TrainedStartupPruner(n_startup_trials=pruner_cfg["n_startup_trials"], pruner=median)
    msg = f"Unsupported Optuna pruner: {pruner_type!r}"
    raise ValueError(msg)


def _sampler_seed(study: Mapping[str, Any]) -> int:
    """Return the explicit persisted Optuna sampler seed without derivation."""
    seed = _require_exact_int(study.get("seed"), label="study.seed", minimum=0)
    if seed >= 2**32:
        msg = f"study.seed must fit Optuna/NumPy's 32-bit sampler domain, got {seed}."
        raise ValueError(msg)
    return seed


def _invocation_sampler_seed(study: Mapping[str, Any], stored_trials: int) -> int:
    """
    Return the sampler seed of one invocation, derived from the study seed and the trials the study stores when it opens.

    A reopened study thus continues with a new random stream instead of repeating its
    first invocation's draws, and the same sequence of invocations reproduces the same trials.
    """
    stored = _require_exact_int(stored_trials, label="stored Optuna trials", minimum=0)
    return common.random.derive_subseed(_sampler_seed(study), f"optuna_sampler_after_{stored}_trials") % 2**32


def _build_sampler(study: Mapping[str, Any], seed: int) -> Any:
    """Build the Optuna sampler of one invocation with its derived seed from the effective settings the signature binds; Optuna stays lazy."""
    settings = _effective_sampler_settings(study.get("sampler"))
    sampler_type = settings.pop("kind")
    optuna = _optuna_module()

    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=seed, **settings)
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    msg = f"Unsupported Optuna sampler: {sampler_type!r}"
    raise ValueError(msg)


def _prepare_trial_config(study_config: OptunaStudyConfig, trial: TrialProtocol, *, sampler_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one trial while preserving the configured fixed training seed; it records its invocation's derived sampler seed."""
    trial_number = _require_exact_int(trial.number, label="Optuna trial.number", minimum=0)
    overrides = search_space.suggest_trial_overrides(trial, study_config.search_space)
    config = search_space.apply_trial_overrides(study_config.base_config, overrides)
    training_seed = _require_exact_int(config["run"]["seed"], label="experiment.run.seed", minimum=0)
    sampler_seed = _require_exact_int(sampler_seed, label="invocation sampler seed", minimum=0)
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

    A mismatched full objective is a configuration/lifecycle error. Training selects
    only eligible points, whose objective is finite, so a NaN or infinite best metric
    is a lifecycle violation rather than a numerical outcome.
    """
    if result.get("objective") != objective:
        msg = "Training result objective does not match the resolved trial objective."
        raise ValueError(msg)
    objective_value = float(result["best_metric"])
    if not math.isfinite(objective_value):
        msg = "Training completed with a selected checkpoint but without a finite held-out objective."
        raise RuntimeError(msg)
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


def _failure_epoch(error: BaseException, reporter: OptunaPointReporter) -> int | None:
    """
    Recover an explicit failure epoch without inventing progress.

    A maintained ``" at epoch N:"`` message marker takes precedence. Otherwise
    only the epoch of the reporter's last validation point is returned.
    """
    marker = " at epoch "
    message = str(error)
    if marker in message:
        suffix = message.split(marker, 1)[1]
        digits = suffix.split(":", 1)[0].strip()
        if digits.isdigit():
            return int(digits)
    return reporter.last_validation_epoch


def _failure_context(
    *,
    error: BaseException,
    device_resolution: learning.device.DeviceResolution,
    reporter: OptunaPointReporter,
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
        "last_validation_point": reporter.last_validation_point,
        "last_reported_point": reporter.last_reported_point,
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
    reporter: OptunaPointReporter,
) -> dict[str, Any]:
    """
    Bind an unfinished trial's progress to its durable best checkpoint.

    Trials never resume and write no ``last_checkpoint.pt``, so no trial summary
    records one. The loop publishes a best before it reports that epoch, so best
    epoch, metric and digest come from ``best_checkpoint.pt`` whenever it exists,
    with its progress when that is later than the last validation point. A finite reported
    best without the file is a lifecycle violation rather than guessed or
    backfilled progress. Every key is returned, ``None`` when nothing was reached.
    """
    best_path = common.paths.resolve_best_checkpoint_file(run_dir)
    if reporter.best_point is not None and not best_path.is_file():
        msg = "A finite best Optuna objective must already have a durable best checkpoint."
        raise experiments.run.RunLifecycleError(msg)
    progress: dict[str, Any] = {
        **dict.fromkeys(("best_epoch", "best_metric", "best_checkpoint", "best_checkpoint_sha256")),
        "completed_epoch": reporter.last_validation_epoch,
        "global_step": reporter.last_global_step,
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
        if reporter.last_validation_epoch is None or payload["completed_epoch"] > reporter.last_validation_epoch:
            progress.update({"completed_epoch": payload["completed_epoch"], "global_step": payload["global_step"]})
    return progress


def _trial_terminal_evidence(
    status: str,
    outcome: Mapping[str, Any] | BaseException,
    *,
    run_dir: Path,
    reporter: OptunaPointReporter,
    objective: Mapping[str, Any],
    device_resolution: learning.device.DeviceResolution,
) -> dict[str, Any]:
    """
    Return the trial facts the shared run lifecycle merges into a terminal summary.

    A result with a selected checkpoint must agree with the reported points and carry
    a finite objective; a raise fails the run. A failure binds its bounded context and the
    best evidence of :func:`_checkpoint_progress` before the best file goes.
    """
    evidence: dict[str, Any] = {
        "last_validation_point": reporter.last_validation_point,
        "last_validation_epoch": reporter.last_validation_epoch,
        "last_reported_point": reporter.last_reported_point,
        "last_reported_objective": reporter.last_reported_objective,
        "ineligible_points": list(reporter.ineligible_points),
        "pruning_point": reporter.last_reported_point if status == "pruned" else None,
    }
    if isinstance(outcome, BaseException):
        failure_context = _failure_context(error=outcome, device_resolution=device_resolution, reporter=reporter)
        return {**evidence, "failure_context": failure_context, **_checkpoint_progress(run_dir, reporter)}
    _validate_completed_reporting(reporter, outcome)
    _finite_objective_value(outcome, objective)
    return {**evidence, "failure_context": None}


def _validate_completed_reporting(
    reporter: OptunaPointReporter,
    result: Mapping[str, Any],
) -> None:
    """
    Require the reported points to reach the terminal state the result represents.

    Training that completed its budget validated its budget-end point at its terminal
    epoch; training that diverged after an eligible point reported that point.
    """
    outcome = result["training_outcome"]["status"]
    if outcome == "completed":
        if reporter.last_validation_point != learning.training.budget.VALIDATION_POINTS or reporter.last_validation_epoch != result.get(
            "completed_epoch"
        ):
            msg = "Completed training did not report its budget-end validation point to Optuna."
            raise RuntimeError(msg)
    elif outcome != "diverged" or reporter.best_point is None:
        msg = f"A selected checkpoint of training outcome {outcome!r} needs an eligible validation point reported to Optuna."
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

    A trial of the same study (which fixes the epoch budget through its directory
    name) that completed its budget with the same objective supplies its stored value.
    Any other existing owner of the identity (another study of the same epoch budget,
    an unfinished, pruned, diverged or ineligible trial) supplies no value, and the
    trial is pruned; the repeat of a diverged trial of the same study gets the same
    violated constraint. The trial's digest includes its epoch budget, so a trial
    of another budget and an ordinary steady run (whose digest is epoch-free) own other
    identities; ``None`` means the identity is new.
    """
    entry = experiments.registry.lookup(config)
    if entry is None:
        return None
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve(strict=False)
    relative = entry.target.resolve(strict=False).relative_to(output_root).as_posix()
    trial.set_user_attr(_DUPLICATE_ATTR, relative)
    same_study = entry.kind == "trial" and entry.target.resolve(strict=False).parent == _trial_run_dir(config).resolve(strict=False).parent
    if not same_study:
        return None, relative
    summary = experiments.run.read_run_summary(entry.target)
    outcome = summary.get("training_outcome")
    outcome_status = outcome.get("status") if isinstance(outcome, Mapping) else None
    value = summary.get("best_metric")
    if (
        summary.get("status") == "completed"
        and outcome_status == "completed"
        and summary.get("objective") == dict(config["evaluation"]["objective"])
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value), relative
    if outcome_status == "diverged":
        trial.set_constraint(_DIVERGENCE_CONSTRAINT, 1.0)
    return None, relative


def _trained_trial_status(result: Mapping[str, Any], trial: TrialProtocol) -> tuple[str, str | None]:
    """
    Return the trial status of a trained result and, for an outcome without a value, its pruning message.

    A diverged trial is pruned, so the median pruner never compares its points, and gets
    a violated constraint, so TPE ranks it behind every other trial; a trial without an
    eligible point is pruned without a value.
    """
    outcome = result["training_outcome"]
    if outcome["status"] == "completed":
        return "completed", None
    if outcome["status"] == "diverged":
        trial.set_constraint(_DIVERGENCE_CONSTRAINT, 1.0)
        return "diverged", f"Trial diverged at {outcome['budget_progress']:.3f} of its budget: {outcome['divergence']}"
    if outcome["status"] == "no_eligible_checkpoint":
        return "no_eligible_checkpoint", "Trial reached no eligible validation point, so it has no value."
    message = f"A trial cannot end with training outcome {outcome['status']!r}."
    raise RuntimeError(message)


def _trial_work(run_dir: Path, units_per_update: int) -> dict[str, Any]:
    """Return the work a trained trial spent from its terminal run summary: optimizer updates, optimisation seconds and k.

    Updates are the committed training work and seconds the attempt's measured optimisation
    clock; each is ``None`` when the summary does not record it.
    """
    summary = experiments.run.read_run_summary(run_dir)
    work = summary.get("training_work")
    return {
        "optimizer_steps": work.get("optimizer_steps") if isinstance(work, Mapping) else None,
        "train_seconds": summary.get("attempted_train_seconds"),
        "units_per_update": units_per_update,
    }


def _resolve_admitted_device(policy: Any, admitted_device: Mapping[str, str], *, trial_number: int) -> learning.device.DeviceResolution:
    """Resolve a trial's device and refuse it before its run exists unless it is the device and model its invocation admitted."""
    resolution = learning.device.resolve_device(policy, path="run.device")
    resolved = _device_identity(resolution)
    if resolved != dict(admitted_device):
        message = f"Trial {trial_number} resolves {resolved}, not the device its invocation admitted {dict(admitted_device)}."
        raise RuntimeError(message)
    return resolution


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
    *,
    sampler_seed: int,
    admitted_device: Mapping[str, str],
) -> float:
    """
    Execute one trial of either task through the shared prepared-run lifecycle.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Validated study contract and runtime overrides.
    trial : TrialProtocol
        Optuna-compatible trial supplying suggestions, reports, pruning, and attrs.
    sampler_seed : int
        The seed the invocation derived for its sampler, which the trial records.
    admitted_device : Mapping[str, str]
        The concrete device and device model the invocation admitted; a trial that resolves another one is refused
        before its run exists.

    Returns
    -------
    float
        Finite terminal held-out value of the resolved primary objective.

    Raises
    ------
    optuna.TrialPruned
        For explicit pruning, divergence (with a violated constraint), training
        without an eligible validation point, a non-finite numerical failure,
        narrowly classified OOM, or a duplicate whose stored value cannot be reused.
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
    deleted; this driver sets ``terminal_status``, records a trained trial's
    ``work`` and prints the one ``trial`` line of every outcome. These closing
    steps and the cleanup each run even when another of them fails; their
    failures become notes on the trial's own propagating error, or the first is
    raised after a normal return.

    """
    started = time.monotonic()
    trial_pruned = _trial_pruned_error()
    returned = False
    status = "failed"
    value: float | None = None
    device: torch.device | None = None
    # Setup runs inside the guarded attempt, so a refused sample still ends with a status and one trial line.
    context: Mapping[str, Any] = {"trial_number": trial.number, "sampled_parameters": {}}
    reporter: OptunaPointReporter | None = None
    trained: tuple[Path, int] | None = None

    def attempt() -> float:
        """Answer a duplicate from stored evidence, or train the trial through the shared run lifecycle."""
        nonlocal status, device, context, reporter, trained
        validated_config, _ = _validate_study_contract(study_config)
        config, context = _prepare_trial_config(validated_config, trial, sampler_seed=sampler_seed)
        objective = experiments.config.loader.get_resolved_objective(config)
        point_reporter = reporter = OptunaPointReporter(
            trial=trial,
            objective_id=str(objective["id"]),
            direction=str(objective["direction"]),
        )

        def report(step: int, metrics: dict[str, float]) -> None:
            """Report each validation point to Optuna and expose the trial's best so far to W&B."""
            if "validation/point" not in metrics:
                return
            point_reporter(step, metrics)
            if point_reporter.best_value is not None:
                metrics["optuna/best_objective"] = point_reporter.best_value

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
        device_resolution = _resolve_admitted_device(config["run"]["device"], admitted_device, trial_number=trial.number)
        device = device_resolution.device
        experiments.run.validate_deterministic_model_device_policy(config, device_resolution)
        # Preparation and execution classify a failure of the trial's run like the trial itself.
        resolve_status = functools.partial(_trial_terminal_status, trial_pruned=trial_pruned)
        run_dir = experiments.run.prepare_fresh_run(
            config, run_dir=_trial_run_dir(config), summary_extra=context, terminal_status_resolver=resolve_status
        )
        trained = (run_dir, int(config["training"]["units_per_update"]))
        result = experiments.run.execute_prepared_run(
            config,
            run_dir=run_dir,
            persisted_config=config,
            event_callback=report,
            summary_extra=context,
            device_resolution=device_resolution,
            terminal_status_resolver=resolve_status,
            terminal_evidence=lambda terminal, outcome: _trial_terminal_evidence(
                terminal, outcome, run_dir=run_dir, reporter=point_reporter, objective=objective, device_resolution=device_resolution
            ),
        )
        status, pruned_message = _trained_trial_status(result, trial)
        if pruned_message is not None:
            raise trial_pruned(pruned_message)
        # The completed evidence admitted this value as finite before the run was published.
        return float(result["best_metric"])

    try:
        value = attempt()
    except BaseException as error:
        if status not in {"duplicate_pruned", "diverged", "no_eligible_checkpoint"}:
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
        # Each closing step runs even when an earlier one fails; their failures are settled after all of them.
        close_failures: list[tuple[str, BaseException]] = []
        try:
            trial.set_user_attr("terminal_status", status)
        except BaseException as error:  # noqa: BLE001 -- settled below, after the trial line and cleanup
            close_failures.append((f"Optuna attribute terminal_status={status!r} was not recorded", error))
        if trained is not None:
            try:
                trial.set_user_attr("work", _trial_work(*trained))
            except BaseException as error:  # noqa: BLE001 -- settled below, after the trial line and cleanup
                close_failures.append(("The trial's work was not recorded", error))
        try:
            experiments.console.optuna_trial_event(
                status,
                trial=int(context["trial_number"]),
                sampled=context["sampled_parameters"],
                duration_seconds=time.monotonic() - started,
                objective=value if value is not None or reporter is None else reporter.last_reported_objective,
                step=None if reporter is None else reporter.last_reported_point,
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


def create_objective(study_config: OptunaStudyConfig, *, sampler_seed: int, admitted_device: Mapping[str, str]) -> Callable[[TrialProtocol], float]:
    """
    Create an Optuna objective callable for a resolved study config.

    Parameters
    ----------
    study_config : OptunaStudyConfig
        Resolved Optuna study configuration
    sampler_seed : int
        The invocation's derived sampler seed, recorded by every trial
    admitted_device : Mapping[str, str]
        The concrete device and device model the invocation admitted, which every trial must resolve

    Returns
    -------
    Callable[[TrialProtocol], float]
        Objective function suitable for study.optimize

    """

    def objective(trial: TrialProtocol) -> float:
        """Run one trial for Optuna's optimize loop."""
        return run_trial(study_config, trial, sampler_seed=sampler_seed, admitted_device=admitted_device)

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

    Digest, canonical payload, and full objective are published
    together before optimization so later reopen admission can fail closed.
    """
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
    _require_direction(study, objective)
    environment = numerical_environment
    if environment is None:
        environment = learning.training.checkpoint.numerical_environment_identity()
    required = {
        _INITIALIZATION_ATTR: {"state": "complete", "signature": signature["digest"]},
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
    sampler_for: Callable[[int], Any],
    storage: str,
) -> tuple[Any, bool, int]:
    """
    Create a study or explicitly load the exact duplicate-name study; return it, whether it existed and its stored trials.

    Creation never uses ``load_if_exists``. Only Optuna's dedicated duplicate
    exception selects the load path, and the returned boolean requires callers
    to perform semantic-signature admission before optimization. ``sampler_for``
    builds the invocation's sampler from the number of trials the study stores.
    """
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
            sampler=sampler_for(0),
            storage=storage,
            load_if_exists=False,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            pruner=pruner,
        )
        # The stored trials are known only once the study is loaded; the sampler continues after them. A base an
        # interrupted creation queued has not run, so the retry draws as the uninterrupted creation would have.
        stored_trials = _allocated_trials(study.get_trials(deepcopy=False))
        study.sampler = sampler_for(stored_trials)
        return study, True, stored_trials
    return study, False, 0


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


def _fail_orphaned_trials(study: Any, trial_root: Path) -> None:
    """
    Mark FAIL every trial an aborted invocation left RUNNING, recording why and what its run directories hold.

    The caller holds the study's lock, so no invocation runs a trial of this study: a RUNNING trial was left by
    one that ended without finalizing it (node loss, SIGKILL), and TPE must not keep it as pending. The study
    attribute ``aborted_trial/<number>`` records the reason, the trial's directories and each run summary's status.
    """
    states = _optuna_module().trial.TrialState
    for frozen in study.get_trials(deepcopy=False, states=(states.RUNNING,)):
        directories = sorted(trial_root.glob(f"{frozen.number:04d}__*"))
        summaries: dict[str, Any] = {}
        for path in sorted(summary for directory in directories for summary in directory.rglob(common.paths.RUN_SUMMARY_FILENAME)):
            run = path.parent.relative_to(trial_root).as_posix()
            try:
                summaries[run] = experiments.run.read_run_summary(path.parent).get("status")
            except experiments.run.RunLifecycleError as error:
                summaries[run] = f"unreadable: {error}"
        record = {
            "reason": "left RUNNING by an aborted invocation; failed when the study was next opened",
            "trial_directories": [directory.name for directory in directories],
            "run_summaries": summaries,
        }
        study.set_user_attr(f"{_ABORTED_TRIAL_ATTR}/{frozen.number}", record)
        study.tell(frozen.number, state=states.FAIL)


def _base_trial_parameters(config: OptunaStudyConfig) -> dict[str, Any]:
    """Return the configured base value of every sampled parameter by its name: the study's first, counted trial."""
    parameters: dict[str, Any] = {}
    for parameter in config.search_space:
        if parameter.kind != "fixed":
            node: Any = config.base_config
            for token in parameter.path.split("."):
                node = node[int(token)] if isinstance(node, list) else node[token]
            parameters[parameter.name] = copy.deepcopy(node)
    return parameters


def _allocated_trials(trials: Sequence[Any]) -> int:
    """Return the trials that used the study's allocation: all but an enqueued trial no invocation started yet."""
    return sum(_trial_state_name(trial) != "waiting" for trial in trials)


def _device_identity(resolution: learning.device.DeviceResolution) -> dict[str, str]:
    """Return a resolved device and its model: the CUDA device's name, else the device type."""
    device = resolution.device
    return {"device": str(device), "device_model": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type}


def _require_direction(study: Any, objective: Mapping[str, Any]) -> None:
    """Refuse a stored study whose optimization direction is not the resolved objective's."""
    actual_direction = str(study.direction.name).lower()
    if actual_direction != objective["direction"]:
        msg = f"Existing Optuna study direction {actual_direction!r} does not match resolved objective direction {objective['direction']!r}."
        raise ValueError(msg)


def _initialize_study(
    study: Any,
    config: OptunaStudyConfig,
    *,
    signature: Mapping[str, Any],
    objective: Mapping[str, Any],
    numerical_environment: Mapping[str, Any],
) -> None:
    """
    Initialize a study under its lock, or complete an initialization an aborted invocation interrupted.

    The record ``initialization`` is written first as ``started`` with the signature digest; then the configured base
    is enqueued as the only trial, the semantic metadata is published and the record becomes ``complete``. Each step
    repeats safely, so a retry completes whatever a crash left. A study without the record must be empty (the crash
    fell between its creation and the record); one started for another signature, or holding anything but its
    queued base, is refused.
    """
    _require_direction(study, objective)
    started = {"state": "started", "signature": signature["digest"]}
    record = study.user_attrs.get(_INITIALIZATION_ATTR)
    trials = study.trials
    if record is None and (study.user_attrs or trials):
        msg = "Existing Optuna study has no initialization record but holds trials or attributes; it was not created by this driver."
        raise ValueError(msg)
    if record not in (None, started):
        msg = f"Existing Optuna study is an interrupted initialization of another signature (record {record!r}); it is refused."
        raise ValueError(msg)
    base = _base_trial_parameters(config)
    queued = [trial.system_attrs.get("fixed_params") if _trial_state_name(trial) == "waiting" else None for trial in trials]
    if queued not in ([], [base]):
        msg = "Existing Optuna study's interrupted initialization holds trials other than its queued configured base; it is refused."
        raise ValueError(msg)
    if record is None:
        study.set_user_attr(_INITIALIZATION_ATTR, started)
    if not trials:
        study.enqueue_trial(base)
    _publish_study_signature(study, signature, objective, numerical_environment=numerical_environment)
    study.set_user_attr(_INITIALIZATION_ATTR, {"state": "complete", "signature": signature["digest"]})


def _trial_status(trial: Any) -> str:
    """Return a trial's terminal status, or its Optuna state for a trial running or ended without one."""
    attrs = getattr(trial, "user_attrs", {})
    state = _trial_state_name(trial)
    if state != "running" and isinstance(attrs, Mapping) and "terminal_status" in attrs:
        return str(attrs["terminal_status"])
    return _OPTUNA_STATE_STATUSES.get(state, state)


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
    """
    Build one bounded study report from persisted Optuna trial and study facts.

    Trials are counted per terminal status, so diverged, ineligible and duplicate
    trials stay apart from completed and pruned ones. The best and the best so far
    after every trial come from completed trainings: a completed trial, or a duplicate,
    whose value is read from a completed training of the same study, so the best
    survives an owner whose Optuna trial crashed after its run was published; the
    earlier trial wins a tie. Parameter importance reads only the completed trials that
    trained. Each trained trial carries its work, and the study's seconds sum every
    ended invocation.
    """
    trials = sorted(study.trials, key=lambda item: int(item.number))
    states = [_trial_state_name(trial) for trial in trials]
    # The allocation counts trials by their Optuna state: a duplicate is finished, a diverged trial pruned, an aborted one failed.
    allocation = {
        "total": int(config.study["n_trials"]),
        "finished": states.count("complete"),
        "pruned": states.count("pruned"),
        "failed": states.count("fail"),
        "remaining": max(int(config.study["n_trials"]) - _allocated_trials(trials), 0),
    }
    statuses = [_trial_status(trial) for trial in trials]
    counts = dict.fromkeys((*_TRIAL_OUTCOMES, *_OPTUNA_STATE_STATUSES.values()), 0)
    for trial_status in statuses:
        counts[trial_status] = counts.get(trial_status, 0) + 1
    best: Any = None
    best_value: float | None = None
    inventory: list[dict[str, Any]] = []
    trained = [
        trial
        for trial, trial_status in zip(trials, statuses, strict=True)
        if trial_status == "completed" and trial.value is not None and _DUPLICATE_ATTR not in trial.user_attrs
    ]
    for trial, trial_status in zip(trials, statuses, strict=True):
        value = None if trial.value is None else float(trial.value)
        if trial_status in _BEST_STATUSES and value is not None and (best_value is None or _is_better(value, best_value, objective["direction"])):
            best, best_value = trial, value
        started = getattr(trial, "datetime_start", None)
        finished = getattr(trial, "datetime_complete", None)
        duration = (finished - started).total_seconds() if started is not None and finished is not None else None
        attrs = getattr(trial, "user_attrs", {})
        attrs = attrs if isinstance(attrs, Mapping) else {}
        inventory.append(
            {
                "number": int(trial.number),
                "state": _trial_state_name(trial),
                "status": trial_status,
                "value": trial.value,
                "best_objective_so_far": best_value,
                "params": copy.deepcopy(dict(getattr(trial, "params", {}))),
                "run_name": attrs.get("run_name"),
                "training_seed": attrs.get("run_seed"),
                "sampler_seed": attrs.get("sampler_seed"),
                "work": copy.deepcopy(attrs.get("work")),
                "duration_seconds": duration,
            }
        )
    invocations = [record["seconds"] for key, record in study.user_attrs.items() if key.startswith(f"{_INVOCATIONS_ATTR}/")]

    importance: dict[str, Any]
    if include_importance and len(trained) >= _PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS and any(trial.params for trial in trained):
        optuna = _optuna_module()
        try:
            # An in-memory view of the trained trials only, so a duplicate's repeated point gets no extra weight.
            view = optuna.create_study(direction=study.direction)
            view.add_trials(trained)
            values = optuna.importance.get_param_importances(view)
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
        "sampler": {**_effective_sampler_settings(config.study["sampler"]), "seed": _sampler_seed(config.study)},
        "pruner": copy.deepcopy(config.study["pruner"]),
        "training_seed": int(config.base_config["run"]["seed"]),
        "full_budget_epochs": config.study["full_budget_epochs"],
        "budget_fraction": config.study["budget_fraction"],
        "trial_budget_epochs": config.base_config["training"]["budget_epochs"],
        "allocation": allocation,
        "invocation_trials_requested": requested_trials,
        "continuation": "reopened" if reopened else "new",
        "attempted_trials": len(trials),
        "trial_counts": counts,
        "best_trial_number": int(best.number) if best is not None else None,
        "best_objective": best_value,
        "best_parameters": copy.deepcopy(dict(best.params)) if best is not None else {},
        "elapsed_seconds": elapsed_seconds,
        # An invocation still running, or one that ended without recording its seconds, is counted, never summed as zero.
        "study_seconds": math.fsum(seconds for seconds in invocations if seconds is not None),
        "unrecorded_invocations": sum(seconds is None for seconds in invocations),
        "parameter_importance": importance,
        "trials": inventory,
    }


def _task_base_path(task: str) -> Path:
    """Return the maintained ``base.yaml`` of one task, wherever a study file lives."""
    return Path(common.paths.get_project_root(), *config_naming.CONFIG_TREE, task, experiments.config.loader.AUTHORED_BASE_FILENAME)


def _authored_difference(
    merged: Mapping[str, Any], base: Mapping[str, Any], search_paths: Sequence[str], *, prefix: str = "", study_only: bool = False
) -> dict[str, Any]:
    """Return the leaves of ``merged`` that differ from ``base``; study-only sections keep only the searched paths."""
    result: dict[str, Any] = {}
    for key, value in merged.items():
        path = f"{prefix}.{key}" if prefix else key
        only = study_only or path in _STUDY_ONLY_PATHS
        base_value = base.get(key)
        if isinstance(value, Mapping):
            nested = _authored_difference(value, base_value if isinstance(base_value, Mapping) else {}, search_paths, prefix=path, study_only=only)
            if nested:
                result[key] = nested
        elif value != base_value or key not in base:
            if not only or any(searched == path or searched.startswith(f"{path}.") for searched in search_paths):
                result[key] = value
    return result


def _best_experiment_yaml(config: OptunaStudyConfig, summary: Mapping[str, Any], study_name: str) -> str:
    """Return the best trial and the study's K as an experiment config on the task's ``base.yaml`` that trains once copied to ``experiments/``."""
    overrides = {
        parameter.path: parameter.value if parameter.kind == "fixed" else summary["best_parameters"][parameter.name]
        for parameter in config.search_space
    }
    authored = experiments.config.loader.load_yaml(config.path)["experiment"]
    # The trials derive from the study's K, an --epochs override included, so the export states that K wherever it differs from the base.
    merged = experiments.config.loader.apply_epoch_budget(
        experiments.config.loader.apply_authored_base(authored, origin=config.path), config.study["full_budget_epochs"]
    )
    for path, value in overrides.items():
        # An omitted default is absent from the authored block: its first missing level is copied from the resolved config.
        authored_node: Any = merged
        resolved_node: Any = config.base_config
        for token in path.split("."):
            key: Any = int(token) if isinstance(authored_node, list) else token
            if isinstance(authored_node, dict) and key not in authored_node:
                authored_node[key] = copy.deepcopy(resolved_node[key])
                break
            authored_node, resolved_node = authored_node[key], resolved_node[int(token) if isinstance(resolved_node, list) else token]
        search_space.set_config_path(merged, path, value)
    base = experiments.config.loader.load_yaml(_task_base_path(config.base_config["task"]))
    task = merged.pop("task")
    best_config = search_space.apply_trial_overrides(config.base_config, overrides)
    # A transient study's experiment is its matched plan, so its differences already state A0's sampling under its stage.
    differences = config_naming.complete_token_groups(_authored_difference(merged, base, list(overrides)), best_config)
    body = {"task": task, "base": "../../base.yaml", **experiments.config.loader.ordered_sections(differences)}
    label = config_naming.config_name(best_config, differences)
    best = f"# Best trial {summary['best_trial_number']} of study {study_name}, objective {summary['best_objective']:.6g}; "
    if config.study["mode"] == "stage_a_only":
        header = (
            f"{best}transfer these values into the matched experiment under configs/03_learning/{task}/experiments/architecture_comparison/, "
            "which carries B's sampling and the B and A1 memory batches; the name is "
            f"{label}; B/A1 settings it sets appear in the pair folder (configs/README.md).\n"
        )
    else:
        header = f"{best}copy to configs/03_learning/{task}/experiments/<group>/{label}.yaml to train it.\n"
    return header + yaml.dump(body, default_flow_style=False, sort_keys=False)


def _publish_study_lifecycle_summary(paths: _OptunaStudyPaths, summary: Mapping[str, Any], config: OptunaStudyConfig) -> None:
    """Persist the best trial's config, then the summary as the commit record, in the study directory.

    The best config is built before anything is written, so a build error leaves both files untouched.
    """
    best_path = paths.study_dir / _BEST_CONFIG_FILENAME
    if summary["best_trial_number"] is None:
        best_path.unlink(missing_ok=True)
    else:
        common.serialization.atomic_write_text(best_path, _best_experiment_yaml(config, summary, paths.study_dir.name))
    common.serialization.atomic_write_json(paths.study_dir / _STUDY_SUMMARY_FILENAME, dict(summary))


def _emit_study_summary(summary: Mapping[str, Any], *, storage: str) -> None:
    """Emit one concise W&B-independent study terminal event naming the study's database and summary file."""
    experiments.console.optuna_study_event(
        str(summary["status"]),
        study=summary["study_name"],
        attempted_trials=summary["attempted_trials"],
        trial_counts=summary["trial_counts"],
        best_trial=summary["best_trial_number"],
        best_objective=summary["best_objective"],
        best_parameters=summary["best_parameters"],
        duration_seconds=summary["elapsed_seconds"],
        study_seconds=summary["study_seconds"],
        storage=storage,
        summary_file=_STUDY_SUMMARY_FILENAME,
    )


def _run_study_invocation(
    study_config: OptunaStudyConfig,
    *,
    objective: Mapping[str, Any],
    signature: Mapping[str, Any],
    numerical_environment: Mapping[str, Any],
    admitted_device: Mapping[str, str],
    study_paths: _OptunaStudyPaths,
    trial_limit: int | None,
) -> Any:
    """
    Open or create the study, run this invocation's trials and publish the summary; the caller holds the study's lock.

    A study without a complete initialization record is initialized, or its interrupted initialization completed
    (:func:`_initialize_study`); an initialized one is admitted and fails the trials an aborted invocation left
    RUNNING. The invocation then runs what is left of ``study.n_trials``, at most ``trial_limit``, each trial on
    the admitted device.
    """
    optuna = _optuna_module()
    pruner = _build_pruner(study_config.study)
    study_name = common.paths.validate_logical_name(study_config.study["name"], label="study.name")
    study_paths.study_dir.mkdir(parents=True, exist_ok=True)
    common.paths.record_job_log_owner(study_paths.study_dir)

    def sampler_for(stored_trials: int) -> Any:
        """Build the invocation's sampler with the seed derived from the trials the study stores."""
        return _build_sampler(study_config.study, _invocation_sampler_seed(study_config.study, stored_trials))

    study, reopened, stored_trials = _create_or_load_study(
        optuna=optuna,
        study_name=_database_study_id(study_config),
        direction=str(objective["direction"]),
        pruner=pruner,
        sampler_for=sampler_for,
        storage=study_paths.storage,
    )
    sampler_seed = _invocation_sampler_seed(study_config.study, stored_trials)
    if reopened:
        # An existing study is continued in the tree it is bound to unless the job switched explicitly.
        experiments.source_binding.admit_continuation(
            experiments.source_binding.existing_study_run_dirs(study_config.base_config),
            study=study_paths.study_dir,
        )
    record = study.user_attrs.get(_INITIALIZATION_ATTR)
    if isinstance(record, Mapping) and record.get("state") == "complete":
        _validate_existing_study(
            study,
            signature=signature,
            objective=objective,
            numerical_environment=numerical_environment,
        )
        _fail_orphaned_trials(study, study_paths.trial_root)
    else:
        _initialize_study(study, study_config, signature=signature, objective=objective, numerical_environment=numerical_environment)
    # Recorded before any trial, so a study that never reached a trial session keeps its tree.
    experiments.source_binding.record_study_session(study_paths.study_dir)
    remaining = max(int(study_config.study["n_trials"]) - _allocated_trials(study.trials), 0)
    trial_count = remaining if trial_limit is None else min(trial_limit, remaining)

    storage_label = str(study_paths.local_storage_path)
    # One record per invocation: the stored trial count its sampler seed derives from, and its seconds once it ends.
    invocation_key = f"{_INVOCATIONS_ATTR}/{uuid.uuid4().hex}"
    study.set_user_attr(invocation_key, {"stored_trials": stored_trials, "seconds": None})
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
        budget_epochs=int(study_config.base_config["training"]["budget_epochs"]),
        full_budget_epochs=int(study_config.study["full_budget_epochs"]),
        continuation="reopened" if reopened else "new",
    )
    run_objective = create_objective(study_config, sampler_seed=sampler_seed, admitted_device=admitted_device)
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
        elapsed = time.perf_counter() - invocation_started
        _fail_running_trials(study, started_trials, error)
        try:
            study.set_user_attr(invocation_key, {"stored_trials": stored_trials, "seconds": elapsed})
        except Exception as record_error:  # noqa: BLE001 -- the error that ended the study stays the one the caller sees
            error.add_note(f"The invocation's seconds were not recorded: {record_error}")
        summary = _study_lifecycle_summary(
            study,
            config=study_config,
            objective=objective,
            requested_trials=trial_count,
            reopened=reopened,
            status="failed",
            elapsed_seconds=elapsed,
            include_importance=False,
        )
        try:
            _publish_study_lifecycle_summary(study_paths, summary, study_config)
        except Exception as publication_error:  # noqa: BLE001 -- the original study error must reach the caller
            sys.stderr.write(f"Warning: the study summary was not published after the study failed: {publication_error}\n")
        _emit_study_summary(summary, storage=storage_label)
        raise
    elapsed = time.perf_counter() - invocation_started
    study.set_user_attr(invocation_key, {"stored_trials": stored_trials, "seconds": elapsed})
    summary = _study_lifecycle_summary(
        study,
        config=study_config,
        objective=objective,
        requested_trials=trial_count,
        reopened=reopened,
        status="completed",
        elapsed_seconds=elapsed,
        include_importance=True,
    )
    _publish_study_lifecycle_summary(study_paths, summary, study_config)
    _emit_study_summary(summary, storage=storage_label)
    return study


def run_optuna_study(
    config: OptunaStudyConfig | Path | str,
    *,
    n_trials: int | None = None,
    device: str | None = None,
    output_root: Path | str | None = None,
) -> Any:
    """
    Run the fresh trials left of the study's allocation in a new or semantically identical study.

    Parameters
    ----------
    config : OptunaStudyConfig | pathlib.Path | str
        Loaded study or YAML path.
    n_trials : int | None, optional
        Positive limit on this invocation's trials; it runs at most the trials left of ``study.n_trials``, the
        study's total allocation, and all of them when omitted.
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
        If an existing study's direction, objective, semantic signature or numerical environment (device model
        included) differs.
    TypeError
        If the trial limit is not an exact integer.
    RuntimeError
        If another invocation runs the same study.

    Notes
    -----
    Reopening adds fresh trial numbers. It never resumes partial run directories.
    A trial an aborted invocation left RUNNING is failed when the study is next opened.
    One invocation runs a study at a time: it holds the study's lock from opening
    the study and counting its trials until its summary is published, and a second
    invocation of the same study is refused before it opens the study.

    """
    study_config = load_optuna_study_config(config) if isinstance(config, (str, Path)) else config
    study_config = with_runtime_overrides(study_config, device=device, output_root=output_root)
    study_config, objective = _validate_study_contract(study_config)
    study_config = _bind_configured_dataset_identities(study_config)
    signature = build_study_signature(study_config)
    # The invocation admits one concrete device that every trial must resolve; its model is bound with the numerical
    # environment, so a study runs on one GPU type.
    admitted_device = _device_identity(learning.device.resolve_device(study_config.base_config["run"]["device"], path="run.device"))
    numerical_environment = {
        **learning.training.checkpoint.numerical_environment_identity(),
        "device_model": admitted_device["device_model"],
    }

    if n_trials is not None and type(n_trials) is not int:
        msg = f"Optuna n_trials must be an integer, got: {n_trials!r}"
        raise TypeError(msg)
    if n_trials is not None and n_trials <= 0:
        msg = f"Optuna n_trials must be positive, got {n_trials}."
        raise ValueError(msg)

    study_paths = _resolve_study_paths(study_config)
    study_name = common.paths.validate_logical_name(study_config.study["name"], label="study.name")
    # The study's database lives in its directory, so the lock there binds every invocation of the study.
    with ExitStack() as invocation:
        try:
            invocation.enter_context(common.locking.exclusive_file_lock(study_paths.study_dir / _STUDY_LOCK_FILENAME, blocking=False))
        except common.locking.FileLockUnavailableError as error:
            msg = (
                f"Optuna study {study_name!r} is already run by another invocation ({study_paths.study_dir}); "
                "one invocation runs a study at a time, so start this one after it ends."
            )
            raise RuntimeError(msg) from error
        return _run_study_invocation(
            study_config,
            objective=objective,
            signature=signature,
            numerical_environment=numerical_environment,
            admitted_device=admitted_device,
            study_paths=study_paths,
            trial_limit=n_trials,
        )
