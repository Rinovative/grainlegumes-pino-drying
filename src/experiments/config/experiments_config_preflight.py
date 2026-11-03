"""Classify and validate executable experiment, saved-resume, and Optuna configs.

Responsibilities:
  - Identify workflow families from strict root-schema ownership
  - Resolve single-stage, matched-comparison, and Optuna configurations through their owners
  - Validate an exact run-owned resolved config without applying authored overrides
  - Return typed task, model, physics, run-label, and source-path summaries
  - Report the existing run leaves the requested job may write and the frozen source tree its unit last ran in
  - Reject workflow mismatches with actionable commands and matching study suggestions

Design principles:
  - Mixed schema ownership fails before workflow dispatch
  - Semantic admission delegates to the maintained configuration and study loaders
  - Inspection allocates no GPU, scheduler, run, or study state
  - Source-tree discovery reads saved summaries only after the workflow is admitted, never for suggestion candidates
  - Only leaves the job may write bind a tree: completed matched stages and unselected stages are read-only prerequisites
  - Saved-config resume checks structure only; the scientific implementation that runs it is known once the binding is

This module does NOT:
  - Submit the suggested commands or execute training and optimization
  - Define model, loss, temporal, or matched-budget semantics
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import common, experiments

from . import experiments_config_loader as loader
from . import experiments_config_naming as config_naming
from . import experiments_config_transient_plan as transient_plan

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPERIMENT_FAMILY = "experiment"
OPTUNA_FAMILY = "optuna"
_MATCHED_STAGES: tuple[transient_plan.MatchedStageName, ...] = ("a0", "b", "a1")
_WORKFLOW_FAMILY = {"train": EXPERIMENT_FAMILY, "optuna": OPTUNA_FAMILY}


@dataclass(frozen=True, slots=True)
class ConfigPreflight:
    """Describe one strictly validated executable config without runtime allocation.

    ``unit_run_dirs`` are the existing run leaves the request would write into: a
    steady run or the one resumed stage, or the selected stages of a matched plan.
    ``completed_leaves_read_only`` marks matched plans, whose completed stages are
    read-only prerequisites. ``unit_study_dir`` is an Optuna study's directory.
    ``bound_source_tree`` is the ``<commit12>-<fingerprint12>`` tree the unit last
    ran in and is set by :func:`validate_workflow` only.
    """

    family: str
    task: str
    model_kind: str
    physics_enabled: bool
    run_label: str
    source_path: Path
    canonical_path: str
    unit_run_dirs: tuple[Path, ...] = ()
    completed_leaves_read_only: bool = False
    unit_study_dir: Path | None = None
    bound_source_tree: str | None = None


class WorkflowMismatchError(ValueError):
    """Report a valid config supplied to the wrong executable workflow."""

    def __init__(self, message: str, *, result: ConfigPreflight) -> None:
        """Initialize the mismatch with its validated supplied-config result."""
        super().__init__(message)
        self.result = result


def _canonical_config_path(path: Path) -> str:
    """Return a stable repository-relative config path when one is available."""
    source = path.expanduser().resolve()
    project_root = common.paths.get_project_root().expanduser().resolve()
    try:
        return source.relative_to(project_root).as_posix()
    except ValueError:
        return str(source)


def _classify_root(raw: dict[str, object]) -> str:
    """Classify strict schema ownership from parsed root keys, never path names."""
    from src.experiments.tuning import experiments_tuning_optuna as optuna  # noqa: PLC0415

    keys = set(raw)
    experiment_keys = keys.intersection(loader.EXPERIMENT_ROOT_KEYS)
    optuna_keys = keys.intersection(optuna.OPTUNA_ROOT_KEYS)
    if experiment_keys and optuna_keys:
        msg = (
            "Configuration root mixes normal experiment and Optuna wrapper sections. "
            f"Experiment-owned={sorted(experiment_keys)}, optuna-owned={sorted(optuna_keys)}."
        )
        raise loader.ConfigError(msg)
    if optuna_keys:
        return OPTUNA_FAMILY
    if experiment_keys:
        return EXPERIMENT_FAMILY
    msg = f"Configuration family could not be identified from strict root ownership. Found root keys: {sorted(keys)}."
    raise loader.ConfigError(msg)


def _saved_resume_config_path(source: Path, resume: Path | str | None) -> Path | None:
    """Identify an exact run-owned config path and validate its storage boundary."""
    if resume is None:
        return None
    source_path = source.resolve(strict=True)
    run_dir = Path(resume).expanduser().resolve(strict=False)
    if source_path != (run_dir / common.paths.RUN_CONFIG_FILENAME).resolve(strict=False):
        return None
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir():
        msg = f"Resume target is not a run directory: {run_dir}"
        raise loader.ConfigError(msg)
    storage_root = common.paths.get_storage_root().expanduser().resolve(strict=False)
    if not run_dir.is_relative_to(storage_root):
        msg = f"Saved-effective resume must remain below durable storage: {run_dir}"
        raise loader.ConfigError(msg)
    saved_config = (run_dir / common.paths.RUN_CONFIG_FILENAME).resolve(strict=True)
    if saved_config != source_path or not saved_config.is_file():
        msg = f"Saved-effective resume config must be the exact run-owned {common.paths.RUN_CONFIG_FILENAME}: {source_path}"
        raise loader.ConfigError(msg)
    return saved_config


def _selected_stages(train_stages: tuple[str, ...] | None) -> tuple[transient_plan.MatchedStageName, ...]:
    """Return the selected matched stages in operational order; ``None`` selects all."""
    if train_stages is None:
        return _MATCHED_STAGES
    if not train_stages or len(set(train_stages)) != len(train_stages) or not set(train_stages) <= set(_MATCHED_STAGES):
        msg = "--train-stages requires distinct a0,b,a1 tokens in a comma-separated list."
        raise loader.ConfigError(msg)
    return tuple(name for name in _MATCHED_STAGES if name in train_stages)


def inspect_config(
    path: Path | str,
    *,
    epochs: int | None = None,
    train_dataset: str | None = None,
    train_seconds: float | None = None,
    rollout_length: int | None = None,
    lr_scale: float | None = None,
    revision: str | None = None,
    resume: Path | str | None = None,
    train_stages: tuple[str, ...] | None = None,
) -> ConfigPreflight:
    """Classify and validate one executable config without allocating runtime state.

    An exact ``resume/config.yaml`` source is admitted as saved effective state;
    authored semantic overrides are rejected for that path, and only its structure
    is validated because the implementation that will run it is chosen afterwards.
    ``train_stages`` selects the matched stages whose leaves the job may write.
    """
    source = Path(path).expanduser()
    raw = loader.load_authored_config(source)
    family = _classify_root(raw)
    unit_run_dirs: tuple[Path, ...]
    completed_leaves_read_only = False
    unit_study_dir: Path | None = None
    if family == EXPERIMENT_FAMILY:
        saved_resume_config = _saved_resume_config_path(source, resume) if resume is not None else None
        resolved: Mapping[str, Any]
        if saved_resume_config is not None and resume is not None:
            if any(value is not None for value in (epochs, train_dataset, train_seconds, rollout_length, lr_scale, revision)):
                msg = "Saved-effective resume config does not accept semantic overrides; resume the saved config unchanged."
                raise loader.ConfigError(msg)
            raw = loader.validate_resolved_config(raw, verify_layout_identity=False)
            stage = raw["run"]["layout"]["stage"]
            if train_stages is not None and train_stages != (stage,):
                msg = "Saved-effective resume selects only its exact saved stage."
                raise loader.ConfigError(msg)
            if stage != "single":
                common.paths.resolve_config_output_root(raw, Path(resume).expanduser().absolute())
            resolved = raw
            unit_run_dirs = (saved_resume_config.parent,)
        else:
            raw = loader.apply_training_overrides(
                raw,
                epochs=epochs,
                train_dataset=train_dataset,
                train_seconds=train_seconds,
                rollout_length=rollout_length,
                lr_scale=lr_scale,
                revision=revision,
            )
            config_source = config_naming.config_source_from_path(source)
            if transient_plan.is_matched_a1_b_config(raw):
                plan = transient_plan.resolve_matched_a1_b_plan(raw, source=config_source)
                resolved = plan.a0
                unit_run_dirs = experiments.source_binding.existing_train_run_dirs(plan.stage(name) for name in _selected_stages(train_stages))
                completed_leaves_read_only = True
            else:
                if train_stages is not None:
                    msg = "--train-stages applies only to explicit matched_a1_b plans."
                    raise loader.ConfigError(msg)
                resolved = loader.resolve_config(raw, source=config_source)
                unit_run_dirs = experiments.source_binding.existing_train_run_dirs((resolved,))
    else:
        if resume is not None:
            msg = "--resume applies only to the train workflow."
            raise loader.ConfigError(msg)
        if any(value is not None for value in (train_seconds, rollout_length, lr_scale)):
            msg = "--train-seconds, --rollout-length and --lr-scale apply only to matched transient training."
            raise loader.ConfigError(msg)
        if train_stages is not None:
            msg = "--train-stages applies only to the train workflow."
            raise loader.ConfigError(msg)
        from src.experiments.tuning import experiments_tuning_optuna as optuna  # noqa: PLC0415

        resolved = optuna.load_optuna_study_config(source, epochs=epochs, train_dataset=train_dataset, revision=revision).base_config
        unit_run_dirs = experiments.source_binding.existing_study_run_dirs(resolved)
        unit_study_dir = experiments.source_binding.study_dir(resolved)
    model = resolved.get("model")
    loss = resolved.get("loss")
    if not isinstance(model, dict) or not isinstance(loss, dict):
        msg = "Resolved executable config must contain model and loss mappings."
        raise loader.ConfigError(msg)
    physics = loss.get("physics")
    if not isinstance(physics, dict) or type(physics.get("enabled")) is not bool:
        msg = "Resolved executable config must contain boolean loss.physics.enabled."
        raise loader.ConfigError(msg)
    task = resolved.get("task")
    model_kind = model.get("kind")
    run = resolved.get("run")
    run_label = run.get("name") if isinstance(run, dict) else None
    if not isinstance(task, str) or not isinstance(model_kind, str) or not isinstance(run_label, str):
        msg = "Resolved executable config must contain string task, model.kind, and run.name values."
        raise loader.ConfigError(msg)
    return ConfigPreflight(
        family=family,
        task=task,
        model_kind=model_kind,
        physics_enabled=physics["enabled"],
        run_label=run_label,
        source_path=source.resolve(),
        canonical_path=_canonical_config_path(source),
        unit_run_dirs=unit_run_dirs,
        completed_leaves_read_only=completed_leaves_read_only,
        unit_study_dir=unit_study_dir,
    )


def matching_optuna_configs(result: ConfigPreflight) -> tuple[ConfigPreflight, ...]:
    """Return valid task/model-family wrappers matching one normal experiment."""
    if result.family != EXPERIMENT_FAMILY:
        return ()
    root = common.paths.get_project_root().expanduser().resolve() / "configs/03_learning"
    candidates: list[ConfigPreflight] = []
    for path in sorted(root.glob(f"{result.task}/optuna/**/*.yaml")):
        try:
            candidate = inspect_config(path)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if (
            candidate.family == OPTUNA_FAMILY
            and candidate.task == result.task
            and candidate.model_kind == result.model_kind
            and candidate.physics_enabled is result.physics_enabled
        ):
            candidates.append(candidate)
    return tuple(candidates)


def _quoted(value: str) -> str:
    """Single-quote one shell argument without relying on caller interpolation."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def mismatch_message(result: ConfigPreflight, *, requested_workflow: str) -> str:
    """Build the actionable wrong-workflow message for a validated config."""
    if requested_workflow == "train":
        return "\n".join(
            (
                "Configuration workflow mismatch.",
                f"Supplied config family: {result.family}",
                "Requested workflow: train",
                "",
                "Use:",
                "Replace RESOURCE_OPTIONS with matching CPU or typed-GPU flags; see ./scripts/slurm_ml.sh --help.",
                f"./scripts/slurm_ml.sh RESOURCE_OPTIONS optuna {_quoted(result.canonical_path)}",
            )
        )
    lines = [
        "Configuration workflow mismatch.",
        f"Supplied config family: {result.family}",
        "Requested workflow: optuna",
        "",
        "This file is a normal training experiment.",
        "",
        "Train it with:",
        "Replace RESOURCE_OPTIONS with matching CPU or typed-GPU flags; see ./scripts/slurm_ml.sh --help.",
        f"./scripts/slurm_ml.sh RESOURCE_OPTIONS train {_quoted(result.canonical_path)}",
    ]
    matches = matching_optuna_configs(result)
    if len(matches) == 1:
        lines.extend(
            (
                "",
                "Matching Optuna study:",
                f"./scripts/slurm_ml.sh RESOURCE_OPTIONS optuna {_quoted(matches[0].canonical_path)}",
            )
        )
    return "\n".join(lines)


def validate_workflow(
    path: Path | str,
    *,
    requested_workflow: str,
    epochs: int | None = None,
    train_dataset: str | None = None,
    train_seconds: float | None = None,
    rollout_length: int | None = None,
    lr_scale: float | None = None,
    revision: str | None = None,
    resume: Path | str | None = None,
    train_stages: tuple[str, ...] | None = None,
) -> ConfigPreflight:
    """Strictly validate one config and reject workflow-family mismatches.

    ``resume`` is train-only and selects saved-effective validation only when
    ``path`` is exactly the run directory's ``config.yaml``. ``train_stages`` is
    train-only and selects the matched stages the job trains. The result lists the
    leaves the job may write (completed matched stages are read-only and dropped)
    and names the frozen source tree their newest Slurm session ran in (``None``
    for a new unit).
    """
    if requested_workflow not in _WORKFLOW_FAMILY:
        msg = f"Unsupported config workflow preflight: {requested_workflow!r}."
        raise ValueError(msg)
    if resume is not None and requested_workflow != "train":
        msg = "--resume applies only to the train workflow."
        raise ValueError(msg)
    result = inspect_config(
        path,
        epochs=epochs,
        train_dataset=train_dataset,
        train_seconds=train_seconds,
        rollout_length=rollout_length,
        lr_scale=lr_scale,
        revision=revision,
        resume=resume,
        train_stages=train_stages,
    )
    if result.family != _WORKFLOW_FAMILY[requested_workflow]:
        raise WorkflowMismatchError(
            mismatch_message(result, requested_workflow=requested_workflow),
            result=result,
        )
    writable = result.unit_run_dirs
    if result.completed_leaves_read_only:
        writable = experiments.source_binding.unfinished_run_dirs(writable)
    return replace(
        result,
        unit_run_dirs=writable,
        bound_source_tree=experiments.source_binding.bound_source_tree_id(writable, study=result.unit_study_dir),
    )
