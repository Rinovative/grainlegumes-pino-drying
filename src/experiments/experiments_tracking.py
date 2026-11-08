"""Mirror authoritative local experiment state to optional W&B observability.

Responsibilities:
  - Keep disabled tracking free of SDK imports and tracking side effects
  - Persist opaque fresh identities and exact-resume identities locally first
  - Publish bounded semantic config, completed-epoch history, summaries, and curated media
  - Mirror authoritative ID, OOD, and physics event values without recomputation
  - Fail closed on requested online or offline observer failures
  - Keep the SDK console silent and the online SDK directory node-local, preserved in the run folder after an observer failure
  - Keep built-in system telemetry secondary to scientific metrics

Design principles:
  - Local config, split, normalizer, checkpoints, summaries and artifacts win
  - W&B observes training and artifacts but never chooses or reconstructs them
  - Normal histories contain genuine completed epochs only. Final results are summaries
  - Fresh and exact-resume sessions have strict fail-closed identity semantics
  - Secrets, arbitrary environment state and incidental absolute paths are absent

This module does NOT:
  - Own training, checkpoint, scheduler, pruning, or local lifecycle decisions
  - Upload raw datasets, resume-only checkpoints, arbitrary files, or cache internals
  - Make remote availability or W&B state a prerequisite for local correctness
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from src import datasets, learning

from . import experiments_run_identity as run_identity
from . import experiments_wandb_schema as wandb_schema

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

EpochEndCallback = Callable[[int, dict[str, float]], None]
TrackingStateUpdater = Callable[[Mapping[str, Any]], None]

POST_ARTIFACT_MEDIA_KEYS = frozenset(
    {
        "run_summary_table",
        "accuracy_physics_pareto",
        "physics_residual_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    }
)
_POST_ARTIFACT_FILE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".html", ".pdf"})
_SECRET_KEY_PATTERN = re.compile(r"(?i)(WANDB_API_KEY|api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")
_MAX_SAFE_ERROR_LENGTH = 600
_MAX_EVALUATION_SUMMARY_ENTRIES = 512
_MAX_EVALUATION_SUMMARY_TEXT = 512
_TRANSIENT_EVALUATION_HORIZONS = "1|2|4|8|16|32|64|128|full"
_TRANSIENT_EVALUATION_METRIC_KEY = re.compile(
    rf"^evaluation/(?:0|[1-9][0-9]*)/(?:id|additional_evaluation)/(?:teacher_forced_one_step|autonomous_full|rolling_origin)/(?:{_TRANSIENT_EVALUATION_HORIZONS})/(?:cumulative|endpoint)/(?P<metric>[A-Za-z0-9_]+)$"
)
_TRANSIENT_EVALUATION_PIPELINE_KEY = re.compile(r"^evaluation/pipeline/(?:0|[1-9][0-9]*)/(?P<metric>[A-Za-z0-9_]+)$")
_STEADY_EVALUATION_METRIC_KEY = re.compile(r"^evaluation/(?:id|additional_evaluation)/(?P<metric>[A-Za-z0-9_]+)$")
_EVALUATION_DATASET_KEY = re.compile(r"^evaluation/identity/dataset/(?:id|additional_evaluation)/name$")
_TRANSIENT_EVALUATION_SPEEDUP_KEY = re.compile(
    r"^evaluation/timing/speedup/(?P<name>drying_only_solver_speedup|full_pipeline_solver_speedup|hybrid_component_speedup|comsol_process_speedup|generation_compute_end_to_end_speedup)/(?P<metric>available_count|unavailable_count|ratio_of_sums)$"
)
_EVALUATION_COMPONENT_KEY = re.compile(
    r"^evaluation/timing/component/(?P<name>comsol_transient_drying_seconds|drying_no_rollout_model_seconds|dataset_materialization_seconds|batch_amortized_neural_operator_forward_seconds)/(?P<metric>available_count|median_seconds)$"
)
_STEADY_EVALUATION_COUNT_METRICS = frozenset(
    {"valid_case_count", "nonfinite_case_count", "required_value_count", "computed_value_count", "finite_value_count", "nonfinite_value_count"}
)
_STEADY_EVALUATION_SUMMARY_METRICS = frozenset(
    {
        *_STEADY_EVALUATION_COUNT_METRICS,
        "normalized_group_macro_rmse",
        *(f"{space}_rmse_{field}" for space in ("normalized", "physical") for field in ("p", "u", "v")),
    }
)
_TRANSIENT_EVALUATION_SUMMARY_METRICS = frozenset(
    {
        "normalized_drying_group_macro_rmse",
        "contributing_record_count",
        "contributing_case_count",
        "unavailable_case_count",
        "elapsed_physical_time_median",
        "physical_w_gr_rmse",
        "physical_w_gr_mae",
        "bulk_dry_basis_rmse",
        "bulk_wet_basis_rmse",
        "bulk_moisture_valid_count",
        "bulk_moisture_unavailable_count",
        "target_available_count",
        "predicted_right_censored_count",
        "target_time_error_count",
        "target_time_error_mean",
        "target_time_error_mae",
        "target_gap_count",
        "predicted_final_target_gap_mean",
        "reference_final_target_gap_mean",
        "target_final_gap_error_mean",
        "target_final_gap_error_mae",
        "nonfinite_values",
        "negative_moisture_values",
        "abnormal_increment_growth_count",
        *(f"normalized_rmse_{field}" for field in ("T", "phi", "w_surf", "w_int")),
        *(f"physical_rmse_{field}" for field in ("T", "phi", "w_surf", "w_int")),
        *(f"physical_mae_{field}" for field in ("T", "phi", "w_surf", "w_int")),
    }
)
_TRANSIENT_EVALUATION_COUNT_METRICS = frozenset(
    {
        "contributing_record_count",
        "contributing_case_count",
        "unavailable_case_count",
        "bulk_moisture_valid_count",
        "bulk_moisture_unavailable_count",
        "target_available_count",
        "predicted_right_censored_count",
        "target_time_error_count",
        "target_gap_count",
        "nonfinite_values",
        "negative_moisture_values",
        "abnormal_increment_growth_count",
    }
)
_TRANSIENT_EVALUATION_PIPELINE_METRICS = frozenset(
    {
        "drying_surrogate_error",
        "c_available",
        "complete_pipeline_error",
        "airflow_substitution_discrepancy",
        "signed_airflow_degradation",
        "airflow_degradation_ratio",
        "upstream_airflow_error",
    }
)
_EVALUATION_HASH_IDENTITIES = frozenset(
    {
        "evaluation/identity/checkpoint_sha256",
        "evaluation/identity/timing_evidence_sha256",
        "evaluation/identity/dataset_sha256",
        "evaluation/timing/hardware_identity_sha256",
    }
)
_EVALUATION_EXACT_TEXT_IDENTITIES = {
    "evaluation/identity/input_profile": frozenset({"canonical_physics_complete_v1"}),
    "evaluation/identity/model_kind": frozenset({"fno", "uno", "rno"}),
    "evaluation/timing/precision": frozenset({"float32"}),
}
_EVALUATION_LOGICAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}$")
_EVALUATION_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVALUATION_DEVICE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)(?:\|(?:cpu|cuda(?::[0-9]+)?))*$")
AUTOMATIC_HISTORY_TOP_LEVEL_PREFIXES = (
    "Overview",
    "Accuracy",
    "Loss",
    "Optimization",
    "Curriculum",
    "Performance",
    "Physics",
    "Diagnostics",
    "Optuna",
)


def _is_nonnegative_int(value: Any) -> bool:
    """Return whether one outbound aggregate is an exact non-negative integer."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_finite_number(value: Any) -> bool:
    """Return whether one outbound aggregate is a finite real non-boolean."""
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))


def _require_summary_number(key: str, value: Any, *, count: bool) -> None:
    """Require one Evaluation count to be a non-negative integer and one aggregate to be finite."""
    if count and not _is_nonnegative_int(value):
        msg = f"Evaluation count {key!r} must be a non-negative integer."
        raise TrackingError(msg)
    if not count and not _is_finite_number(value):
        msg = f"Evaluation aggregate {key!r} must be finite numeric evidence."
        raise TrackingError(msg)


def _validate_metric_entry(key: str, value: Any) -> bool:
    """Validate one steady role, transient mode/horizon or pipeline aggregate key when recognized."""
    for pattern, vocabulary, counts in (
        (_STEADY_EVALUATION_METRIC_KEY, _STEADY_EVALUATION_SUMMARY_METRICS, _STEADY_EVALUATION_COUNT_METRICS),
        (_TRANSIENT_EVALUATION_METRIC_KEY, _TRANSIENT_EVALUATION_SUMMARY_METRICS, _TRANSIENT_EVALUATION_COUNT_METRICS),
    ):
        metric_match = pattern.fullmatch(key)
        if metric_match is not None:
            if metric_match.group("metric") not in vocabulary:
                return False
            _require_summary_number(key, value, count=metric_match.group("metric") in counts)
            return True

    pipeline_match = _TRANSIENT_EVALUATION_PIPELINE_KEY.fullmatch(key)
    if pipeline_match is None:
        return False
    metric = pipeline_match.group("metric")
    if metric not in _TRANSIENT_EVALUATION_PIPELINE_METRICS:
        return False
    if metric == "c_available":
        if not _is_nonnegative_int(value) or int(value) not in {0, 1}:
            msg = "Transient Evaluation pipeline availability must be exactly 0 or 1."
            raise TrackingError(msg)
    elif not _is_finite_number(value):
        msg = f"Transient Evaluation pipeline aggregate {key!r} must be finite numeric evidence."
        raise TrackingError(msg)
    return True


def _validate_identity_entry(key: str, value: Any) -> bool:
    """Validate one exact approved Dataset, model, or Training identity key."""
    if key in _EVALUATION_HASH_IDENTITIES:
        if not isinstance(value, str) or _EVALUATION_SHA256.fullmatch(value) is None:
            msg = f"Evaluation identity {key!r} must be one SHA-256 digest."
            raise TrackingError(msg)
        return True
    allowed = _EVALUATION_EXACT_TEXT_IDENTITIES.get(key)
    if allowed is not None:
        if value not in allowed:
            msg = f"Evaluation identity {key!r} is unsupported."
            raise TrackingError(msg)
        return True
    if key == "evaluation/identity/checkpoint_epoch":
        if not _is_nonnegative_int(value):
            msg = "Evaluation checkpoint epoch must be a non-negative integer."
            raise TrackingError(msg)
        return True
    if _EVALUATION_DATASET_KEY.fullmatch(key) is not None:
        if not isinstance(value, str) or _EVALUATION_LOGICAL_NAME.fullmatch(value) is None:
            msg = "Evaluation Dataset identity must be one bounded logical name."
            raise TrackingError(msg)
        return True
    return False


def _validate_timing_entry(key: str, value: Any) -> bool:
    """Validate one approved aggregate timing, device, or speedup entry."""
    if key == "evaluation/timing/device":
        if not isinstance(value, str) or _EVALUATION_DEVICE.fullmatch(value) is None:
            msg = "Evaluation device identity is unsupported."
            raise TrackingError(msg)
        return True
    if key == "evaluation/timing/warmup_passes":
        if not _is_nonnegative_int(value):
            msg = "Evaluation warm-up count must be a non-negative integer."
            raise TrackingError(msg)
        return True
    if key == "evaluation/timing/component_composed":
        if not isinstance(value, bool):
            msg = "Transient Evaluation component-composed identity must be boolean."
            raise TrackingError(msg)
        return True

    speedup_match = _TRANSIENT_EVALUATION_SPEEDUP_KEY.fullmatch(key)
    if speedup_match is not None:
        metric = speedup_match.group("metric")
        if metric in {"available_count", "unavailable_count"}:
            if not _is_nonnegative_int(value):
                msg = f"Transient Evaluation speedup count {key!r} must be a non-negative integer."
                raise TrackingError(msg)
        elif not _is_finite_number(value) or float(value) <= 0.0:
            msg = f"Transient Evaluation speedup {key!r} must be finite and positive."
            raise TrackingError(msg)
        return True

    component_match = _EVALUATION_COMPONENT_KEY.fullmatch(key)
    if component_match is None:
        return False
    metric = component_match.group("metric")
    if metric == "available_count":
        if not _is_nonnegative_int(value):
            msg = f"Evaluation timing count {key!r} must be a non-negative integer."
            raise TrackingError(msg)
    elif not _is_finite_number(value) or float(value) < 0.0:
        msg = f"Evaluation timing median {key!r} must be finite and non-negative."
        raise TrackingError(msg)
    return True


def _validate_evaluation_summary_entry(key: Any, value: Any) -> str:
    """Admit only the declared aggregate and identity publication vocabulary of both tasks."""
    if not isinstance(key, str) or not key:
        msg = "Evaluation summary keys must be non-empty text."
        raise TrackingError(msg)
    if _validate_metric_entry(key, value) or _validate_identity_entry(key, value) or _validate_timing_entry(key, value):
        return key
    msg = f"Unsupported Evaluation summary key {key!r}."
    raise TrackingError(msg)


def automatic_history_metric_definitions(
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
    physics_training_enabled: bool,
    physics_monitor_enabled: bool,
    cuda_enabled: bool,
    optuna_trial: bool = False,
    task_id: str | None = None,
    state_aux_enabled: bool = False,
    ood_enabled: bool = True,
) -> tuple[wandb_schema.MetricProjection, ...]:
    """Return the ordered curated history contract; the schema module owns sections and key names."""
    return wandb_schema.curated_metric_projections(
        task_id,
        evaluation_metrics,
        objective_id=objective_id,
        physics_training_enabled=physics_training_enabled,
        physics_monitor_enabled=physics_monitor_enabled,
        state_aux_enabled=state_aux_enabled,
        ood_enabled=ood_enabled,
        cuda_enabled=cuda_enabled,
        optuna_trial=optuna_trial,
    )


class TrackingError(RuntimeError):
    """
    Base class for failures owned by the optional tracking boundary.

    These errors describe observer initialization, local offline persistence, or
    upload-admission failure. They never represent local scientific-run validity.
    """


class TrackingInitializationError(TrackingError):
    """
    Represent failure to initialize an explicitly enabled session before epoch one.

    Raised after sanitized failure facts are persisted locally. No training
    telemetry has been accepted by the observer at this boundary.
    """


class TrackingIOError(TrackingError):
    """Represent loss of records in an explicitly requested tracking mode."""


class TrackingUploadError(TrackingError):
    """
    Represent rejection by the run-file or curated-media upload allowlist.

    The error is raised before an SDK mutation for unsupported kinds, paths,
    formats, cache ownership, or disabled checkpoint publication.
    """


class _WandbRun(Protocol):
    """Describe the SDK run surface used by the lifecycle adapter."""

    summary: MutableMapping[str, Any]
    tags: Sequence[str] | None
    url: str | None

    def define_metric(self, name: str, **kwargs: Any) -> Any:
        """Define one supported metric family and its epoch step contract."""

    def log(self, data: Mapping[str, Any], *, step: int) -> None:
        """Log one completed epoch."""

    def finish(self, exit_code: int = 0) -> None:
        """Finalize the tracking run."""


class _WandbArtifact(Protocol):
    """Describe the bounded W&B artifact bundle surface."""

    def add_file(self, path: str, *, name: str) -> None:
        """Add one explicit rendered media file."""

    def add(self, value: Any, name: str) -> None:
        """Add one explicit prebuilt table object."""


class _WandbModule(Protocol):
    """Describe the lazily imported W&B module surface."""

    Settings: Callable[..., Any]

    def setup(self, *, settings: Any) -> Any:
        """Apply process-wide SDK settings."""

    def init(self, **kwargs: Any) -> _WandbRun | None:
        """Initialize one W&B run."""

    def Table(self, *, columns: Sequence[str], data: Sequence[Sequence[object]]) -> Any:  # noqa: N802
        """Build one W&B table from a neutral curated payload."""


def _require_initialized_run(run: _WandbRun | None) -> _WandbRun:
    """Return the SDK run or fail through the initialization wrapper."""
    if run is None:
        msg = "wandb.init() did not return a run."
        raise RuntimeError(msg)
    return run


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _safe_error(error: BaseException) -> dict[str, str]:
    """
    Return bounded exception context safe for local or remote tracking records.

    Active W&B keys, key-like assignments, and the current home path are
    redacted before truncation. Only the exception class and sanitized message
    are returned. Traceback, environment, and arbitrary object state are absent.
    """
    message = str(error)
    secret = os.environ.get("WANDB_API_KEY")
    if secret:
        message = message.replace(secret, "<redacted>")
    message = _SECRET_KEY_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", message)
    with suppress(RuntimeError):
        home = str(Path.home())
        if home and home != "/":
            message = message.replace(home, "<home>")
    return {
        "error_class": type(error).__name__,
        "error_message": message[:_MAX_SAFE_ERROR_LENGTH],
    }


def _sanitize_semantic_value(value: Any, *, key: str = "") -> Any:
    """
    Recursively copy a semantic value while excluding secrets and host paths.

    Secret-like mapping keys are removed, absolute paths retain only a basename,
    and unsupported objects fail instead of being stringified. The result is a
    bounded JSON-like structure suitable for W&B config, never scientific state.
    """
    lowered = key.lower()
    if any(marker in lowered for marker in ("api_key", "password", "secret", "token")):
        return None
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for child_key, child_value in value.items():
            name = str(child_key)
            if any(marker in name.lower() for marker in ("api_key", "password", "secret", "token")):
                continue
            sanitized[name] = _sanitize_semantic_value(child_value, key=name)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_semantic_value(item, key=key) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if Path(value).is_absolute():
            return Path(value).name or "<absolute-path-omitted>"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    msg = f"Semantic tracking payload contains unsupported value at {key or '<root>'}: {type(value).__name__}."
    raise TypeError(msg)


def _git_metadata() -> dict[str, Any]:
    """Return the shared bounded read-only repository provenance."""
    return run_identity.source_repository_evidence()


def model_parameter_counts(model: Any) -> dict[str, int]:
    """Count the exact instantiated total and trainable model parameters."""
    return {
        "total": sum(int(parameter.numel()) for parameter in model.parameters()),
        "trainable": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
    }


def _package_versions() -> dict[str, str | None]:
    """Return a bounded reproducibility package inventory without SDK imports."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for label, distribution in (
        ("numpy", "numpy"),
        ("neuraloperator", "neuraloperator"),
        ("optuna", "optuna"),
        ("wandb", "wandb"),
    ):
        versions[label] = None
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[label] = importlib.metadata.version(distribution)
    return versions


def _split_semantics(split_indices: Mapping[str, Any], *, split_indices_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the admitted split of either task into its W&B Dataset identities and split evidence."""
    split = datasets.preprocessing.splits.admit_split(split_indices)
    payload = split.as_payload()
    split_payload = {
        "schema_kind": payload["schema_kind"],
        "schema_version": split.schema_version,
        "artifact": "split_indices.pt",
        "artifact_sha256": split_indices_sha256,
        "split_seed": split.split_seed,
        "case_partition_digest": split.case_partition_digest,
        "roles": payload["roles"],
        "evaluation_assignments": payload["evaluation_assignments"],
        "extension": payload["extension"],
    }
    return payload["dataset_identity"], split_payload


def _build_transient_semantic_config(
    config: Mapping[str, Any],
    *,
    split_indices: Mapping[str, Any],
    split_indices_sha256: str,
    normalizer_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    model: Any,
    device_metadata: Mapping[str, Any],
    duration_contract: Mapping[str, Any],
    transient_scaling: Mapping[str, Any] | None,
    transient_handoff: Mapping[str, Any] | None,
    tuning_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build bounded transient-drying provenance without steady split assumptions."""
    task_contract = cast("Mapping[str, Any]", config["task_contract"])
    data_config = cast("Mapping[str, Any]", config["data"])
    run_config = cast("Mapping[str, Any]", config["run"])
    datasets_payload, split_payload = _split_semantics(split_indices, split_indices_sha256=split_indices_sha256)
    scaling_payload = copy.deepcopy(dict(transient_scaling or {}))
    scaling_payload["artifact"] = "normalizer.pt"
    scaling_payload["artifact_sha256"] = normalizer_sha256
    model_payload = copy.deepcopy(dict(cast("Mapping[str, Any]", config["model"])))
    model_payload["parameter_counts"] = model_parameter_counts(model)
    payload: dict[str, Any] = {
        "task": {"id": config["task"], "contract_digest": task_contract.get("digest"), "contract": copy.deepcopy(dict(task_contract))},
        "run": {
            "name": run_config["name"],
            "seed": run_config["seed"],
            "stage": cast("Mapping[str, Any]", config["training"]).get("stage"),
        },
        "data": {
            "datasets": datasets_payload,
            "configured_identity": {
                "train_dataset": data_config["train_dataset"],
                "additional_evaluation_datasets": copy.deepcopy(data_config["additional_evaluation_datasets"]),
            },
            "loader": {
                key: copy.deepcopy(data_config.get(key))
                for key in ("batch_size", "num_workers", "pin_memory", "persistent_workers", "spatial_stride")
            },
            "split": split_payload,
            "normalization": scaling_payload,
        },
        "model": model_payload,
        "loss": copy.deepcopy(config["loss"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
        "optimizer": copy.deepcopy(config["optimizer"]),
        "scheduler": copy.deepcopy(config.get("scheduler")),
        "training": copy.deepcopy(config["training"]),
        "checkpoint": copy.deepcopy(dict(checkpoint_identity)),
        "teacher_handoff": copy.deepcopy(dict(transient_handoff)) if transient_handoff is not None else None,
        "provenance": {
            "repository": _git_metadata(),
            "resolved_config_sha256": run_identity.resolved_config_digest(config),
            "checkpoint_config_digest": checkpoint_identity.get("effective_config_digest"),
            "task_contract_digest": task_contract.get("digest"),
        },
        "runtime": {
            "device": copy.deepcopy(dict(device_metadata)),
            "packages": {**_package_versions(), "pytorch": device_metadata.get("pytorch_version")},
            "duration_contract": copy.deepcopy(dict(duration_contract)),
        },
    }
    if tuning_context is not None:
        payload["tuning"] = copy.deepcopy(dict(tuning_context))
    return cast("dict[str, Any]", _sanitize_semantic_value(payload))


def build_semantic_config(
    config: Mapping[str, Any],
    *,
    split_indices: Mapping[str, Any],
    split_indices_sha256: str,
    normalizer_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    model: Any,
    device_metadata: Mapping[str, Any],
    duration_contract: Mapping[str, Any],
    tuning_context: Mapping[str, Any] | None = None,
    transient_scaling: Mapping[str, Any] | None = None,
    transient_handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one complete, nested, path-free scientific W&B configuration."""
    if config.get("task") == "transient_drying":
        return _build_transient_semantic_config(
            config,
            split_indices=split_indices,
            split_indices_sha256=split_indices_sha256,
            normalizer_sha256=normalizer_sha256,
            checkpoint_identity=checkpoint_identity,
            model=model,
            device_metadata=device_metadata,
            duration_contract=duration_contract,
            transient_scaling=transient_scaling,
            transient_handoff=transient_handoff,
            tuning_context=tuning_context,
        )
    task_contract = config.get("task_contract")
    if not isinstance(task_contract, Mapping):
        msg = "Semantic tracking config requires a resolved task contract."
        raise TypeError(msg)
    dataset_payload, split_payload = _split_semantics(split_indices, split_indices_sha256=split_indices_sha256)
    data_config = cast("Mapping[str, Any]", config["data"])
    loader_keys = ("batch_size", "num_workers", "pin_memory", "persistent_workers")
    preprocessing = task_contract.get("preprocessing")
    physics_contract = task_contract.get("physics")
    if not isinstance(preprocessing, Mapping) or not isinstance(physics_contract, Mapping):
        msg = "Semantic tracking config requires task preprocessing and physics semantics."
        raise TypeError(msg)

    parameter_counts = model_parameter_counts(model)
    model_payload = copy.deepcopy(dict(cast("Mapping[str, Any]", config["model"])))
    loss_config = cast("Mapping[str, Any]", config["loss"])
    physics_config = cast("Mapping[str, Any]", loss_config["physics"])
    derivative_config = physics_config.get("derivatives")
    if not isinstance(derivative_config, Mapping):
        msg = "Semantic tracking config requires physics derivative semantics."
        raise TypeError(msg)
    physics_enabled = bool(physics_config["enabled"])
    loss_payload = {
        "data": copy.deepcopy(dict(cast("Mapping[str, Any]", loss_config["data"]))),
        "physics": copy.deepcopy(dict(physics_config)) if physics_enabled else {"enabled": False},
    }
    wandb_config = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    monitor_config = cast("Mapping[str, Any]", wandb_config["monitor"])
    diagnostics_payload = {
        "physics_monitor": {
            "enabled": bool(monitor_config["enabled"]),
            "role": "id",
            "membership": "bounded_saved_evaluation_prefix",
            "interval_epochs": int(monitor_config["interval"]),
            "max_cases": int(monitor_config["max_cases"]),
            "physics_kind": physics_contract.get("kind"),
            "equation_set": physics_contract.get("equation_set"),
            "continuity": physics_contract.get("continuity"),
            "boundary": physics_contract.get("boundary"),
            "derivatives": copy.deepcopy(dict(derivative_config)),
            "interior_crop": int(physics_config["interior_crop"]),
            "metric_ids": list(wandb_schema.PHYSICS_VALIDATION_METRICS),
        }
    }
    model_payload["variant"] = learning.models.naming.architecture_label(config)
    model_payload["parameter_counts"] = parameter_counts
    evaluation_payload = copy.deepcopy(dict(cast("Mapping[str, Any]", config["evaluation"])))
    training_config = cast("Mapping[str, Any]", config["training"])
    evaluation_payload["roles"] = {
        "selection": "id",
        "id_interpretation": "in_distribution_validation",
        "diagnostic": ["additional_evaluation", "physics"],
        "event_model": "completed_epoch_interval_or_terminal",
        "id_interval_epochs": int(training_config["evaluation_interval"]),
        "additional_evaluation_interval_epochs": int(training_config["additional_evaluation_interval"]),
        "physics_interval_epochs": int(monitor_config["interval"]),
        "epoch_zero_evaluation": False,
    }
    source = _git_metadata()
    payload: dict[str, Any] = {
        "task": {
            "id": config["task"],
            "contract_digest": task_contract.get("digest"),
            "contract": copy.deepcopy(dict(task_contract)),
        },
        "data": {
            "datasets": dataset_payload,
            "loader": {key: copy.deepcopy(data_config.get(key)) for key in loader_keys},
            "split": split_payload,
            "normalization": {
                **copy.deepcopy(dict(preprocessing)),
                "artifact": "normalizer.pt",
                "artifact_sha256": normalizer_sha256,
            },
        },
        "model": model_payload,
        "loss": loss_payload,
        "diagnostics": diagnostics_payload,
        "evaluation": evaluation_payload,
        "optimizer": copy.deepcopy(dict(cast("Mapping[str, Any]", config["optimizer"]))),
        "scheduler": copy.deepcopy(config.get("scheduler")),
        "training": copy.deepcopy(dict(cast("Mapping[str, Any]", config["training"]))),
        "run": copy.deepcopy(dict(cast("Mapping[str, Any]", config["run"]))),
        "provenance": {
            "repository": source,
            "config_digest": checkpoint_identity.get("effective_config_digest"),
            "task_contract_digest": task_contract.get("digest"),
        },
        "runtime": {
            "device": copy.deepcopy(dict(device_metadata)),
            "packages": {**_package_versions(), "pytorch": device_metadata.get("pytorch_version")},
            "duration_contract": copy.deepcopy(dict(duration_contract)),
        },
    }
    if tuning_context is not None:
        payload["tuning"] = copy.deepcopy(dict(tuning_context))
    return cast("dict[str, Any]", _sanitize_semantic_value(payload))


def build_monitor_membership(
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    Build the exact saved-evaluation prefix identity used by physics monitors.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved experiment config with validated W&B monitor settings.
    split_indices : Mapping[str, Any]
        Persisted split artifact including ordered indices and dataset identity.

    Returns
    -------
    dict[str, Any] | None
        Source indices, sample IDs, membership digests, and configured bound when
        monitoring is enabled. Otherwise ``None``.

    Notes
    -----
    The membership is fixed before training telemetry and never resampled by epoch.

    """
    settings = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    monitor = cast("Mapping[str, Any]", settings["monitor"])
    if settings["mode"] == "disabled" or not bool(monitor["enabled"]):
        return None
    evidence = datasets.preprocessing.splits.admit_split(split_indices).validation
    selected = list(evidence.index_values[: int(monitor["max_cases"])])
    digest = datasets.contracts.identity.membership_digest(
        role="wandb_physics_monitor",
        dataset_fingerprint=evidence.dataset["dataset_digest"],
        sample_ids=evidence.dataset["case_ids"],
        indices=selected,
    )
    return {
        "source_indices": selected,
        "sample_ids": list(evidence.case_ids[: int(monitor["max_cases"])]),
        "membership_digest": digest,
        "saved_eval_membership_digest": evidence.membership_digest,
        "max_cases": int(monitor["max_cases"]),
    }


def persisted_wandb_identity(summary: Mapping[str, Any]) -> tuple[str, int | None]:
    """
    Recover the sole W&B identity and latest successful epoch from local sessions.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Authoritative run summary containing append-only runtime session records.

    Returns
    -------
    tuple[str, int | None]
        Persisted W&B run ID and maximum locally recorded logged epoch.

    Raises
    ------
    TrackingInitializationError
        If runtime sessions are malformed or contain zero/multiple run IDs.

    """
    raw_sessions = summary.get("runtime_sessions", [])
    if not isinstance(raw_sessions, list):
        msg = "Run summary runtime_sessions must be a list."
        raise TrackingInitializationError(msg)
    identities: set[str] = set()
    last_epoch: int | None = None
    for raw_session in raw_sessions:
        if not isinstance(raw_session, Mapping):
            continue
        state = raw_session.get("tracking")
        if not isinstance(state, Mapping):
            continue
        run_id = state.get("wandb_run_id")
        if isinstance(run_id, str) and run_id:
            identities.add(run_id)
        raw_epoch = state.get("last_logged_epoch")
        if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool):
            last_epoch = raw_epoch if last_epoch is None else max(last_epoch, raw_epoch)
    if len(identities) != 1:
        msg = f"Exact W&B resume requires one persisted run ID, found {len(identities)}."
        raise TrackingInitializationError(msg)
    return next(iter(identities)), last_epoch


def persisted_evaluation_summary_keys(summary: Mapping[str, Any]) -> frozenset[str]:
    """Return every Evaluation summary key that any recorded runtime session published to W&B."""
    states = (session.get("tracking") for session in summary.get("runtime_sessions", []) if isinstance(session, Mapping))
    return frozenset(key for state in states if isinstance(state, Mapping) for key in state.get("evaluation_summary_keys", ()))


@dataclass(slots=True)
class WandbSession:
    """Own one optional, fail-closed W&B mirror of authoritative local state."""

    _run: _WandbRun | None
    _wandb: Any | None
    objective_id: str
    objective_direction: str
    evaluation_metric_ids: frozenset[str]
    mode: str
    run_id: str | None
    run_dir: Path
    run_name: str
    task_id: str
    upload_settings: Mapping[str, Any]
    semantic_config: Mapping[str, Any] = field(default_factory=dict)
    history_metric_definitions: tuple[wandb_schema.MetricProjection, ...] = ()
    state_updater: TrackingStateUpdater | None = None
    _resumed_through_epoch: int | None = None
    _published_summary_keys: frozenset[str] = frozenset()
    _last_logged_epoch: int | None = None
    _finished: bool = False
    _observer_failed: bool = False
    _failed_operation: str | None = None
    _uploaded_media: list[str] = field(default_factory=list)
    _local_root: Path | None = None

    def _preserve_local_state(self) -> None:
        """Copy the node-local SDK directory into the run folder; a failed copy is a warning and never masks the run outcome."""
        source = None if self._local_root is None else self._local_root / "wandb"
        if source is None or not source.is_dir():
            return
        try:
            shutil.copytree(source, self.run_dir / "wandb", dirs_exist_ok=True)
        except OSError as copy_error:
            print(f"Local W&B state was not preserved in the run folder: {type(copy_error).__name__}", file=sys.stderr)

    @property
    def enabled(self) -> bool:
        """Return whether this session owns an initialized SDK run."""
        return self._run is not None

    @property
    def url(self) -> str | None:
        """Return the active W&B run URL when the SDK exposes one."""
        value = getattr(self._run, "url", None)
        return value if isinstance(value, str) and value else None

    @property
    def history_definition_by_source(self) -> dict[str, wandb_schema.MetricProjection]:
        """Return complete source projections including presentation-only unit scaling."""
        return {definition.source_key: definition for definition in self.history_metric_definitions}

    def _persist(self, updates: Mapping[str, Any]) -> None:
        """Persist bounded observer state through the authoritative local writer."""
        if self.state_updater is not None:
            self.state_updater(copy.deepcopy(dict(updates)))

    def _operation_failure(self, error: BaseException, *, operation: str) -> TrackingIOError:
        """Persist sanitized context and return one fail-closed tracking error."""
        context = _safe_error(error)
        self._observer_failed = True
        self._failed_operation = operation
        self._persist({"status": "failed", "failed_operation": operation, **context})
        msg = f"Requested {self.mode} W&B {operation} failed: {context['error_class']}: {context['error_message']}"
        return TrackingIOError(msg)

    def _run_operation(self, operation: str, action: Callable[[], None]) -> bool:
        """Apply one SDK mutation or fail the explicitly requested observer."""
        if self._run is None or self._finished:
            return False
        try:
            action()
        except Exception as error:
            raise self._operation_failure(error, operation=operation) from error
        return True

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """
        Log one strictly increasing completed epoch with only approved keys.

        A resume trains the epochs after its restored ``last_checkpoint.pt`` again;
        an epoch an earlier session already logged keeps its first values and is
        not sent again, while the in-session order stays strict.
        """
        if self._run is None or self._finished:
            return
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            msg = "W&B completed-epoch history requires an integer epoch >= 1."
            raise TrackingError(msg)
        if self._last_logged_epoch is not None and epoch <= self._last_logged_epoch:
            msg = f"W&B completed-epoch history cannot rewrite epoch {epoch}. The last successful epoch is {self._last_logged_epoch}."
            raise TrackingError(msg)
        if self._resumed_through_epoch is not None and epoch <= self._resumed_through_epoch:
            self._last_logged_epoch = epoch
            return

        payload: dict[str, float | int] = {"epoch": epoch}
        definition_by_source = self.history_definition_by_source
        for key, value in metrics.items():
            definition = definition_by_source.get(key)
            if definition is not None:
                payload[definition.wandb_key] = float(value) * definition.multiplier

        if self._run_operation(
            "history",
            lambda: cast("_WandbRun", self._run).log(payload, step=epoch),
        ):
            self._last_logged_epoch = epoch
            self._persist({"last_logged_epoch": epoch})

    def log_evaluation_summary(self, evidence: Mapping[str, Any]) -> None:
        """Log one schema-restricted steady or transient Evaluation summary."""
        if self._run is None or self._finished:
            return
        if not isinstance(evidence, Mapping) or not evidence or len(evidence) > _MAX_EVALUATION_SUMMARY_ENTRIES:
            msg = f"W&B Evaluation summary must be one non-empty bounded mapping with at most {_MAX_EVALUATION_SUMMARY_ENTRIES} entries."
            raise TrackingError(msg)
        payload: dict[str, bool | float | int | str] = {}
        for key, value in evidence.items():
            admitted_key = _validate_evaluation_summary_entry(key, value)
            if isinstance(value, (bool, int)):
                payload[admitted_key] = value
            elif isinstance(value, float):
                if not math.isfinite(value):
                    msg = f"W&B Evaluation summary value {admitted_key!r} must be finite."
                    raise TrackingError(msg)
                payload[admitted_key] = value
            elif isinstance(value, str) and value and len(value) <= _MAX_EVALUATION_SUMMARY_TEXT:
                payload[admitted_key] = value
            else:
                msg = f"W&B Evaluation summary value {admitted_key!r} must be a bounded boolean, integer, finite float, or non-empty string."
                raise TrackingError(msg)

        def update_summary() -> None:
            """Replace the tracker-published Evaluation summary without writing history arrays."""
            summary = cast("Any", getattr(self._run, "summary", None))
            if not callable(getattr(summary, "update", None)):
                msg_0 = "The active W&B run does not expose summary updates."
                raise TrackingUploadError(msg_0)
            stale = sorted((self._published_summary_keys - payload.keys()) & set(summary.keys()))
            if stale and not hasattr(summary, "__delitem__"):
                msg_1 = "The active W&B run does not expose summary key removal."
                raise TrackingUploadError(msg_1)
            for key in stale:
                del summary[key]
            summary.update(payload)

        if self._run_operation("evaluation_summary", update_summary):
            self._published_summary_keys = frozenset(payload)
            self._persist(
                {
                    "evaluation_summary_keys": sorted(payload),
                    "evaluation_summary_entry_count": len(payload),
                }
            )

    def _validate_run_file(self, kind: str, path: Path) -> None:
        """Admit only one completed artifact-provenance file under this run."""
        if kind != "artifact_provenance":
            msg = f"Unsupported tracked file kind {kind!r}."
            raise TrackingUploadError(msg)
        resolved = path.resolve()
        try:
            resolved.relative_to((self.run_dir / "analysis").resolve())
        except ValueError as error:
            msg = "Artifact provenance must be inside the current run analysis root."
            raise TrackingUploadError(msg) from error
        if resolved.name != "artifact_provenance.json":
            msg = "Artifact provenance upload requires artifact_provenance.json."
            raise TrackingUploadError(msg)
        if not resolved.is_file():
            msg = f"Tracked file is not complete: {resolved}"
            raise FileNotFoundError(msg)

    def upload_files(self, files: Mapping[str, Path]) -> None:
        """Upload only explicitly selected evaluation provenance files."""
        if self._run is None or self._finished or not files:
            return
        if not bool(self.upload_settings["evaluation_artifacts"]):
            msg = "Evaluation-artifact upload is disabled by configuration."
            raise TrackingUploadError(msg)
        candidates = [(kind, Path(path)) for kind, path in files.items()]
        for kind, candidate in candidates:
            self._validate_run_file(kind, candidate)
        raw_save = getattr(self._run, "save", None)
        if not callable(raw_save):
            error = TrackingUploadError("The active W&B run does not expose bounded file upload.")
            raise self._operation_failure(error, operation="evaluation_provenance_upload")
        save = cast("Callable[..., Any]", raw_save)
        for _kind, candidate in candidates:

            def upload_candidate(candidate: Path = candidate) -> None:
                """Upload one prevalidated evaluation provenance file."""
                save(
                    str(candidate),
                    base_path=str(self.run_dir),
                    policy="now",
                )

            self._run_operation(
                "evaluation_provenance_upload",
                upload_candidate,
            )
        self._persist(
            {"uploaded_provenance_files": sorted(str(candidate.resolve().relative_to(self.run_dir.resolve())) for _kind, candidate in candidates)}
        )

    def upload_post_artifact(
        self,
        *,
        artifact_root: Path,
        media_files: Mapping[str, Path] | None = None,
        tables: Mapping[str, Any] | None = None,
    ) -> None:
        """Upload the exact curated evaluation bundle without scanning a directory."""
        files = dict(media_files or {})
        table_values = dict(tables or {})
        names = set(files).union(table_values)
        unsupported = sorted(names.difference(POST_ARTIFACT_MEDIA_KEYS))
        if unsupported:
            msg = f"Unsupported post-artifact media key(s): {unsupported}."
            raise TrackingUploadError(msg)
        if names and not bool(self.upload_settings["evaluation_artifacts"]):
            msg = "Evaluation-artifact upload is disabled by configuration."
            raise TrackingUploadError(msg)
        missing = sorted(POST_ARTIFACT_MEDIA_KEYS.difference(names))
        if names and missing:
            msg = f"Curated evaluation artifact is missing required key(s): {missing}."
            raise TrackingUploadError(msg)
        if set(files).intersection(table_values):
            msg = "Post-artifact keys cannot identify both a file and a table."
            raise TrackingUploadError(msg)
        if "run_summary_table" in files:
            msg = "run_summary_table must be supplied as an already-built table object."
            raise TrackingUploadError(msg)
        if any(name != "run_summary_table" for name in table_values):
            msg = "Only run_summary_table accepts a table object."
            raise TrackingUploadError(msg)
        if not names or self._run is None or self._finished:
            return

        root = Path(artifact_root).resolve()
        resolved_files: dict[str, Path] = {}
        for name, raw_path in files.items():
            candidate = Path(raw_path).resolve()
            if candidate.is_relative_to(root):
                msg = f"Curated media {name!r} must be rendered outside the immutable artifact cache."
                raise TrackingUploadError(msg)
            if candidate.suffix.lower() not in _POST_ARTIFACT_FILE_SUFFIXES or not candidate.is_file():
                msg = f"Curated media {name!r} is not an allowed complete rendered file."
                raise TrackingUploadError(msg)
            resolved_files[name] = candidate

        normalized_tables: dict[str, Any] = {}
        for name, table in table_values.items():
            value = table
            if isinstance(table, Mapping) and set(table) == {"columns", "data"}:
                columns = table["columns"]
                data = table["data"]
                if (
                    not isinstance(columns, Sequence)
                    or isinstance(columns, (str, bytes))
                    or any(not isinstance(column, str) or not column for column in columns)
                    or not isinstance(data, Sequence)
                    or isinstance(data, (str, bytes))
                ):
                    msg = "Neutral run_summary_table payload must contain string columns and row data."
                    raise TrackingUploadError(msg)
                table_factory = getattr(self._wandb, "Table", None)
                if not callable(table_factory):
                    error = TrackingUploadError("The active W&B SDK cannot serialize the run summary table.")
                    raise self._operation_failure(error, operation="evaluation_artifact_upload")
                value = table_factory(columns=list(columns), data=list(data))
            normalized_tables[name] = value

        raw_artifact_factory = getattr(self._wandb, "Artifact", None)
        raw_log_artifact = getattr(self._run, "log_artifact", None)
        if not callable(raw_artifact_factory) or not callable(raw_log_artifact):
            error = TrackingUploadError("The active W&B SDK does not expose evaluation artifact upload.")
            raise self._operation_failure(error, operation="evaluation_artifact_upload")
        artifact_factory = cast("Callable[..., _WandbArtifact]", raw_artifact_factory)
        log_artifact = cast("Callable[..., None]", raw_log_artifact)

        def upload_bundle() -> None:
            """Create and publish the prevalidated bounded evaluation artifact."""
            bundle = artifact_factory(
                name=f"{self.task_id}-{self.run_name}-curated-media",
                type="evaluation",
                metadata={"wandb_run_id": self.run_id, "inventory": sorted(names)},
            )
            for name, path in resolved_files.items():
                bundle.add_file(str(path), name=f"{name}{path.suffix.lower()}")
            for name, table in normalized_tables.items():
                bundle.add(table, name)
            log_artifact(bundle, aliases=["latest"])

        if self._run_operation("evaluation_artifact_upload", upload_bundle):
            self._uploaded_media = sorted(names)
            self._persist({"uploaded_media": self._uploaded_media})

    def _terminal_summary(  # noqa: C901, PLR0912
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None,
        local_summary: Mapping[str, Any] | None,
        error: BaseException | str | None,
    ) -> dict[str, Any]:
        """Build one concise terminal summary from authoritative local facts."""
        summary: dict[str, Any] = {
            "run/status": status,
            "run/name": self.run_name,
            "objective/id": self.objective_id,
            "objective/direction": self.objective_direction,
            "objective/selection_role": "id_validation",
            "tracking/status": "failed" if self._observer_failed else "finished",
            "tracking/mode": self.mode,
            "tracking/run_id": self.run_id,
        }
        if self._failed_operation is not None:
            summary["tracking/failed_operation"] = self._failed_operation
        task = self.semantic_config.get("task")
        if isinstance(task, Mapping):
            summary["task/id"] = task.get("id")
            summary["task/contract_digest"] = task.get("contract_digest")
        provenance = self.semantic_config.get("provenance")
        if isinstance(provenance, Mapping):
            summary["config/digest"] = provenance.get("config_digest")
        model = self.semantic_config.get("model")
        if isinstance(model, Mapping) and isinstance(model.get("parameter_counts"), Mapping):
            counts = cast("Mapping[str, Any]", model["parameter_counts"])
            summary["model/variant"] = model.get("variant")
            summary["model/parameters_total"] = counts.get("total")
            summary["model/parameters_trainable"] = counts.get("trainable")
        data = self.semantic_config.get("data")
        if isinstance(data, Mapping):
            split = data.get("split")
            if isinstance(split, Mapping):
                summary["data/split_sha256"] = split.get("artifact_sha256")
            normalization = data.get("normalization")
            if isinstance(normalization, Mapping):
                summary["data/normalizer_sha256"] = normalization.get("artifact_sha256")
        tuning = self.semantic_config.get("tuning")
        if isinstance(tuning, Mapping):
            for source, target in (
                ("study_name", "tuning/study_name"),
                ("study_role", "tuning/study_role"),
                ("trial_number", "tuning/trial_number"),
                ("training_seed", "tuning/training_seed"),
                ("sampler_seed", "tuning/sampler_seed"),
                ("search_signature", "tuning/search_signature"),
                ("sampled_parameters", "tuning/sampled_parameters"),
            ):
                if source in tuning:
                    summary[target] = copy.deepcopy(tuning[source])
            summary["tuning/final_state"] = status
            summary["Optuna/trial_number"] = tuning.get("trial_number")
            summary["Optuna/state"] = status
            summary["Optuna/pruned"] = status in {"pruned", "nonfinite_pruned", "oom_pruned"}

        if result is not None:
            for source, target in (
                ("completed_epoch", "run/completed_epoch"),
                ("global_step", "run/global_step"),
                ("selected_epoch", "selected/epoch"),
                ("terminal_epoch", "terminal/epoch"),
            ):
                if source in result:
                    summary[target] = result[source]
            selected_metrics = result.get("selected_metrics")
            if isinstance(selected_metrics, Mapping):
                summary.update({key: value for key, value in selected_metrics.items() if isinstance(key, str) and key.startswith("selected/")})
            terminal_metrics = result.get("terminal_metrics")
            if isinstance(terminal_metrics, Mapping):
                for key, value in terminal_metrics.items():
                    if isinstance(key, str) and key.startswith("train/"):
                        summary[f"terminal/{key}"] = value
        if local_summary is not None:
            for source, target in (
                ("effective_config_digest", "config/digest"),
                ("split_indices_sha256", "data/split_sha256"),
                ("normalizer_sha256", "data/normalizer_sha256"),
                ("elapsed_seconds", "run/duration_seconds"),
            ):
                if source in local_summary:
                    summary[target] = local_summary[source]
            if isinstance(local_summary.get("training_work"), Mapping):
                # The local summary holds the counts in every terminal state; W&B shows them as training_work.<count>.
                summary["training_work"] = dict(local_summary["training_work"])
            if isinstance(tuning, Mapping) and "elapsed_seconds" in local_summary:
                summary["Optuna/trial_duration_seconds"] = local_summary["elapsed_seconds"]
            checkpoint_digest = local_summary.get("best_checkpoint_sha256")
            if isinstance(checkpoint_digest, str):
                summary["selected/checkpoint_sha256_short"] = checkpoint_digest[:16]
            sessions = local_summary.get("runtime_sessions")
            if isinstance(sessions, list):
                resume_count = 0
                for session in sessions:
                    if not isinstance(session, Mapping):
                        continue
                    state = session.get("tracking")
                    if isinstance(state, Mapping) and state.get("session_kind") == "resume":
                        resume_count += 1
                summary["run/resume_count"] = resume_count
        if error is not None:
            context = _safe_error(error if isinstance(error, BaseException) else RuntimeError(error))
            summary["run/error_class"] = context["error_class"]
            summary["run/error_message"] = context["error_message"]
        return summary

    def finish(
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        local_summary: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Publish terminal facts and finish exactly once, failing closed."""
        if self._run is None or self._finished:
            return
        exit_code = 0 if status == "completed" else 1
        try:
            terminal = self._terminal_summary(
                status=status,
                result=result,
                local_summary=local_summary,
                error=error,
            )
            for key, value in terminal.items():
                self._run.summary[key] = value
            self._run.finish(exit_code=exit_code)
        except Exception as failure:
            self._finished = True
            tracking_error = self._operation_failure(failure, operation="finish")
            self._preserve_local_state()
            raise tracking_error from failure
        self._finished = True
        if self._observer_failed:
            self._preserve_local_state()
        terminal_tracking_status = "failed" if self._observer_failed else "finished"
        self._persist({"status": terminal_tracking_status, "finished_at": _utc_now()})


def initialize_wandb(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str,
    semantic_config: Mapping[str, Any] | None = None,
    resume: bool = False,
    persisted_run_id: str | None = None,
    previous_last_logged_epoch: int | None = None,
    previous_evaluation_summary_keys: Collection[str] = (),
    state_updater: TrackingStateUpdater | None = None,
) -> WandbSession:
    """Initialize the configured W&B mode with strict identity and metric rules."""
    objective = cast("Mapping[str, Any]", config["evaluation"])["objective"]
    objective_mapping = cast("Mapping[str, Any]", objective)
    objective_id = str(objective_mapping["id"])
    objective_direction = str(objective_mapping["direction"])
    evaluation_metrics = cast(
        "Sequence[Mapping[str, Any]]",
        cast("Mapping[str, Any]", config["evaluation"])["metrics"],
    )
    metric_ids = frozenset(str(metric["id"]) for metric in evaluation_metrics)
    settings = cast(
        "Mapping[str, Any]",
        cast("Mapping[str, Any]", config["tracking"])["wandb"],
    )
    path = Path(run_dir)
    run_name = str(cast("Mapping[str, Any]", config["run"])["name"])
    task_id = str(config["task"])
    upload_settings = cast("Mapping[str, Any]", settings["upload"])
    monitor_settings = cast("Mapping[str, Any]", settings["monitor"])
    mode = str(settings["mode"])
    workflow = str(settings["workflow"])

    if mode == "disabled":
        return WandbSession(
            None,
            None,
            objective_id,
            objective_direction,
            metric_ids,
            mode,
            None,
            path,
            run_name,
            task_id,
            upload_settings,
            semantic_config=copy.deepcopy(dict(semantic_config or {})),
        )

    group, job_type, wandb_name = wandb_schema.run_organization(config, workflow=workflow)
    runtime_tags = list(cast("Sequence[str]", settings["tags"]))
    run_id = persisted_run_id if resume else uuid.uuid4().hex
    if not isinstance(run_id, str) or not run_id:
        msg = "Exact W&B resume requires the persisted non-empty run ID."
        raise TrackingInitializationError(msg)
    resume_policy: str | None
    offline_fallback: str | None = None
    if mode == "offline":
        resume_policy = None
        if resume:
            offline_fallback = "offline_same_persisted_id_segment"
    else:
        resume_policy = "must" if resume else "never"

    base_state: dict[str, Any] = {
        "requested_mode": mode,
        "wandb_run_id": run_id,
        "project": settings["project"],
        "entity": settings["entity"],
        "tags": runtime_tags,
        "group": group,
        "job_type": job_type,
        "session_started_at": _utc_now(),
        "session_kind": "resume" if resume else "fresh",
        "status": "offline" if mode == "offline" else "active",
    }
    if offline_fallback is not None:
        base_state["offline_resume_fallback"] = offline_fallback
    if state_updater is not None:
        state_updater(base_state)

    if mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        error = RuntimeError("WANDB_API_KEY is missing or blank. Online tracking requires non-interactive environment authentication.")
        context = _safe_error(error)
        if state_updater is not None:
            state_updater(
                {
                    "status": "failed_before_start",
                    "failed_operation": "authentication",
                    **context,
                }
            )
        message = f"tracking.wandb.mode='online' authentication failed before epoch 1: {context['error_message']}"
        raise TrackingInitializationError(message) from error

    physics_config = cast(
        "Mapping[str, Any]",
        cast("Mapping[str, Any]", config["loss"])["physics"],
    )
    data_config = cast("Mapping[str, Any]", config["data"])
    ood_datasets = list(data_config["additional_evaluation_datasets"])
    runtime_config = semantic_config.get("runtime") if isinstance(semantic_config, Mapping) else None
    runtime_device = runtime_config.get("device") if isinstance(runtime_config, Mapping) else None
    resolved_device = runtime_device.get("resolved_device") if isinstance(runtime_device, Mapping) else None
    requested_device = cast("Mapping[str, Any]", config["run"]).get("device")
    cuda_enabled = (isinstance(resolved_device, str) and resolved_device.startswith("cuda")) or (
        resolved_device is None and isinstance(requested_device, str) and requested_device.startswith("cuda")
    )
    history_definitions = automatic_history_metric_definitions(
        evaluation_metrics,
        objective_id=objective_id,
        physics_training_enabled=bool(physics_config["enabled"]),
        physics_monitor_enabled=bool(monitor_settings["enabled"]),
        cuda_enabled=cuda_enabled,
        optuna_trial=workflow == "optuna_trial",
        task_id=task_id,
        state_aux_enabled=float(cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["loss"])["data"]).get("state_aux_weight", 0.0)) > 0.0,
        ood_enabled=bool(ood_datasets),
    )

    # Online SDK state lives on the process TMPDIR (node-local, removed at container exit) and is copied into the run folder
    # when the session fails; offline state stays in the run folder because `wandb sync` needs it.
    local_root = path if mode == "offline" else Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / f"wandb-{run_id}"
    sdk_run: _WandbRun | None = None
    try:
        wandb = cast("_WandbModule", importlib.import_module("wandb"))
        # Process-wide: the credential line is printed before per-run settings apply.
        wandb.setup(settings=wandb.Settings(silent=True))
        sdk_run = _require_initialized_run(
            wandb.init(
                project=str(settings["project"]),
                entity=cast("str | None", settings["entity"]),
                tags=None if resume and mode == "online" else runtime_tags,
                group=group,
                job_type=job_type,
                mode=mode,
                name=wandb_name,
                id=run_id,
                resume=resume_policy,
                dir=str(local_root),
                config=copy.deepcopy(dict(semantic_config or {})),
                save_code=False,
                settings={
                    "disable_git": True,
                    "disable_code": True,
                },
            )
        )
        required_tags = tuple(runtime_tags)
        existing_tags = tuple(str(tag) for tag in (sdk_run.tags or ()))
        merged_tags = (*existing_tags, *(tag for tag in required_tags if tag not in existing_tags))
        if merged_tags != existing_tags:
            sdk_run.tags = merged_tags

        sdk_run.define_metric("epoch", hidden=True, summary="none")
        for definition in history_definitions:
            # summary=None keeps the SDK default, the last value under the history key; "none" keeps a series out of the summary.
            sdk_run.define_metric(
                definition.wandb_key,
                step_metric="epoch",
                step_sync=False,
                summary=None if definition.in_summary else "none",
            )
        sdk_run.summary.update({"data/train_dataset": data_config["train_dataset"], "data/ood_datasets": ood_datasets})
    except Exception as error:
        if sdk_run is not None:
            with suppress(Exception):
                sdk_run.finish(exit_code=1)
        context = _safe_error(error)
        if state_updater is not None:
            state_updater(
                {
                    "status": "failed_before_start",
                    "failed_operation": "initialization",
                    **context,
                }
            )
        message = (
            f"tracking.wandb.mode={mode!r} initialization failed before epoch 1: "
            f"{context['error_class']}: {context['error_message']}. "
            "Verify W&B installation, local write access, and online authentication."
        )
        raise TrackingInitializationError(message) from error

    return WandbSession(
        sdk_run,
        wandb,
        objective_id,
        objective_direction,
        metric_ids,
        mode,
        run_id,
        path,
        run_name,
        task_id,
        upload_settings,
        semantic_config=copy.deepcopy(dict(semantic_config or {})),
        history_metric_definitions=history_definitions,
        state_updater=state_updater,
        _resumed_through_epoch=previous_last_logged_epoch,
        _published_summary_keys=frozenset(previous_evaluation_summary_keys),
        _local_root=None if mode == "offline" else local_root,
    )


def epoch_callback(
    session: WandbSession,
    optimizer: Optimizer | None = None,
) -> EpochEndCallback | None:
    """
    Bind an enabled W&B session to completed-epoch telemetry.

    Parameters
    ----------
    session : WandbSession
        Initialized or disabled observer session.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose first parameter-group learning rate is added when absent.

    Returns
    -------
    EpochEndCallback | None
        Callback for the training loop, or ``None`` for a disabled session.

    """
    if not session.enabled:
        return None

    def callback(epoch: int, metrics: dict[str, float]) -> None:
        """Add the current optimizer rate and forward one completed-epoch payload."""
        values = dict(metrics)
        if "optimization/learning_rate" not in values and optimizer is not None:
            parameter_groups = optimizer.param_groups
            if not parameter_groups:
                msg = "Cannot log W&B learning rate: optimizer has no parameter groups."
                raise RuntimeError(msg)
            values["optimization/learning_rate"] = float(parameter_groups[0]["lr"])
        session.log_epoch(epoch, values)

    return callback


def combine_epoch_callbacks(
    *callbacks: EpochEndCallback | None,
) -> EpochEndCallback | None:
    """
    Combine non-null completed-epoch consumers without changing their order.

    Parameters
    ----------
    *callbacks : EpochEndCallback | None
        Scheduler-independent lifecycle callbacks such as Optuna reporting and W&B.

    Returns
    -------
    EpochEndCallback | None
        Ordered composite, or ``None`` when every input is disabled.

    Notes
    -----
    Exceptions propagate immediately, so later consumers do not observe an epoch
    rejected by an earlier authoritative consumer.

    """
    active = tuple(callback for callback in callbacks if callback is not None)
    if not active:
        return None

    def combined(epoch: int, metrics: dict[str, float]) -> None:
        """Invoke lifecycle consumers in caller-specified order for one epoch."""
        for callback in active:
            callback(epoch, metrics)

    return combined
