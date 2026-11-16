"""Mirror authoritative local experiment state to optional W&B observability.

Responsibilities:
  - Keep disabled tracking free of SDK imports and tracking side effects
  - Persist opaque fresh identities and exact-resume identities locally first
  - Publish bounded semantic config, training-event history over the committed supervised transitions, and terminal summaries
  - Mirror authoritative validation and physics event values without recomputation
  - Fail closed on requested online or offline observer failures
  - Keep the SDK console silent and the online SDK directory node-local, preserved in the run folder after an observer failure
  - Keep built-in system telemetry secondary to scientific metrics, and bound to the job's allocated GPUs

Design principles:
  - Local config, split, normalizer, checkpoints, summaries and artifacts win
  - W&B observes training but never chooses or reconstructs local state
  - Normal histories contain genuine training events (validation points, epoch and budget ends) only. Final results are summaries
  - Fresh and exact-resume sessions have strict fail-closed identity semantics
  - Secrets, arbitrary environment state and incidental absolute paths are absent

This module does NOT:
  - Own training, checkpoint, scheduler, pruning, or local lifecycle decisions
  - Upload datasets, checkpoints, artifacts, or media (the SDK syncs only its own run metadata and system metrics)
  - Make remote availability or W&B state a prerequisite for local correctness
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from src import datasets, learning

from . import experiments_run_identity as run_identity
from . import experiments_wandb_schema as wandb_schema

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

EventCallback = Callable[[int, dict[str, float]], None]
TrackingStateUpdater = Callable[[Mapping[str, Any]], None]

_SECRET_KEY_PATTERN = re.compile(r"(?i)(WANDB_API_KEY|api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")
_MAX_SAFE_ERROR_LENGTH = 600


def automatic_history_metric_definitions(
    evaluation_metrics: Sequence[Mapping[str, Any]],
    *,
    objective_id: str,
    physics_training_enabled: bool,
    physics_monitor_enabled: bool,
    cuda_enabled: bool,
    optuna_trial: bool = False,
    task_id: str | None = None,
    rollout_curriculum: bool = False,
) -> tuple[wandb_schema.MetricProjection, ...]:
    """Return the ordered curated history contract; the schema module owns sections and key names."""
    return wandb_schema.curated_metric_projections(
        task_id,
        evaluation_metrics,
        objective_id=objective_id,
        physics_training_enabled=physics_training_enabled,
        physics_monitor_enabled=physics_monitor_enabled,
        # Training and Optuna evaluate no held-out role, so their history has no OOD series and no OOD page.
        ood_roles=(),
        cuda_enabled=cuda_enabled,
        optuna_trial=optuna_trial,
        rollout_curriculum=rollout_curriculum,
    )


class TrackingError(RuntimeError):
    """
    Base class for failures owned by the optional tracking boundary.

    These errors describe observer initialization or local offline persistence
    failure. They never represent local scientific-run validity.
    """


class TrackingInitializationError(TrackingError):
    """
    Represent failure to initialize an explicitly enabled session before epoch one.

    Raised after sanitized failure facts are persisted locally. No training
    telemetry has been accepted by the observer at this boundary.
    """


class TrackingIOError(TrackingError):
    """Represent loss of records in an explicitly requested tracking mode."""


class _WandbRun(Protocol):
    """Describe the SDK run surface used by the lifecycle adapter."""

    summary: MutableMapping[str, Any]
    tags: Sequence[str] | None
    url: str | None

    def define_metric(self, name: str, **kwargs: Any) -> Any:
        """Define one supported metric family and its x-axis contract."""

    def log(self, data: Mapping[str, Any]) -> None:
        """Log one training event; ``supervised_transitions`` in the data is the x-axis."""

    def finish(self, exit_code: int = 0) -> None:
        """Finalize the tracking run."""


class _WandbModule(Protocol):
    """Describe the lazily imported W&B module surface."""

    Settings: Callable[..., Any]

    def setup(self, *, settings: Any) -> Any:
        """Apply process-wide SDK settings."""

    def init(self, **kwargs: Any) -> _WandbRun | None:
        """Initialize one W&B run."""


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
        "artifact": "split_indices.pt",
        "artifact_sha256": split_indices_sha256,
        "train_subset": split.train_subset,
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
                "material_ood_datasets": copy.deepcopy(data_config["material_ood_datasets"]),
            },
            "loader": {
                key: copy.deepcopy(data_config.get(key))
                for key in ("evaluation_batch_size", "num_workers", "pin_memory", "persistent_workers", "spatial_stride")
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
    loader_keys = ("evaluation_batch_size", "num_workers", "pin_memory", "persistent_workers")
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
            "cadence": "validation_points",
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
    evaluation_payload["roles"] = {
        "selection": "id",
        "id_interpretation": "in_distribution_validation",
        # Training never evaluates ID-test or OOD; the separate artifact step owns them.
        "diagnostic": ["physics"],
        "event_model": "budget_validation_points",
        "validation_points": learning.training.budget.VALIDATION_POINTS,
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


def allocated_gpu_ids(visible_devices: str | None) -> list[int] | None:
    """
    Return the node GPU indices of the job's allocation for the W&B system monitor, or ``None`` to monitor every visible GPU.

    Slurm exposes the allocated GPUs as ``CUDA_VISIBLE_DEVICES`` with node indices; on nodes without device isolation the
    monitor otherwise records the other jobs' GPUs as well. Without the variable every GPU the process sees is its own.

    Raises
    ------
    ValueError
        If the variable names devices by anything but non-negative integer indices.

    """
    if visible_devices is None or not visible_devices.strip():
        return None
    entries = [entry.strip() for entry in visible_devices.split(",")]
    if not all(entry.isdigit() for entry in entries):
        msg = f"CUDA_VISIBLE_DEVICES must list integer GPU indices to bind the W&B system monitor, got {visible_devices!r}."
        raise ValueError(msg)
    return [int(entry) for entry in entries]


def persisted_wandb_identity(summary: Mapping[str, Any]) -> tuple[str, int | None, int | None]:
    """
    Recover the sole W&B identity and the latest successfully logged global step and validation point from local sessions.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Authoritative run summary containing append-only runtime session records.

    Returns
    -------
    tuple[str, int | None, int | None]
        Persisted W&B run ID, maximum locally recorded logged global step and maximum logged validation point.

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
    last_step: int | None = None
    last_point: int | None = None
    for raw_session in raw_sessions:
        if not isinstance(raw_session, Mapping):
            continue
        state = raw_session.get("tracking")
        if not isinstance(state, Mapping):
            continue
        run_id = state.get("wandb_run_id")
        if isinstance(run_id, str) and run_id:
            identities.add(run_id)
        raw_step = state.get("last_logged_step")
        if isinstance(raw_step, int) and not isinstance(raw_step, bool):
            last_step = raw_step if last_step is None else max(last_step, raw_step)
        raw_point = state.get("last_logged_point")
        if isinstance(raw_point, int) and not isinstance(raw_point, bool):
            last_point = raw_point if last_point is None else max(last_point, raw_point)
    if len(identities) != 1:
        msg = f"Exact W&B resume requires one persisted run ID, found {len(identities)}."
        raise TrackingInitializationError(msg)
    return next(iter(identities)), last_step, last_point


@dataclass(slots=True)
class WandbSession:
    """Own one optional, fail-closed W&B mirror of authoritative local state."""

    _run: _WandbRun | None
    objective_id: str
    objective_direction: str
    evaluation_metric_ids: frozenset[str]
    mode: str
    run_id: str | None
    run_dir: Path
    run_name: str
    semantic_config: Mapping[str, Any] = field(default_factory=dict)
    history_metric_definitions: tuple[wandb_schema.MetricProjection, ...] = ()
    state_updater: TrackingStateUpdater | None = None
    _resumed_through_step: int | None = None
    _resumed_through_point: int | None = None
    _last_logged_step: int | None = None
    _finished: bool = False
    _observer_failed: bool = False
    _failed_operation: str | None = None
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

    def log_event(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Log one training event at its strictly increasing global step with only approved keys.

        A resume trains the groups after its restored ``last_checkpoint.pt`` again;
        the training series of a step an earlier session already logged keep
        their first values and are not sent again, while the in-session order
        stays strict. The validation series are identified by their
        ``validation/point`` index instead: a point an earlier session logged is
        not sent again, at any step, and a new point is sent even at a lower one.
        The point is persisted before it is sent, so a crash in between loses at
        most that point and never duplicates it. History is logged without W&B's
        own step, which only counts calls; ``supervised_transitions`` (the stage's committed supervised transitions,
        a steady run's seen cases) in the payload is the x-axis, so curves of different GPUs and memory batches align.
        """
        # Checked before the disabled return, so runs without W&B also hold the producer to the axis contract.
        if "budget/supervised_transitions" not in metrics:
            msg = "W&B training-event history requires budget/supervised_transitions, the x-axis."
            raise TrackingError(msg)
        if self._run is None or self._finished:
            return
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            msg = "W&B training-event history requires an integer global step >= 1."
            raise TrackingError(msg)
        if self._last_logged_step is not None and step <= self._last_logged_step:
            msg = f"W&B training-event history cannot rewrite step {step}. The last successful step is {self._last_logged_step}."
            raise TrackingError(msg)
        raw_point = metrics.get("validation/point")
        point = None if raw_point is None else int(raw_point)
        send_training = self._resumed_through_step is None or step > self._resumed_through_step
        send_validation = point is not None and (self._resumed_through_point is None or point > self._resumed_through_point)

        payload: dict[str, float | int] = {"global_step": step, "supervised_transitions": int(metrics["budget/supervised_transitions"])}
        # Projection order, not producer order, so first-appearance panel ordering in W&B follows the layout of the sections.
        for definition in self.history_metric_definitions:
            if definition.source_key in metrics and (send_validation if definition.validation else send_training):
                payload[definition.wandb_key] = float(metrics[definition.source_key]) * definition.multiplier
        self._last_logged_step = step
        if payload.keys() <= {"global_step", "supervised_transitions"}:  # only the hidden axes
            return

        if send_validation:
            self._persist({"last_logged_point": point})
        if self._run_operation("history", lambda: cast("_WandbRun", self._run).log(payload)) and send_training:
            self._persist({"last_logged_step": step})

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
        training_outcome = result.get("training_outcome") if result is not None else None
        if isinstance(training_outcome, Mapping):
            summary["run/outcome"] = str(training_outcome["status"])
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
            # A trial whose training ended states its training outcome: a diverged or ineligible trial is pruned in its study.
            trained = isinstance(training_outcome, Mapping) and status in {"completed", "unsuccessful"}
            optuna_state = str(cast("Mapping[str, Any]", training_outcome)["status"]) if trained else status
            summary["Optuna/state"] = optuna_state
            summary["Optuna/pruned"] = optuna_state in wandb_schema.PRUNED_OUTCOMES

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
                # Only the series the history projects; summary.json keeps every selected metric.
                projected = {
                    key
                    for definition in self.history_metric_definitions
                    if (key := wandb_schema.selected_summary_key(definition.source_key)) is not None
                }
                summary.update({key: value for key, value in selected_metrics.items() if key in projected})
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
    previous_last_logged_step: int | None = None,
    previous_last_logged_point: int | None = None,
    state_updater: TrackingStateUpdater | None = None,
) -> WandbSession:
    """Initialize the configured W&B mode with strict identity and metric rules; the history has no held-out series."""
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
    monitor_settings = cast("Mapping[str, Any]", settings["monitor"])
    mode = str(settings["mode"])
    workflow = str(settings["workflow"])

    if mode == "disabled":
        return WandbSession(
            None,
            objective_id,
            objective_direction,
            metric_ids,
            mode,
            None,
            path,
            run_name,
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
    ood_datasets = list(data_config["material_ood_datasets"])
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
        # The resolved training section carries a curriculum only for the stage that draws rollout horizons (Stage B).
        rollout_curriculum="curriculum" in cast("Mapping[str, Any]", config["training"]),
    )

    # Online SDK state lives on the process TMPDIR (node-local, removed at container exit) and is copied into the run folder
    # when the session fails; offline state stays in the run folder because `wandb sync` needs it.
    local_root = path if mode == "offline" else Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / f"wandb-{run_id}"
    sdk_run: _WandbRun | None = None
    try:
        wandb = cast("_WandbModule", importlib.import_module("wandb"))
        # Process-wide: the credential line is printed before per-run settings apply.
        wandb.setup(settings=wandb.Settings(silent=True))
        sdk_settings: dict[str, Any] = {"disable_git": True, "disable_code": True}
        if cuda_enabled and (gpu_ids := allocated_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))) is not None:
            sdk_settings["x_stats_gpu_device_ids"] = gpu_ids
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
                settings=sdk_settings,
            )
        )
        required_tags = tuple(runtime_tags)
        existing_tags = tuple(str(tag) for tag in (sdk_run.tags or ()))
        merged_tags = (*existing_tags, *(tag for tag in required_tags if tag not in existing_tags))
        if merged_tags != existing_tags:
            sdk_run.tags = merged_tags

        sdk_run.define_metric("global_step", hidden=True, summary="none")
        sdk_run.define_metric("supervised_transitions", hidden=True, summary="none")
        for name in wandb_schema.SUMMARY_ONLY_KEYS:
            sdk_run.define_metric(name, hidden=True)
        for definition in history_definitions:
            # summary=None keeps the SDK default, the last value under the history key; "none" keeps a series out of the summary.
            sdk_run.define_metric(
                definition.wandb_key,
                step_metric="supervised_transitions",
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
        objective_id,
        objective_direction,
        metric_ids,
        mode,
        run_id,
        path,
        run_name,
        semantic_config=copy.deepcopy(dict(semantic_config or {})),
        history_metric_definitions=history_definitions,
        state_updater=state_updater,
        _resumed_through_step=previous_last_logged_step,
        _resumed_through_point=previous_last_logged_point,
        _local_root=None if mode == "offline" else local_root,
    )


def event_callback(
    session: WandbSession,
    optimizer: Optimizer | None = None,
) -> EventCallback | None:
    """
    Bind an enabled W&B session to training-event telemetry.

    Parameters
    ----------
    session : WandbSession
        Initialized or disabled observer session.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose first parameter-group learning rate is added when absent.

    Returns
    -------
    EventCallback | None
        Callback for the training loop, or ``None`` for a disabled session.

    """
    if not session.enabled:
        return None

    def callback(step: int, metrics: dict[str, float]) -> None:
        """Add the current optimizer rate and forward one training-event payload."""
        values = dict(metrics)
        if "optimization/learning_rate" not in values and optimizer is not None:
            parameter_groups = optimizer.param_groups
            if not parameter_groups:
                msg = "Cannot log W&B learning rate: optimizer has no parameter groups."
                raise RuntimeError(msg)
            values["optimization/learning_rate"] = float(parameter_groups[0]["lr"])
        session.log_event(step, values)

    return callback


def combine_event_callbacks(
    *callbacks: EventCallback | None,
) -> EventCallback | None:
    """
    Combine non-null training-event consumers without changing their order.

    Parameters
    ----------
    *callbacks : EventCallback | None
        Scheduler-independent lifecycle callbacks such as console, Optuna reporting and W&B.

    Returns
    -------
    EventCallback | None
        Ordered composite, or ``None`` when every input is disabled.

    Notes
    -----
    Exceptions propagate immediately, so later consumers do not observe an event
    rejected by an earlier authoritative consumer.

    """
    active = tuple(callback for callback in callbacks if callback is not None)
    if not active:
        return None

    def combined(step: int, metrics: dict[str, float]) -> None:
        """Invoke lifecycle consumers in caller-specified order for one event."""
        for callback in active:
            callback(step, metrics)

    return combined
