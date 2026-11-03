"""Serve explicit physical transient-drying inference from completed run evidence.

Responsibilities:
  - Reconstruct a transient model, tensorizer, and fitted scaling artifact from one completed run
  - Validate explicit physical one-step and rollout request evidence
  - Apply the admitted airflow-source policy without mutating caller tensors
  - Execute stateless or recurrent model calls and return physical reconstructed states with factual timing
  - Build the RNO hidden state over reference prefixes for warm one-step and rolling-origin diagnostics

Design principles:
  - Completed-run admission is the sole persisted-artifact read boundary
  - Tensorizer and scaling artifacts remain authoritative for channel assembly and reconstruction
  - Public recurrent requests start with no hidden state and carry it only within that request
  - Diagnostic requests may start from a state the caller built over the same trajectory's reference prefix
  - Model timing excludes request validation, transfer, tensorization, scaling, and reconstruction

This module does NOT:
  - Load Dataset packages, resolve saved splits, or refit scaling state
  - Alter the steady inference tuple API
  - Claim speedup, benchmark performance, or execute training
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

from src import domain, experiments, learning
from src.learning.models import learning_models_factory as model_factory
from src.learning.transient import learning_transient_rollout as rollout
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec
from src.learning.transient.learning_transient_scaling import TransientScalingArtifact
from src.learning.transient.learning_transient_tensorizer import TransientTensorizer

if TYPE_CHECKING:
    from pathlib import Path

_COMSOL_REFERENCE_AIRFLOW_SOURCE = "comsol_reference"
_AIRFLOW_SOURCES = {_COMSOL_REFERENCE_AIRFLOW_SOURCE, "external"}
_MODEL_KINDS = {"fno", "uno", "rno"}
_PRECISION = "float32"
_TIME_ATOL = 1.0e-6
_PREPARED_REQUEST_TOKEN = object()


@dataclass(frozen=True, slots=True)
class TransientInferenceContext:
    """Hold one completed-run transient model and its admitted preprocessing evidence."""

    model: nn.Module
    tensorizer: TransientTensorizer
    scaling: TransientScalingArtifact
    device: torch.device
    model_kind: Literal["fno", "uno", "rno"]
    precision: Literal["float32"]
    training_spatial_shape: tuple[int, int]
    evaluation_spatial_shape: tuple[int, int]
    architecture_spatial_compatibility: Mapping[str, Any]
    scaling_spatial_compatibility: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TransientPreparedRequest:
    """Hold one strictly admitted transient request on its inference device."""

    context: TransientInferenceContext
    state: torch.Tensor
    static: torch.Tensor
    boundary: torch.Tensor
    scalars: torch.Tensor
    t_n: torch.Tensor
    t_next: torch.Tensor
    dt: torch.Tensor
    _versions: tuple[int, ...] = field(repr=False)
    _owner_token: object = field(repr=False)

    def __post_init__(self) -> None:
        """Reject construction outside the strict preparation owner."""
        if self._owner_token is not _PREPARED_REQUEST_TOKEN:
            msg = "Prepared transient requests must come from strict admission."
            raise TypeError(msg)

    @property
    def length(self) -> int:
        """Return the exact request transition count."""
        return int(self.boundary.shape[1])


@dataclass(frozen=True, slots=True)
class TransientTiming:
    """Describe factual elapsed model-call time for one transient request."""

    model_seconds: float
    device: str
    precision: Literal["float32"]
    mode: Literal["one_step", "autonomous_rollout"]
    model_calls: int


@dataclass(frozen=True, slots=True)
class TransientStepResult:
    """Return one physical next state, its scaled increment, and model timing."""

    next_state: torch.Tensor
    scaled_delta: torch.Tensor
    timing: TransientTiming
    decoded_delta: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TransientRolloutResult:
    """Return autonomous physical states, scaled increments, and model timing."""

    states: torch.Tensor
    scaled_deltas: torch.Tensor
    timing: TransientTiming
    decoded_deltas: torch.Tensor | None = None
    prediction_available: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TransientTeacherForcedResult:
    """
    Return warm one-step diagnostics over a reference trajectory.

    Step ``k`` predicts from the reference state ``k`` with the hidden state
    built over reference steps ``0..k-1``. ``captured_hidden`` maps requested
    step counts to that state; ``0`` is the model's own initial state.
    """

    next_states: torch.Tensor
    scaled_deltas: torch.Tensor
    decoded_deltas: torch.Tensor
    step_seconds: tuple[float, ...]
    timing: TransientTiming
    captured_hidden: Mapping[int, object | None]


_STATE_BATCH_RANK = 4
_STATE_CHANNEL_COUNT = 4
_BOUNDARY_CHANNEL_COUNT = 9
_BOUNDARY_BATCH_RANK = 3


def _require_precision(value: str) -> Literal["float32"]:
    """Require the sole supported explicit transient inference precision."""
    if value != _PRECISION:
        msg = "Transient inference supports only precision 'float32'."
        raise ValueError(msg)
    return cast('Literal["float32"]', _PRECISION)


def _require_model_kind(value: object) -> Literal["fno", "uno", "rno"]:
    """Require one persisted transient model kind."""
    if value not in _MODEL_KINDS:
        msg = "Transient inference model.kind must be 'fno', 'uno', or 'rno'."
        raise ValueError(msg)
    return cast('Literal["fno", "uno", "rno"]', value)


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve one concrete inference device without implicit fallback."""
    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, str):
        resolved = learning.device.resolve_device(device, path="transient inference device").device
    else:
        msg = "Transient inference device must be one string policy or torch.device."
        raise TypeError(msg)
    if resolved.type not in {"cpu", "cuda"}:
        msg = "Transient inference requires one concrete CPU or CUDA device."
        raise ValueError(msg)
    if resolved.type == "cuda" and resolved.index is None:
        msg = "Transient inference requires an indexed CUDA device."
        raise ValueError(msg)
    return resolved


def _place_model_float32(model: nn.Module, *, device: torch.device) -> nn.Module:
    """Place real and complex model state at float32-equivalent precision."""
    model.to(device=device)
    converted_state: dict[str, object] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            converted_state[name] = value
        elif value.is_complex():
            converted_state[name] = value.to(device=device, dtype=torch.complex64)
        elif value.is_floating_point():
            converted_state[name] = value.to(device=device, dtype=torch.float32)
        else:
            converted_state[name] = value.to(device=device)
    model.load_state_dict(converted_state, strict=True, assign=True)
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if value.device != device:
            msg = f"Transient inference model tensor {name!r} is not on the resolved device."
            raise RuntimeError(msg)
        expected = torch.complex64 if value.is_complex() else torch.float32 if value.is_floating_point() else value.dtype
        if value.dtype != expected:
            msg = f"Transient inference model tensor {name!r} violates float32-equivalent precision."
            raise RuntimeError(msg)
    return model


def _spatial_preflight_error(
    *,
    model_kind: object,
    training_shape: tuple[int, int],
    evaluation_shape: tuple[int, int],
    training_stride: object,
    checkpoint_identity: object,
    reason: str,
) -> ValueError:
    """Build one actionable artifact-grid incompatibility error."""
    return ValueError(
        "Transient artifact Evaluation-grid preflight rejected the requested spatial representation: "
        f"architecture={model_kind!r}, checkpoint={checkpoint_identity!r}, "
        f"Training shape={training_shape}, Evaluation shape={evaluation_shape}, reason={reason}. "
        f"Use the common supported --evaluation-spatial-stride {training_stride!r} "
        "for compared models or choose another explicitly validated common grid."
    )


def _prove_cross_resolution_output_shape(
    *,
    model: nn.Module,
    model_kind: Literal["fno", "uno", "rno"],
    config: Mapping[str, Any],
    device: torch.device,
    training_shape: tuple[int, int],
    evaluation_shape: tuple[int, int],
    training_stride: object,
    checkpoint_identity: object,
) -> dict[str, Any]:
    """Execute one synthetic model step and require exact requested output shape."""
    model_config = config.get("model")
    params = model_config.get("params") if isinstance(model_config, Mapping) else None
    input_channels = params.get("in_channels") if isinstance(params, Mapping) else None
    if isinstance(input_channels, bool) or not isinstance(input_channels, int) or input_channels < 1:
        msg = "Transient model preflight requires a positive resolved in_channels value."
        raise TypeError(msg)
    probe = torch.zeros(
        (1, input_channels, *evaluation_shape),
        device=device,
        dtype=torch.float32,
    )
    try:
        with torch.inference_mode():
            prediction, _hidden = rollout.predict_step(
                model,
                probe,
                model_kind=model_kind,
                hidden=None,
            )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
        raise _spatial_preflight_error(
            model_kind=model_kind,
            training_shape=training_shape,
            evaluation_shape=evaluation_shape,
            training_stride=training_stride,
            checkpoint_identity=checkpoint_identity,
            reason=f"synthetic requested-grid forward failed: {type(error).__name__}: {error}",
        ) from error
    return {
        "kind": "synthetic_single_step_exact_output",
        "input_shape": list(probe.shape),
        "output_shape": list(prediction.shape),
        "hidden_state_scope": "single_request" if model_kind == "rno" else "not_applicable",
    }


def build_transient_inference_context(
    *,
    completed: Mapping[str, Any],
    device: str | torch.device = "cpu",
    precision: str = _PRECISION,
    evaluation_spatial_shape: tuple[int, int] | None = None,
) -> TransientInferenceContext:
    """
    Build one inference context from already admitted completed-run evidence.

    The caller retains ownership of completed-run admission. This construction
    validates tensorizer, scaler, architecture, checkpoint, requested grid, and
    cross-resolution output shape without reopening persisted run evidence.
    """
    resolved_precision = _require_precision(precision)
    config = completed.get("config")
    if not isinstance(config, Mapping) or config.get("task") != "transient_drying":
        msg = "Transient inference requires admitted completed transient_drying evidence."
        raise RuntimeError(msg)
    if experiments.run.is_trial_config(config):
        msg = "Transient inference is unavailable for an Optuna trial: trials keep no checkpoints. Train the selected configuration as a normal run."
        raise RuntimeError(msg)
    normalizer_state = completed.get("normalizer_state")
    if not isinstance(normalizer_state, Mapping):
        msg = "Completed transient run must provide one scaling-artifact mapping."
        raise TypeError(msg)
    scaling = TransientScalingArtifact.from_state_dict(normalizer_state)
    resolved_device = _resolve_device(device)
    scaling = scaling.to(resolved_device)
    training_shape = scaling.spatial_shape
    requested_shape = training_shape if evaluation_spatial_shape is None else evaluation_spatial_shape
    scaling_compatibility = scaling.assess_spatial_compatibility(requested_shape)
    architecture_compatibility = model_factory.assess_transient_model_spatial_compatibility(
        config,
        training_shape=training_shape,
        evaluation_shape=requested_shape,
    )
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        msg = "Completed transient run must provide model configuration."
        raise TypeError(msg)
    kind = _require_model_kind(model_config.get("kind"))
    decision = architecture_compatibility.get("decision")
    training_stride = scaling.dataset_identity.get("spatial_stride")
    checkpoint_identity = {
        "identity": completed.get("checkpoint_identity"),
        "best_checkpoint_sha256": completed.get("selected_checkpoint_sha256"),
        "best_checkpoint_epoch": completed.get("selected_checkpoint_epoch"),
    }
    if decision not in {"SUPPORTED_EXACTLY", "SUPPORTED_WITH_CONTRACT"}:
        raise _spatial_preflight_error(
            model_kind=kind,
            training_shape=training_shape,
            evaluation_shape=requested_shape,
            training_stride=training_stride,
            checkpoint_identity=checkpoint_identity,
            reason=str(architecture_compatibility.get("reason", decision)),
        )
    spec = TransientTensorizerSpec.from_mapping(
        {
            "input_profile": config.get("input_profile"),
            "temporal_conditioning": config.get("temporal", {}).get("temporal_conditioning") if isinstance(config.get("temporal"), Mapping) else None,
        }
    )
    if spec != scaling.tensorizer:
        msg = "Completed transient config tensorizer disagrees with admitted scaling artifact."
        raise RuntimeError(msg)
    model = model_factory.build_model(dict(config), device=resolved_device)
    best_checkpoint = completed.get("best_checkpoint")
    if (
        not isinstance(best_checkpoint, Mapping)
        or best_checkpoint.get("schema_version") != learning.training.checkpoint.CHECKPOINT_SCHEMA_VERSION
        or not isinstance(best_checkpoint.get("model_state_dict"), Mapping)
    ):
        msg = "Completed transient run must provide an admitted best model state dict."
        raise TypeError(msg)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model = _place_model_float32(model, device=resolved_device)
    model.eval()
    architecture_evidence = dict(architecture_compatibility)
    if requested_shape != training_shape:
        architecture_evidence["forward_output_shape_proof"] = _prove_cross_resolution_output_shape(
            model=model,
            model_kind=kind,
            config=config,
            device=resolved_device,
            training_shape=training_shape,
            evaluation_shape=requested_shape,
            training_stride=training_stride,
            checkpoint_identity=checkpoint_identity,
        )
    return TransientInferenceContext(
        model=model,
        tensorizer=TransientTensorizer(spec, scaling),
        scaling=scaling,
        device=resolved_device,
        model_kind=kind,
        precision=resolved_precision,
        training_spatial_shape=training_shape,
        evaluation_spatial_shape=requested_shape,
        architecture_spatial_compatibility=architecture_evidence,
        scaling_spatial_compatibility=scaling_compatibility,
    )


def load_transient_inference_context(
    *,
    run_dir: str | Path,
    device: str | torch.device = "cpu",
    precision: str = _PRECISION,
    evaluation_spatial_shape: tuple[int, int] | None = None,
) -> TransientInferenceContext:
    """
    Reconstruct one transient inference context from an admitted completed run.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Completed transient run directory admitted by experiments.run.
    device : str or torch.device, optional
        Explicit device policy or concrete device. Default is "cpu".
    precision : str, optional
        Explicit precision. Only "float32" is supported.
    evaluation_spatial_shape : tuple[int, int] | None, optional
        Exact requested Evaluation shape. Omission retains the Training shape.

    Returns
    -------
    TransientInferenceContext
        Loaded model plus fitted preprocessing and explicit grid compatibility.

    """
    completed = experiments.run.validate_completed_run(run_dir)
    return build_transient_inference_context(
        completed=completed,
        device=device,
        precision=precision,
        evaluation_spatial_shape=evaluation_spatial_shape,
    )


def _require_request_tensor(value: object, *, label: str) -> torch.Tensor:
    """Require one finite float32 tensor without moving or copying caller data."""
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        msg = f"Transient inference {label} must be one torch.float32 tensor."
        raise TypeError(msg)
    if not bool(torch.isfinite(value).all().item()):
        msg = f"Transient inference {label} must contain only finite values."
        raise ValueError(msg)
    return value


def _resolve_airflow_source(value: str | None) -> str:
    """Resolve the registered default airflow policy and reject task-policy drift."""
    task = domain.tasks.registry.get_task("transient_drying")
    declared = task.training_airflow_source
    if declared != _COMSOL_REFERENCE_AIRFLOW_SOURCE or declared not in _AIRFLOW_SOURCES:
        msg = "Transient task airflow policy must declare exactly 'comsol_reference'."
        raise RuntimeError(msg)
    source = declared if value is None else value
    if source not in _AIRFLOW_SOURCES:
        msg = "airflow_source must be exactly 'comsol_reference' or 'external'."
        raise ValueError(msg)
    return source


def _validate_request(
    *,
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    scalars: torch.Tensor,
    t_n: torch.Tensor,
    t_next: torch.Tensor,
    dt: torch.Tensor,
    external_airflow: torch.Tensor | None,
    airflow_source: str,
    spatial_shape: tuple[int, int],
    horizon: float,
) -> tuple[int, int, int, int]:
    """Validate exact source-side physical transient request evidence."""
    tensors = {
        "state": state,
        "static": static,
        "boundary": boundary,
        "scalars": scalars,
        "t_n": t_n,
        "t_next": t_next,
        "dt": dt,
    }
    for label, value in tensors.items():
        _require_request_tensor(value, label=label)
    if state.ndim != _STATE_BATCH_RANK or state.shape[1] != _STATE_CHANNEL_COUNT:
        msg = "Transient inference state must have shape [B,4,Y,X]."
        raise ValueError(msg)
    batch_size, _, height, width = state.shape
    if min(batch_size, height, width) < 1:
        msg = "Transient inference state must have non-empty B, Y, and X axes."
        raise ValueError(msg)
    if (height, width) != spatial_shape:
        msg = "Transient inference state spatial shape must match the admitted Evaluation grid."
        raise ValueError(msg)
    if static.shape != (batch_size, 7, height, width):
        msg = "Transient inference static must have shape [B,7,Y,X]."
        raise ValueError(msg)
    if (
        boundary.ndim != _BOUNDARY_BATCH_RANK
        or boundary.shape[0] != batch_size
        or boundary.shape[2] != _BOUNDARY_CHANNEL_COUNT
        or boundary.shape[1] < 1
    ):
        msg = "Transient inference boundary must have non-empty shape [B,L,9]."
        raise ValueError(msg)
    length = int(boundary.shape[1])
    if scalars.shape != (batch_size, 8):
        msg = "Transient inference scalars must have shape [B,8]."
        raise ValueError(msg)
    if any(value.shape != (batch_size, length) for value in (t_n, t_next, dt)):
        msg = "Transient inference t_n, t_next, and dt must each have shape [B,L]."
        raise ValueError(msg)
    source_device = state.device
    if any(value.device != source_device for value in tensors.values()):
        msg = "Transient inference request tensors must share one source device."
        raise ValueError(msg)
    if airflow_source == "comsol_reference":
        if external_airflow is not None:
            msg = "comsol_reference airflow does not accept external_airflow."
            raise ValueError(msg)
    else:
        external = _require_request_tensor(external_airflow, label="external_airflow")
        if external.shape != (batch_size, 3, height, width) or external.device != source_device:
            msg = "external_airflow must have shape [B,3,Y,X] on the request source device."
            raise ValueError(msg)
    ones = torch.ones_like(dt)
    if not torch.allclose(dt, ones, rtol=0.0, atol=_TIME_ATOL) or not torch.allclose(t_next - t_n, dt, rtol=0.0, atol=_TIME_ATOL):
        msg = "Transient inference requires exact regular fixed 1h time evidence."
        raise ValueError(msg)
    if bool((t_n < -_TIME_ATOL).any()) or bool((t_next > horizon + _TIME_ATOL).any()):
        msg = "Transient inference time evidence must remain within the admitted scaling horizon."
        raise ValueError(msg)
    return batch_size, length, height, width


def _owned_runtime_inputs(
    *,
    context: TransientInferenceContext,
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    scalars: torch.Tensor,
    t_n: torch.Tensor,
    t_next: torch.Tensor,
    dt: torch.Tensor,
    external_airflow: torch.Tensor | None,
    airflow_source: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place validated request data on the context device and apply airflow policy."""
    runtime_values = tuple(value.to(device=context.device, dtype=torch.float32) for value in (state, static, boundary, scalars, t_n, t_next, dt))
    runtime_state, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, runtime_t_next, runtime_dt = runtime_values
    if airflow_source == "external":
        if external_airflow is None:
            msg = "Validated external airflow unexpectedly missing."
            raise RuntimeError(msg)
        runtime_airflow = external_airflow.to(
            device=context.device,
            dtype=torch.float32,
        )
        runtime_static = runtime_static.clone()
        runtime_static[:, 2:5] = runtime_airflow
    return runtime_state, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, runtime_t_next, runtime_dt


def prepare_transient_request(
    context: TransientInferenceContext,
    *,
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    scalars: torch.Tensor,
    t_n: torch.Tensor,
    t_next: torch.Tensor,
    dt: torch.Tensor,
    airflow_source: str | None = None,
    external_airflow: torch.Tensor | None = None,
) -> TransientPreparedRequest:
    """Strictly validate one request once and place its tensors for reuse."""
    if not isinstance(context, TransientInferenceContext):
        msg = "context must be one TransientInferenceContext."
        raise TypeError(msg)
    resolved_airflow_source = _resolve_airflow_source(airflow_source)
    _validate_request(
        state=state,
        static=static,
        boundary=boundary,
        scalars=scalars,
        t_n=t_n,
        t_next=t_next,
        dt=dt,
        external_airflow=external_airflow,
        airflow_source=resolved_airflow_source,
        spatial_shape=context.evaluation_spatial_shape,
        horizon=context.scaling.horizon,
    )
    values = _owned_runtime_inputs(
        context=context,
        state=state,
        static=static,
        boundary=boundary,
        scalars=scalars,
        t_n=t_n,
        t_next=t_next,
        dt=dt,
        external_airflow=external_airflow,
        airflow_source=resolved_airflow_source,
    )
    owned = tuple(value.clone() for value in values)
    runtime_state, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, runtime_t_next, runtime_dt = owned
    return TransientPreparedRequest(
        context=context,
        state=runtime_state,
        static=runtime_static,
        boundary=runtime_boundary,
        scalars=runtime_scalars,
        t_n=runtime_t_n,
        t_next=runtime_t_next,
        dt=runtime_dt,
        _versions=tuple(value._version for value in owned),  # noqa: SLF001 -- PyTorch mutation counters and owned request versions protect input identity
        _owner_token=_PREPARED_REQUEST_TOKEN,
    )


def _prepared_tensors(
    request: TransientPreparedRequest,
) -> tuple[torch.Tensor, ...]:
    """Return prepared tensors in their immutable-evidence order."""
    return (
        request.state,
        request.static,
        request.boundary,
        request.scalars,
        request.t_n,
        request.t_next,
        request.dt,
    )


def _require_prepared_request(
    context: TransientInferenceContext,
    request: TransientPreparedRequest,
) -> None:
    """Require one request created for the exact active inference context."""
    if not isinstance(request, TransientPreparedRequest) or request.context is not context:
        msg = "Prepared transient request must belong to the exact inference context."
        raise TypeError(msg)
    tensors = _prepared_tensors(request)
    if any(value.device != context.device or value.dtype != torch.float32 for value in tensors):
        msg = "Prepared transient request placement drifted from its inference context."
        raise RuntimeError(msg)
    if tuple(value._version for value in tensors) != request._versions:  # noqa: SLF001 -- PyTorch mutation counters and owned request versions protect input identity
        msg = "Prepared transient request tensors were mutated after strict admission."
        raise RuntimeError(msg)


def window_transient_prepared_request(
    request: TransientPreparedRequest,
    *,
    origin: int,
    length: int,
    state: torch.Tensor | None = None,
) -> TransientPreparedRequest:
    """Return one zero-copy case-local window from already-admitted runtime inputs."""
    if not isinstance(request, TransientPreparedRequest):
        msg = "request must be one TransientPreparedRequest."
        raise TypeError(msg)
    _require_prepared_request(request.context, request)
    if (
        isinstance(origin, bool)
        or not isinstance(origin, int)
        or isinstance(length, bool)
        or not isinstance(length, int)
        or origin < 0
        or length < 1
        or origin + length > request.length
    ):
        msg = "Prepared transient window must lie inside the admitted request."
        raise ValueError(msg)
    selected_state = request.state if state is None else state
    if origin and state is None:
        msg = "A nonzero prepared transient origin requires its exact current state."
        raise ValueError(msg)
    if (
        not isinstance(selected_state, torch.Tensor)
        or selected_state.shape != request.state.shape
        or selected_state.device != request.context.device
        or selected_state.dtype != torch.float32
    ):
        msg = "Prepared transient window state contradicts runtime shape or placement."
        raise ValueError(msg)
    stop = origin + length
    tensors = (
        selected_state,
        request.static,
        request.boundary[:, origin:stop],
        request.scalars,
        request.t_n[:, origin:stop],
        request.t_next[:, origin:stop],
        request.dt[:, origin:stop],
    )
    return TransientPreparedRequest(
        context=request.context,
        state=tensors[0],
        static=tensors[1],
        boundary=tensors[2],
        scalars=tensors[3],
        t_n=tensors[4],
        t_next=tensors[5],
        dt=tensors[6],
        _versions=tuple(value._version for value in tensors),  # noqa: SLF001 -- PyTorch mutation counters and owned request versions protect input identity
        _owner_token=_PREPARED_REQUEST_TOKEN,
    )


class _ModelCallTimer:
    """Measure only dispatched model calls and synchronize CUDA once per request."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._cpu_calls: list[float] = []
        self._cuda_events: list[tuple[Any, Any]] = []

    def measure(self, invoke: Callable[[], object]) -> object:
        """Measure one model dispatch without input assembly or state reconstruction."""
        if self._device.type == "cuda":
            stream = torch.cuda.current_stream(self._device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            result = invoke()
            end.record(stream)
            self._cuda_events.append((start, end))
            return result
        started = perf_counter()
        result = invoke()
        self._cpu_calls.append(perf_counter() - started)
        return result

    def call_seconds(self) -> tuple[float, ...]:
        """Return model-only seconds of every measured call after one bounded completion wait."""
        if self._device.type == "cuda":
            if not self._cuda_events:
                return ()
            self._cuda_events[-1][1].synchronize()
            calls = tuple(float(start.elapsed_time(end)) / 1000.0 for start, end in self._cuda_events)
        else:
            calls = tuple(self._cpu_calls)
        if any(seconds < 0.0 for seconds in calls):
            msg = "Transient inference measured negative model time."
            raise RuntimeError(msg)
        return calls

    def finish(self) -> float:
        """Return aggregate model-only seconds after one bounded completion wait."""
        return sum(self.call_seconds())


def _timed_predict(
    context: TransientInferenceContext,
    step_input: torch.Tensor,
    *,
    hidden: object | None,
    require_finite_output: bool = True,
) -> tuple[torch.Tensor, object | None, float]:
    """Time only the model invocation supplied to the transient rollout owner."""
    timer = _ModelCallTimer(context.device)
    with torch.no_grad():
        if require_finite_output:
            prediction, next_hidden = rollout.predict_step(
                context.model,
                step_input,
                model_kind=context.model_kind,
                hidden=hidden,
                model_call=timer.measure,
            )
        else:
            prediction, next_hidden = rollout.predict_step(
                context.model,
                step_input,
                model_kind=context.model_kind,
                hidden=hidden,
                model_call=timer.measure,
                require_finite_output=False,
            )
    return prediction, next_hidden, timer.finish()


def _timing(*, seconds: float, context: TransientInferenceContext, mode: Literal["one_step", "autonomous_rollout"], calls: int) -> TransientTiming:
    """Build factual timing metadata without derived performance claims."""
    return TransientTiming(
        model_seconds=seconds,
        device=str(context.device),
        precision=context.precision,
        mode=mode,
        model_calls=calls,
    )


def predict_prepared_transient_step(
    context: TransientInferenceContext,
    request: TransientPreparedRequest,
) -> TransientStepResult:
    """Predict one step from already-admitted runtime tensors."""
    _require_prepared_request(context, request)
    step_input = context.tensorizer.assemble_step(
        request.state,
        request.static,
        request.boundary[:, 0],
        request.scalars,
        request.t_n[:, 0],
    )
    scaled_delta, _, seconds = _timed_predict(context, step_input, hidden=None)
    scaled_delta = scaled_delta.to(
        device=context.scaling.device,
        dtype=torch.float32,
    )
    next_state = context.tensorizer.reconstruct_next_state(
        request.state,
        scaled_delta,
    )
    timing = _timing(
        seconds=seconds,
        context=context,
        mode="one_step",
        calls=1,
    )
    return TransientStepResult(
        next_state=next_state,
        scaled_delta=scaled_delta,
        timing=timing,
    )


def predict_transient_step(
    context: TransientInferenceContext,
    *,
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    scalars: torch.Tensor,
    t_n: torch.Tensor,
    t_next: torch.Tensor,
    dt: torch.Tensor,
    airflow_source: str | None = None,
    external_airflow: torch.Tensor | None = None,
) -> TransientStepResult:
    """Strictly prepare a physical request and predict its first next state."""
    request = prepare_transient_request(
        context,
        state=state,
        static=static,
        boundary=boundary,
        scalars=scalars,
        t_n=t_n,
        t_next=t_next,
        dt=dt,
        airflow_source=airflow_source,
        external_airflow=external_airflow,
    )
    return predict_prepared_transient_step(context, request)


def rollout_prepared_transient_autonomous(
    context: TransientInferenceContext,
    request: TransientPreparedRequest,
) -> TransientRolloutResult:
    """Autonomously reconstruct one already-admitted runtime request window."""
    _require_prepared_request(context, request)
    current = request.state
    batch_size, channels, height, width = current.shape
    output_shape = (batch_size, request.length, channels, height, width)
    states = current.new_empty(output_shape)
    deltas = current.new_empty(output_shape)
    hidden: object | None = None
    timer = _ModelCallTimer(context.device)
    with torch.inference_mode():
        for index in range(request.length):
            step_input = context.tensorizer.assemble_step(
                current,
                request.static,
                request.boundary[:, index],
                request.scalars,
                request.t_n[:, index],
            )
            scaled_delta, hidden = rollout.predict_step(
                context.model,
                step_input,
                model_kind=context.model_kind,
                hidden=hidden,
                model_call=timer.measure,
            )
            scaled_delta = scaled_delta.to(
                device=context.scaling.device,
                dtype=torch.float32,
            )
            current = context.tensorizer.reconstruct_next_state(
                current,
                scaled_delta,
            )
            deltas[:, index].copy_(scaled_delta)
            states[:, index].copy_(current)
    seconds = timer.finish()
    return TransientRolloutResult(
        states=states,
        scaled_deltas=deltas,
        timing=_timing(
            seconds=seconds,
            context=context,
            mode="autonomous_rollout",
            calls=request.length,
        ),
    )


def teacher_forced_prepared_transient_diagnostic(
    context: TransientInferenceContext,
    request: TransientPreparedRequest,
    *,
    reference_states: torch.Tensor,
    capture: Iterable[int] = (),
) -> TransientTeacherForcedResult:
    """
    Predict every step from its reference state while carrying the RNO state.

    Parameters
    ----------
    context : TransientInferenceContext
        Active inference context.
    request : TransientPreparedRequest
        Admitted request starting at its trajectory's first transition.
    reference_states : torch.Tensor
        Reference current states ``[B,L,4,Y,X]`` for every request step; the
        first equals ``request.state``.
    capture : Iterable[int], optional
        Step counts after which the hidden state is returned.

    Returns
    -------
    TransientTeacherForcedResult
        Raw outputs, reconstructed one-step states, per-call model seconds, and
        the captured states. Stateless models return no hidden state.

    Raises
    ------
    ValueError
        If reference states or capture positions contradict the request.

    """
    _require_prepared_request(context, request)
    expected_shape = (request.state.shape[0], request.length, *request.state.shape[1:])
    if (
        not isinstance(reference_states, torch.Tensor)
        or reference_states.shape != expected_shape
        or reference_states.device != context.device
        or reference_states.dtype != torch.float32
        or not torch.equal(reference_states[:, 0], request.state)
    ):
        msg = "Teacher-forced reference states must align with the prepared request."
        raise ValueError(msg)
    positions = frozenset(capture)
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= request.length for value in positions):
        msg = "Captured hidden-state positions must be step counts within the request."
        raise ValueError(msg)
    if context.model_kind != "rno" and positions:
        msg = "Stateless transient models expose no hidden state."
        raise ValueError(msg)
    next_states = reference_states.new_empty(expected_shape)
    scaled_deltas = reference_states.new_empty(expected_shape)
    decoded_deltas = reference_states.new_empty(expected_shape)
    hidden: object | None = None
    captured: dict[int, object | None] = {0: None} if 0 in positions else {}
    timer = _ModelCallTimer(context.device)
    with torch.inference_mode():
        for index in range(request.length):
            current = reference_states[:, index]
            step_input = context.tensorizer.assemble_step(
                current,
                request.static,
                request.boundary[:, index],
                request.scalars,
                request.t_n[:, index],
            )
            scaled_delta, hidden = rollout.predict_step(
                context.model,
                step_input,
                model_kind=context.model_kind,
                hidden=hidden,
                model_call=timer.measure,
                require_finite_output=False,
            )
            scaled_delta = scaled_delta.to(device=context.scaling.device, dtype=torch.float32)
            next_state, decoded_delta = context.tensorizer.reconstruct_next_state_diagnostic(current, scaled_delta)
            scaled_deltas[:, index].copy_(scaled_delta)
            decoded_deltas[:, index].copy_(decoded_delta)
            next_states[:, index].copy_(next_state)
            if index + 1 in positions:
                captured[index + 1] = hidden
    step_seconds = timer.call_seconds()
    return TransientTeacherForcedResult(
        next_states=next_states,
        scaled_deltas=scaled_deltas,
        decoded_deltas=decoded_deltas,
        step_seconds=step_seconds,
        timing=_timing(seconds=sum(step_seconds), context=context, mode="one_step", calls=request.length),
        captured_hidden=captured,
    )


def rollout_prepared_transient_autonomous_diagnostic(
    context: TransientInferenceContext,
    request: TransientPreparedRequest,
    *,
    initial_hidden: object | None = None,
) -> TransientRolloutResult:
    """
    Preserve one computed prediction prefix and stop before invalid feedback.

    ``initial_hidden`` continues an RNO state built over the reference prefix
    of the same trajectory, for example by
    :func:`teacher_forced_prepared_transient_diagnostic`.
    """
    _require_prepared_request(context, request)
    current = request.state
    batch_size, channels, height, width = current.shape
    output_shape = (batch_size, request.length, channels, height, width)
    states = current.new_full(output_shape, float("nan"))
    scaled_deltas = current.new_full(output_shape, float("nan"))
    decoded_deltas = current.new_full(output_shape, float("nan"))
    available = torch.zeros(
        (batch_size, request.length),
        dtype=torch.bool,
        device=current.device,
    )
    if context.model_kind != "rno" and initial_hidden is not None:
        msg = "Stateless transient models cannot start from a hidden state."
        raise ValueError(msg)
    hidden = initial_hidden
    timer = _ModelCallTimer(context.device)
    model_calls = 0
    with torch.inference_mode():
        for index in range(request.length):
            step_input = context.tensorizer.assemble_step(
                current,
                request.static,
                request.boundary[:, index],
                request.scalars,
                request.t_n[:, index],
            )
            scaled_delta, hidden = rollout.predict_step(
                context.model,
                step_input,
                model_kind=context.model_kind,
                hidden=hidden,
                model_call=timer.measure,
                require_finite_output=False,
            )
            scaled_delta = scaled_delta.to(
                device=context.scaling.device,
                dtype=torch.float32,
            )
            next_state, decoded_delta = context.tensorizer.reconstruct_next_state_diagnostic(
                current,
                scaled_delta,
            )
            scaled_deltas[:, index].copy_(scaled_delta)
            decoded_deltas[:, index].copy_(decoded_delta)
            states[:, index].copy_(next_state)
            available[:, index] = True
            model_calls += 1
            if not (
                bool(torch.isfinite(scaled_delta).all().item())
                and bool(torch.isfinite(decoded_delta).all().item())
                and bool(torch.isfinite(next_state).all().item())
            ):
                break
            current = next_state
    seconds = timer.finish()
    return TransientRolloutResult(
        states=states,
        scaled_deltas=scaled_deltas,
        timing=_timing(
            seconds=seconds,
            context=context,
            mode="autonomous_rollout",
            calls=model_calls,
        ),
        decoded_deltas=decoded_deltas,
        prediction_available=available,
    )


def rollout_transient_autonomous(
    context: TransientInferenceContext,
    *,
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    scalars: torch.Tensor,
    t_n: torch.Tensor,
    t_next: torch.Tensor,
    dt: torch.Tensor,
    airflow_source: str | None = None,
    external_airflow: torch.Tensor | None = None,
) -> TransientRolloutResult:
    """Strictly prepare a physical request and autonomously reconstruct it."""
    request = prepare_transient_request(
        context,
        state=state,
        static=static,
        boundary=boundary,
        scalars=scalars,
        t_n=t_n,
        t_next=t_next,
        dt=dt,
        airflow_source=airflow_source,
        external_airflow=external_airflow,
    )
    return rollout_prepared_transient_autonomous(context, request)
