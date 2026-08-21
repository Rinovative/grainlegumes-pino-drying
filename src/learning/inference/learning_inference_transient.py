"""
learning_inference_transient.py

Serve explicit physical transient-drying inference from completed run evidence.

Responsibilities:
  - Reconstruct a transient model, tensorizer, and fitted scaling artifact from one completed run
  - Validate explicit physical one-step and rollout request evidence
  - Apply the admitted airflow-source policy without mutating caller tensors
  - Execute stateless or recurrent model calls and return physical reconstructed states with factual timing

Design principles:
  - Completed-run admission is the sole persisted-artifact read boundary
  - Tensorizer and scaling artifacts remain authoritative for channel assembly and reconstruction
  - Public recurrent requests start with no hidden state and carry it only within that request
  - Model timing excludes request validation, transfer, tensorization, scaling, and reconstruction

This module does NOT:
  - Load Dataset packages, resolve saved splits, or refit scaling state
  - Alter the steady inference tuple API
  - Claim speedup, benchmark performance, or execute training
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

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

# ruff: noqa: EM101, EM102, PLR2004, TRY003


@dataclass(frozen=True, slots=True)
class TransientInferenceContext:
    """Hold one completed-run transient model and its admitted preprocessing evidence."""

    model: nn.Module
    tensorizer: TransientTensorizer
    scaling: TransientScalingArtifact
    device: torch.device
    model_kind: Literal["fno", "uno", "rno"]
    precision: Literal["float32"]


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


@dataclass(frozen=True, slots=True)
class TransientRolloutResult:
    """Return autonomous physical states, scaled increments, and model timing."""

    states: torch.Tensor
    scaled_deltas: torch.Tensor
    timing: TransientTiming


def _require_precision(value: str) -> Literal["float32"]:
    """Require the sole supported explicit transient inference precision."""
    if value != _PRECISION:
        raise ValueError("Transient inference supports only precision 'float32'.")
    return cast('Literal["float32"]', _PRECISION)


def _require_model_kind(value: object) -> Literal["fno", "uno", "rno"]:
    """Require one persisted transient model kind."""
    if value not in _MODEL_KINDS:
        raise ValueError("Transient inference model.kind must be 'fno', 'uno', or 'rno'.")
    return cast('Literal["fno", "uno", "rno"]', value)


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve one concrete inference device without implicit fallback."""
    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, str):
        resolved = learning.device.resolve_device(device, path="transient inference device").device
    else:
        raise TypeError("Transient inference device must be one string policy or torch.device.")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Transient inference requires one concrete CPU or CUDA device.")
    if resolved.type == "cuda" and resolved.index is None:
        raise ValueError("Transient inference requires an indexed CUDA device.")
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
            raise RuntimeError(f"Transient inference model tensor {name!r} is not on the resolved device.")
        expected = torch.complex64 if value.is_complex() else torch.float32 if value.is_floating_point() else value.dtype
        if value.dtype != expected:
            raise RuntimeError(f"Transient inference model tensor {name!r} violates float32-equivalent precision.")
    return model


def load_transient_inference_context(
    *,
    run_dir: str | Path,
    device: str | torch.device = "cpu",
    precision: str = _PRECISION,
) -> TransientInferenceContext:
    """
    Reconstruct one transient inference context from an admitted completed run.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Completed transient run directory admitted by ``experiments.run``.
    device : str or torch.device, optional
        Explicit device policy or concrete device. Default is ``"cpu"``.
    precision : str, optional
        Explicit precision. Only ``"float32"`` is supported.

    Returns
    -------
    TransientInferenceContext
        Loaded model plus exact fitted tensorizer/scaling state.

    Raises
    ------
    RuntimeError
        If the completed bundle is not a transient-drying run or checkpoint loading fails.
    ValueError
        If precision, model kind, spatial modes, or device is unsupported.

    Notes
    -----
    The loader consumes only state returned by completed-run admission and does
    not reopen normalizer or checkpoint artifacts.

    """
    resolved_precision = _require_precision(precision)
    completed = experiments.run.validate_completed_run(run_dir)
    config = completed.get("config")
    if not isinstance(config, Mapping) or config.get("task") != "transient_drying":
        raise RuntimeError("Transient inference requires one completed transient_drying run.")
    normalizer_state = completed.get("normalizer_state")
    if not isinstance(normalizer_state, Mapping):
        raise TypeError("Completed transient run must provide one scaling-artifact mapping.")
    scaling = TransientScalingArtifact.from_state_dict(normalizer_state)
    resolved_device = _resolve_device(device)
    scaling = scaling.to(resolved_device)
    spec = TransientTensorizerSpec.from_mapping(
        {
            "input_profile": config.get("input_profile"),
            "temporal_conditioning": config.get("temporal", {}).get("temporal_conditioning") if isinstance(config.get("temporal"), Mapping) else None,
        }
    )
    if spec != scaling.tensorizer:
        raise RuntimeError("Completed transient config tensorizer disagrees with admitted scaling artifact.")
    model_factory.validate_transient_model_spatial_shape(config, scaling.spatial_shape)
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise TypeError("Completed transient run must provide model configuration.")
    kind = _require_model_kind(model_config.get("kind"))
    model = model_factory.build_model(dict(config), device=resolved_device)
    best_checkpoint = completed.get("best_checkpoint")
    if (
        not isinstance(best_checkpoint, Mapping)
        or best_checkpoint.get("schema_version") != learning.training.checkpoint.CHECKPOINT_SCHEMA_VERSION
        or not isinstance(best_checkpoint.get("model_state_dict"), Mapping)
    ):
        raise TypeError("Completed transient run must provide an admitted best model state dict.")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model = _place_model_float32(model, device=resolved_device)
    model.eval()
    return TransientInferenceContext(
        model=model,
        tensorizer=TransientTensorizer(spec, scaling),
        scaling=scaling,
        device=resolved_device,
        model_kind=kind,
        precision=resolved_precision,
    )


def _require_request_tensor(value: object, *, label: str) -> torch.Tensor:
    """Require one finite float32 tensor without moving or copying caller data."""
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise TypeError(f"Transient inference {label} must be one torch.float32 tensor.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"Transient inference {label} must contain only finite values.")
    return value


def _resolve_airflow_source(value: str | None) -> str:
    """Resolve the registered default airflow policy and reject task-policy drift."""
    task = domain.tasks.registry.get_task("transient_drying")
    declared = task.training_airflow_source
    if declared != _COMSOL_REFERENCE_AIRFLOW_SOURCE or declared not in _AIRFLOW_SOURCES:
        raise RuntimeError("Transient task airflow policy must declare exactly 'comsol_reference'.")
    source = declared if value is None else value
    if source not in _AIRFLOW_SOURCES:
        raise ValueError("airflow_source must be exactly 'comsol_reference' or 'external'.")
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
    if state.ndim != 4 or state.shape[1] != 4:
        raise ValueError("Transient inference state must have shape [B,4,Y,X].")
    batch_size, _, height, width = state.shape
    if min(batch_size, height, width) < 1:
        raise ValueError("Transient inference state must have non-empty B, Y, and X axes.")
    if (height, width) != spatial_shape:
        raise ValueError("Transient inference state spatial shape must match the admitted scaling artifact.")
    if static.shape != (batch_size, 7, height, width):
        raise ValueError("Transient inference static must have shape [B,7,Y,X].")
    if boundary.ndim != 3 or boundary.shape[0] != batch_size or boundary.shape[2] != 9 or boundary.shape[1] < 1:
        raise ValueError("Transient inference boundary must have non-empty shape [B,L,9].")
    length = int(boundary.shape[1])
    if scalars.shape != (batch_size, 8):
        raise ValueError("Transient inference scalars must have shape [B,8].")
    if any(value.shape != (batch_size, length) for value in (t_n, t_next, dt)):
        raise ValueError("Transient inference t_n, t_next, and dt must each have shape [B,L].")
    source_device = state.device
    if any(value.device != source_device for value in tensors.values()):
        raise ValueError("Transient inference request tensors must share one source device.")
    if airflow_source == "comsol_reference":
        if external_airflow is not None:
            raise ValueError("comsol_reference airflow does not accept external_airflow.")
    else:
        external = _require_request_tensor(external_airflow, label="external_airflow")
        if external.shape != (batch_size, 3, height, width) or external.device != source_device:
            raise ValueError("external_airflow must have shape [B,3,Y,X] on the request source device.")
    ones = torch.ones_like(dt)
    if not torch.allclose(dt, ones, rtol=0.0, atol=_TIME_ATOL) or not torch.allclose(t_next - t_n, dt, rtol=0.0, atol=_TIME_ATOL):
        raise ValueError("Transient inference requires exact regular fixed 1h time evidence.")
    if bool((t_n < -_TIME_ATOL).any()) or bool((t_next > horizon + _TIME_ATOL).any()):
        raise ValueError("Transient inference time evidence must remain within the admitted scaling horizon.")
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
    """Copy validated request data to the context device and apply airflow policy."""
    owned = tuple(value.to(device=context.device, dtype=torch.float32).clone() for value in (state, static, boundary, scalars, t_n, t_next, dt))
    runtime_state, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, runtime_t_next, runtime_dt = owned
    if airflow_source == "external":
        if external_airflow is None:
            raise RuntimeError("Validated external airflow unexpectedly missing.")
        runtime_airflow = external_airflow.to(device=context.device, dtype=torch.float32).clone()
        runtime_static[:, 2:5] = runtime_airflow
    return runtime_state, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, runtime_t_next, runtime_dt


def _timed_predict(
    context: TransientInferenceContext,
    step_input: torch.Tensor,
    *,
    hidden: object | None,
) -> tuple[torch.Tensor, object | None, float]:
    """Time only the model invocation supplied to the transient rollout owner."""
    seconds = 0.0

    def measure(invoke: Callable[[], object]) -> object:
        """Measure exactly one dispatched model call and no input/output validation."""
        nonlocal seconds
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(context.device))
            result = invoke()
            end.record(torch.cuda.current_stream(context.device))
            torch.cuda.synchronize(context.device)
            seconds = float(start.elapsed_time(end)) / 1000.0
            return result
        started = perf_counter()
        result = invoke()
        seconds = perf_counter() - started
        return result

    with torch.no_grad():
        prediction, next_hidden = rollout.predict_step(
            context.model,
            step_input,
            model_kind=context.model_kind,
            hidden=hidden,
            model_call=measure,
        )
    if seconds < 0.0:
        raise RuntimeError("Transient inference measured negative model time.")
    return prediction, next_hidden, seconds


def _timing(*, seconds: float, context: TransientInferenceContext, mode: Literal["one_step", "autonomous_rollout"], calls: int) -> TransientTiming:
    """Build factual timing metadata without derived performance claims."""
    return TransientTiming(
        model_seconds=seconds,
        device=str(context.device),
        precision=context.precision,
        mode=mode,
        model_calls=calls,
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
    """Predict the first physical next state from one validated transient request window."""
    if not isinstance(context, TransientInferenceContext):
        raise TypeError("context must be one TransientInferenceContext.")
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
        spatial_shape=context.scaling.spatial_shape,
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
    runtime_state, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, _, _ = values
    step_input = context.tensorizer.assemble_step(runtime_state, runtime_static, runtime_boundary[:, 0], runtime_scalars, runtime_t_n[:, 0])
    scaled_delta, _, seconds = _timed_predict(context, step_input, hidden=None)
    scaled_delta = scaled_delta.to(device=context.scaling.device, dtype=torch.float32)
    next_state = context.tensorizer.reconstruct_next_state(runtime_state, scaled_delta)
    timing = _timing(seconds=seconds, context=context, mode="one_step", calls=1)
    return TransientStepResult(next_state=next_state, scaled_delta=scaled_delta, timing=timing)


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
    """Autonomously reconstruct every next state in one validated request window."""
    if not isinstance(context, TransientInferenceContext):
        raise TypeError("context must be one TransientInferenceContext.")
    resolved_airflow_source = _resolve_airflow_source(airflow_source)
    _, length, _, _ = _validate_request(
        state=state,
        static=static,
        boundary=boundary,
        scalars=scalars,
        t_n=t_n,
        t_next=t_next,
        dt=dt,
        external_airflow=external_airflow,
        airflow_source=resolved_airflow_source,
        spatial_shape=context.scaling.spatial_shape,
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
    current, runtime_static, runtime_boundary, runtime_scalars, runtime_t_n, _, _ = values
    hidden: object | None = None
    deltas: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    seconds = 0.0
    for index in range(length):
        step_input = context.tensorizer.assemble_step(current, runtime_static, runtime_boundary[:, index], runtime_scalars, runtime_t_n[:, index])
        scaled_delta, hidden, elapsed = _timed_predict(context, step_input, hidden=hidden)
        scaled_delta = scaled_delta.to(device=context.scaling.device, dtype=torch.float32)
        current = context.tensorizer.reconstruct_next_state(current, scaled_delta)
        deltas.append(scaled_delta)
        states.append(current)
        seconds += elapsed
    return TransientRolloutResult(
        states=torch.stack(states, dim=1),
        scaled_deltas=torch.stack(deltas, dim=1),
        timing=_timing(seconds=seconds, context=context, mode="autonomous_rollout", calls=length),
    )
