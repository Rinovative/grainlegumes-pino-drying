# ruff: noqa: S101, PLR2004, SLF001
"""Protect completed-run transient inference reconstruction and explicit physical requests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pytest
import torch
from torch import nn

from src import datasets, domain, experiments
from src.learning.inference import learning_inference_transient as transient
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient import learning_transient_rollout as rollout
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec
from src.learning.transient.learning_transient_scaling import SCALE_FLOOR, TransientScalingArtifact
from src.learning.transient.learning_transient_tensorizer import TransientTensorizer


class _IncrementModel(nn.Module):
    def __init__(self, *, use_airflow: bool = False, output_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.use_airflow = use_airflow
        self.output_dtype = output_dtype
        self.hidden_inputs: list[object | None] = []
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.spectral_weight = nn.Parameter(torch.tensor(1.0 + 2.0j, dtype=torch.complex128))

    def forward(self, value: torch.Tensor, **kwargs: object) -> torch.Tensor | tuple[torch.Tensor, int]:
        if value.ndim == 5:
            hidden = kwargs["init_hidden_states"]
            self.hidden_inputs.append(hidden)
            if hidden is not None and not isinstance(hidden, int):
                message = "Synthetic hidden state must be an integer or null."
                raise TypeError(message)
            current = value[:, 0, :4]
            next_hidden = 0 if hidden is None else hidden + 1
            return (current + 1.0).to(self.output_dtype), next_hidden
        if self.use_airflow:
            return value[:, 6:7].expand(-1, 4, -1, -1).to(self.output_dtype)
        return (value[:, :4] + 1.0).to(self.output_dtype)


class _SecondStepNonfiniteModel(_IncrementModel):
    """Emit distinct IEEE invalid values on the second model dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
        output = super().forward(value, **kwargs)
        if not isinstance(output, torch.Tensor):
            message = "Synthetic FNO model unexpectedly returned hidden state."
            raise TypeError(message)
        self.calls += 1
        if self.calls == 2:
            output = output.clone()
            output[:, 0, 0, 0] = float("nan")
            output[:, 1, 0, 0] = float("inf")
            output[:, 2, 0, 0] = float("-inf")
        return output


def _scaling() -> TransientScalingArtifact:
    spec = TransientTensorizerSpec(
        input_profile="canonical_physics_complete_v1",
        temporal_conditioning=TemporalConditioningSpec("none"),
    )
    contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
    names = tuple(
        tuple(field.name for field in group)
        for group in (
            contract.dynamic_state,
            contract.static_spatial_conditioning,
            contract.step_boundary_conditioning,
            contract.scalar_conditioning,
        )
    )
    return TransientScalingArtifact(
        task_contract_digest=domain.tasks.registry.get_task("transient_drying").contract_digest,
        data_contract_digest=domain.tasks.registry.get_task("transient_drying").data_contract_digest,
        tensorizer=spec,
        dataset_identity={"synthetic": "inference"},
        train_membership_digest="a" * 64,
        scale_mode="state_std",
        numerical_floor=SCALE_FLOOR,
        unique_train_state_count=1,
        unique_transition_count=1,
        transition_count=1,
        spatial_shape=(2, 3),
        state_names=names[0],
        static_names=names[1],
        boundary_names=names[2],
        scalar_names=names[3],
        state_mean=torch.zeros(4),
        state_std=torch.ones(4),
        delta_rms=torch.ones(4),
        increment_scale=torch.ones(4),
        static_mean=torch.zeros(7),
        static_std=torch.ones(7),
        scalar_mean=torch.zeros(8),
        scalar_std=torch.ones(8),
        omega_boundary_mean=torch.tensor(0.0),
        omega_boundary_std=torch.tensor(1.0),
        horizon=8.0,
    )


def _context(model: nn.Module, *, model_kind: str = "fno") -> transient.TransientInferenceContext:
    scaling = _scaling()
    return transient.TransientInferenceContext(
        model=model.eval(),
        tensorizer=TransientTensorizer(scaling.tensorizer, scaling),
        scaling=scaling,
        device=torch.device("cpu"),
        model_kind=model_kind,  # type: ignore[arg-type]
        precision="float32",
        training_spatial_shape=scaling.spatial_shape,
        evaluation_spatial_shape=scaling.spatial_shape,
        architecture_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
        scaling_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
    )


class _Request(TypedDict):
    state: torch.Tensor
    static: torch.Tensor
    boundary: torch.Tensor
    scalars: torch.Tensor
    t_n: torch.Tensor
    t_next: torch.Tensor
    dt: torch.Tensor


def _request(length: int = 3) -> _Request:
    return {
        "state": torch.zeros(1, 4, 2, 3),
        "static": torch.stack([torch.arange(7, dtype=torch.float32).view(7, 1, 1).expand(7, 2, 3)]),
        "boundary": torch.zeros(1, length, 9),
        "scalars": torch.zeros(1, 8),
        "t_n": torch.arange(length, dtype=torch.float32).view(1, length),
        "t_next": torch.arange(1, length + 1, dtype=torch.float32).view(1, length),
        "dt": torch.ones(1, length),
    }


def test_completed_loader_uses_admitted_scaling_and_best_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load only admitted transient bundle state and validate its spatial model contract first."""
    scaling = _scaling()
    model = _IncrementModel()
    config: dict[str, Any] = {
        "task": "transient_drying",
        "input_profile": "canonical_physics_complete_v1",
        "temporal": {"temporal_conditioning": {"kind": "none"}},
        "model": {"kind": "fno", "params": {}},
    }
    completed = {
        "is_completed": True,
        "config": config,
        "normalizer_state": scaling.state_dict(),
        "best_checkpoint": {"schema_version": 1, "model_state_dict": model.state_dict()},
    }
    observed: list[tuple[dict[str, Any], tuple[int, int]]] = []
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _run_dir: completed)
    monkeypatch.setattr(transient.model_factory, "validate_transient_model_spatial_shape", lambda cfg, shape: observed.append((cfg, shape)))
    monkeypatch.setattr(transient.model_factory, "build_model", lambda _cfg, *, device: model.to(device))

    context = transient.load_transient_inference_context(run_dir="synthetic", device="cpu")

    assert context.model is model
    assert context.scaling.spatial_shape == (2, 3)
    assert observed == [(config, (2, 3))]
    assert context.model.training is False
    assert context.model.weight.dtype == torch.float32
    assert context.model.spectral_weight.dtype == torch.complex64
    assert context.model.spectral_weight.imag.item() == pytest.approx(2.0)
    monkeypatch.setattr(
        transient.model_factory,
        "validate_transient_model_spatial_shape",
        lambda _config, _shape: (_ for _ in ()).throw(ValueError("invalid spatial modes")),
    )
    with pytest.raises(ValueError, match="spatial modes"):
        transient.load_transient_inference_context(run_dir="synthetic")
    with pytest.raises(ValueError, match="float32"):
        transient.load_transient_inference_context(run_dir="synthetic", precision="float16")
    completed["config"] = {**config, "task": "steady_flow"}
    with pytest.raises(RuntimeError, match="transient_drying"):
        transient.load_transient_inference_context(run_dir="synthetic")


def test_stride_two_trained_rno_admits_original_grid_without_refitting_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regress the real artifact failure through exact cross-grid model and scaler proof."""
    base_scaling = _scaling()
    scaling = replace(
        base_scaling,
        dataset_identity={
            "synthetic": "stride-two-rno",
            "spatial_stride": 2,
            "canonical_spatial_shape": [251, 401],
            "effective_spatial_shape": [126, 201],
        },
        spatial_shape=(126, 201),
    )
    model = _IncrementModel()
    observed_model_inputs: list[tuple[int, ...]] = []

    def capture_model_input(_module: nn.Module, args: tuple[object, ...]) -> None:
        value = args[0]
        assert isinstance(value, torch.Tensor)
        observed_model_inputs.append(tuple(value.shape))

    model.register_forward_pre_hook(capture_model_input)
    config: dict[str, Any] = {
        "task": "transient_drying",
        "input_profile": "canonical_physics_complete_v1",
        "temporal": {"temporal_conditioning": {"kind": "none"}},
        "model": {
            "kind": "rno",
            "params": {"in_channels": 28, "n_modes": [24, 24]},
        },
    }
    completed = {
        "is_completed": True,
        "config": config,
        "normalizer_state": scaling.state_dict(),
        "best_checkpoint": {"schema_version": 1, "model_state_dict": model.state_dict()},
        "checkpoint_identity": {"epoch": 7},
        "selected_checkpoint_sha256": "b" * 64,
        "selected_checkpoint_epoch": 7,
    }
    monkeypatch.setattr(transient.model_factory, "build_model", lambda _cfg, *, device: model.to(device))

    context = transient.build_transient_inference_context(
        completed=completed,
        evaluation_spatial_shape=(251, 401),
    )

    assert context.training_spatial_shape == (126, 201)
    assert context.evaluation_spatial_shape == (251, 401)
    assert context.architecture_spatial_compatibility["decision"] == "SUPPORTED_WITH_CONTRACT"
    assert context.architecture_spatial_compatibility["forward_output_shape_proof"]["output_shape"] == [1, 4, 251, 401]
    assert context.scaling_spatial_compatibility["decision"] == "SUPPORTED_WITH_CONTRACT"
    assert context.scaling.spatial_shape == (126, 201)
    assert context.scaling.digest == scaling.digest

    request: _Request = {
        "state": torch.zeros(1, 4, 251, 401),
        "static": torch.zeros(1, 7, 251, 401),
        "boundary": torch.zeros(1, 1, 9),
        "scalars": torch.zeros(1, 8),
        "t_n": torch.zeros(1, 1),
        "t_next": torch.ones(1, 1),
        "dt": torch.ones(1, 1),
    }
    result = transient.predict_transient_step(context, **request)
    assert observed_model_inputs == [(1, 1, 28, 251, 401), (1, 1, 28, 251, 401)]
    assert result.next_state.shape == (1, 4, 251, 401)


def test_cross_resolution_incompatibility_fails_before_model_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject retained modes that do not fit the requested grid before case inference."""
    scaling = replace(
        _scaling(),
        dataset_identity={"spatial_stride": 2},
        spatial_shape=(126, 201),
    )
    model = _IncrementModel()
    completed = {
        "is_completed": True,
        "config": {
            "task": "transient_drying",
            "input_profile": "canonical_physics_complete_v1",
            "temporal": {"temporal_conditioning": {"kind": "none"}},
            "model": {"kind": "rno", "params": {"in_channels": 20, "n_modes": [64, 64]}},
        },
        "normalizer_state": scaling.state_dict(),
        "best_checkpoint": {"schema_version": 1, "model_state_dict": model.state_dict()},
        "checkpoint_identity": {"epoch": 7},
        "selected_checkpoint_sha256": "b" * 64,
        "selected_checkpoint_epoch": 7,
    }
    monkeypatch.setattr(
        transient.model_factory,
        "build_model",
        lambda *_args, **_kwargs: pytest.fail("incompatible preflight reached model setup"),
    )

    with pytest.raises(ValueError, match=r"checkpoint=.*Training shape=.*Evaluation shape=.*modes"):
        transient.build_transient_inference_context(
            completed=completed,
            evaluation_spatial_shape=(32, 48),
        )


def test_request_validation_rejects_shape_dtype_placement_and_time_contracts() -> None:
    """Reject malformed physical requests before tensorization or model execution."""
    context = _context(_IncrementModel())
    request = _request()
    bad_dtype: _Request = {**request, "state": request["state"].to(torch.float64)}
    with pytest.raises(TypeError, match="float32"):
        transient.predict_transient_step(context, **bad_dtype)
    bad_shape: _Request = {**request, "boundary": torch.zeros(1, 3, 8)}
    with pytest.raises(ValueError, match="boundary"):
        transient.predict_transient_step(context, **bad_shape)
    bad_time: _Request = {**request, "dt": torch.full((1, 3), 2.0)}
    with pytest.raises(ValueError, match="1h"):
        transient.predict_transient_step(context, **bad_time)
    nonfinite: _Request = {**request, "scalars": torch.full((1, 8), float("nan"))}
    with pytest.raises(ValueError, match="finite"):
        transient.predict_transient_step(context, **nonfinite)
    negative_time: _Request = {**request, "t_n": torch.full((1, 3), -1.0), "t_next": torch.zeros(1, 3)}
    with pytest.raises(ValueError, match="horizon"):
        transient.predict_transient_step(context, **negative_time)
    past_horizon: _Request = {
        **request,
        "t_n": torch.arange(8, 11, dtype=torch.float32).view(1, 3),
        "t_next": torch.arange(9, 12, dtype=torch.float32).view(1, 3),
    }
    with pytest.raises(ValueError, match="horizon"):
        transient.predict_transient_step(context, **past_horizon)


def test_airflow_policy_preserves_comsol_and_replaces_only_airflow_without_mutation() -> None:
    """Use static u/v/p only when external airflow is explicitly selected."""
    context = _context(_IncrementModel(use_airflow=True))
    request = _request(1)
    static_before = request["static"].clone()
    comsol = transient.predict_transient_step(context, **request)
    external = torch.full((1, 3, 2, 3), 9.0)
    external_result = transient.predict_transient_step(
        context,
        **request,
        airflow_source="external",
        external_airflow=external,
    )
    assert torch.equal(request["static"], static_before)
    assert torch.allclose(comsol.scaled_delta, torch.full((1, 4, 2, 3), 2.0))
    assert torch.allclose(external_result.scaled_delta, torch.full((1, 4, 2, 3), 9.0))
    with pytest.raises(ValueError, match="does not accept"):
        transient.predict_transient_step(context, **request, external_airflow=external)


def test_prepared_request_owns_runtime_tensors_and_windows_without_recopies() -> None:
    """Own admitted inputs once and reuse zero-copy conditioning windows."""
    context = _context(_IncrementModel())
    request = _request(3)
    prepared = transient.prepare_transient_request(context, **request)

    assert prepared.context is context
    assert prepared.state.device == context.device
    assert prepared.state.dtype == torch.float32
    assert prepared.state.data_ptr() != request["state"].data_ptr()
    request["state"].add_(9.0)
    assert torch.count_nonzero(prepared.state) == 0

    current = torch.full_like(prepared.state, 2.0)
    window = transient.window_transient_prepared_request(
        prepared,
        origin=1,
        length=2,
        state=current,
    )
    assert window.static.data_ptr() == prepared.static.data_ptr()
    assert window.boundary.untyped_storage().data_ptr() == prepared.boundary.untyped_storage().data_ptr()
    assert window.boundary.shape == (1, 2, 9)

    result = transient.rollout_prepared_transient_autonomous(context, window)

    assert result.states.device == context.device
    assert result.states.dtype == torch.float32
    assert torch.allclose(result.states[:, 1], torch.full((1, 4, 2, 3), 11.0))


def test_prepared_request_rejects_post_admission_mutation() -> None:
    """Reject mutable prepared evidence before scaler or model execution."""
    context = _context(_IncrementModel())
    prepared = transient.prepare_transient_request(context, **_request(2))
    prepared.static.add_(1.0)

    with pytest.raises(RuntimeError, match="mutated after strict admission"):
        transient.rollout_prepared_transient_autonomous(context, prepared)


def test_diagnostic_rollout_preserves_ieee_values_and_stops_before_invalid_feedback() -> None:
    """Keep raw invalid output, mark its computed prefix, and never feed it back."""
    strict_context = _context(_SecondStepNonfiniteModel())
    strict_request = transient.prepare_transient_request(
        strict_context,
        **_request(3),
    )
    with pytest.raises(FloatingPointError, match="finite"):
        transient.rollout_prepared_transient_autonomous(
            strict_context,
            strict_request,
        )

    model = _SecondStepNonfiniteModel()
    context = _context(model)
    prepared = transient.prepare_transient_request(context, **_request(3))
    result = transient.rollout_prepared_transient_autonomous_diagnostic(
        context,
        prepared,
    )

    assert model.calls == 2
    assert result.timing.model_calls == 2
    assert result.prediction_available is not None
    assert result.prediction_available.tolist() == [[True, True, False]]
    assert result.decoded_deltas is not None
    assert torch.isnan(result.scaled_deltas[0, 1, 0, 0, 0])
    assert torch.isposinf(result.scaled_deltas[0, 1, 1, 0, 0])
    assert torch.isneginf(result.scaled_deltas[0, 1, 2, 0, 0])
    assert torch.isnan(result.states[:, 2]).all()
    assert torch.isnan(result.scaled_deltas[:, 2]).all()
    assert torch.isnan(result.decoded_deltas[:, 2]).all()


@pytest.mark.parametrize("model_kind", ["fno", "uno"])
def test_one_step_and_autonomous_recurrence_cast_deltas_and_report_cpu_timing(model_kind: str) -> None:
    """Reconstruct FNO/U-NO recurrence from float32 deltas and factual CPU timing."""
    context = _context(_IncrementModel(output_dtype=torch.float64), model_kind=model_kind)
    request = _request(3)
    step = transient.predict_transient_step(context, **request)
    rollout = transient.rollout_transient_autonomous(context, **request)
    assert step.scaled_delta.dtype == torch.float32
    assert torch.allclose(step.next_state, torch.ones(1, 4, 2, 3))
    assert torch.allclose(rollout.states[:, 0], torch.ones(1, 4, 2, 3))
    assert torch.allclose(rollout.states[:, 2], torch.full((1, 4, 2, 3), 7.0))
    assert rollout.timing.device == "cpu"
    assert rollout.timing.model_calls == 3
    assert rollout.timing.model_seconds >= 0.0
    assert not hasattr(rollout.timing, "speedup")


def test_autonomous_rollout_preallocates_outputs_without_list_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain exact recurrence while avoiding a second full trajectory allocation."""

    class _Tensorizer:
        def assemble_step(
            self,
            current: torch.Tensor,
            _static: torch.Tensor,
            _boundary: torch.Tensor,
            _scalars: torch.Tensor,
            _time: torch.Tensor,
        ) -> torch.Tensor:
            return current

        def reconstruct_next_state(
            self,
            current: torch.Tensor,
            delta: torch.Tensor,
        ) -> torch.Tensor:
            return current + delta

    scaling = SimpleNamespace(
        spatial_shape=(2, 3),
        horizon=8.0,
        device=torch.device("cpu"),
    )
    context = transient.TransientInferenceContext(
        model=_IncrementModel().eval(),
        tensorizer=cast("Any", _Tensorizer()),
        scaling=cast("Any", scaling),
        device=torch.device("cpu"),
        model_kind="fno",
        precision="float32",
        training_spatial_shape=(2, 3),
        evaluation_spatial_shape=(2, 3),
        architecture_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
        scaling_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
    )
    request = _request(3)
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *_args, **_kwargs: pytest.fail(
            "autonomous inference used list-to-stack accumulation",
        ),
    )

    result = transient.rollout_transient_autonomous(context, **request)

    assert result.states.shape == (1, 3, 4, 2, 3)
    assert result.scaled_deltas.shape == result.states.shape
    assert torch.allclose(result.states[:, 2], torch.full((1, 4, 2, 3), 7.0))


def test_cuda_rollout_timing_preserves_model_call_boundaries_with_one_completion_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggregate per-dispatch CUDA events with one bounded completion wait."""

    class _Tensorizer:
        def assemble_step(
            self,
            current: torch.Tensor,
            _static: torch.Tensor,
            _boundary: torch.Tensor,
            _scalars: torch.Tensor,
            _time: torch.Tensor,
        ) -> torch.Tensor:
            return current

        def reconstruct_next_state(self, current: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
            return current + delta

    events: list[str] = []

    class _Event:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self, _stream: object) -> None:
            events.append("record")

        def synchronize(self) -> None:
            events.append("synchronize")

        def elapsed_time(self, _other: object) -> float:
            return 12.5

    request = _request(3)
    scaling = SimpleNamespace(spatial_shape=(2, 3), horizon=8.0, device=torch.device("cpu"))
    context = transient.TransientInferenceContext(
        model=_IncrementModel().eval(),
        tensorizer=cast("Any", _Tensorizer()),
        scaling=cast("Any", scaling),
        device=torch.device("cuda", 7),
        model_kind="fno",
        precision="float32",
        training_spatial_shape=(2, 3),
        evaluation_spatial_shape=(2, 3),
        architecture_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
        scaling_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
    )
    monkeypatch.setattr(
        transient,
        "_owned_runtime_inputs",
        lambda **_kwargs: (
            request["state"],
            request["static"],
            request["boundary"],
            request["scalars"],
            request["t_n"],
            request["t_next"],
            request["dt"],
        ),
    )
    monkeypatch.setattr(
        transient,
        "_require_prepared_request",
        lambda _context, _request: None,
    )
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: pytest.fail("rollout used global CUDA synchronization"))

    result = transient.rollout_transient_autonomous(context, **request)

    assert events == ["record", "record", "record", "record", "record", "record", "synchronize"]
    assert result.timing.model_seconds == pytest.approx(0.0375)
    assert result.timing.model_calls == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_runtime_prepares_cpu_request_and_retains_rollout_state() -> None:
    """Move one CPU request once and retain model and rollout tensors on the runtime device."""
    device = torch.device("cuda", torch.cuda.current_device())
    model = _IncrementModel().to(device).eval()
    scaling = _scaling().to(device)
    context = transient.TransientInferenceContext(
        model=model,
        tensorizer=TransientTensorizer(scaling.tensorizer, scaling),
        scaling=scaling,
        device=device,
        model_kind="fno",
        precision="float32",
        training_spatial_shape=(2, 3),
        evaluation_spatial_shape=(2, 3),
        architecture_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
        scaling_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
    )
    model_input_devices: list[torch.device] = []

    def capture_model_input(_module: nn.Module, args: tuple[object, ...]) -> None:
        value = args[0]
        assert isinstance(value, torch.Tensor)
        model_input_devices.append(value.device)

    handle = model.register_forward_pre_hook(capture_model_input)
    request = _request(3)

    try:
        result = transient.rollout_transient_autonomous(context, **request)
    finally:
        handle.remove()

    request_tensors = (
        request["state"],
        request["static"],
        request["boundary"],
        request["scalars"],
        request["t_n"],
        request["t_next"],
        request["dt"],
    )
    assert all(value.device.type == "cpu" for value in request_tensors)
    assert model_input_devices == [device, device, device]
    assert result.states.device == device
    assert result.scaled_deltas.device == device


def test_rno_carries_hidden_within_each_rollout_and_resets_between_public_requests() -> None:
    """Keep official RNO state request-local while FNO/UNO remain stateless."""
    model = _IncrementModel()
    context = _context(model, model_kind="rno")
    request = _request(3)
    first = transient.rollout_transient_autonomous(context, **request)
    second = transient.rollout_transient_autonomous(context, **request)
    assert first.states.shape == (1, 3, 4, 2, 3)
    assert second.states.shape == first.states.shape
    assert model.hidden_inputs == [None, 0, 1, None, 0, 1]


def test_predict_step_rejects_nonfloating_outputs_and_times_only_model_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model timing wrappers outside input/output validation while rejecting invalid outputs."""
    input_value = torch.zeros(1, 5, 2, 3)

    class _OutputModel(nn.Module):
        def __init__(self, output: torch.Tensor, events: list[str] | None = None) -> None:
            super().__init__()
            self.output = output
            self.events = events

        def forward(self, _value: torch.Tensor) -> torch.Tensor:
            if self.events is not None:
                self.events.append("model")
            return self.output

    for output in (torch.zeros(1, 4, 2, 3, dtype=torch.int64), torch.zeros(1, 4, 2, 3, dtype=torch.complex64)):
        with pytest.raises(TypeError, match="real floating"):
            rollout.predict_step(_OutputModel(output), input_value, model_kind="fno", hidden=None)

    events: list[str] = []
    original_require_prediction = rollout._require_prediction

    def observe_validation(value: object, *, batch_size: int, height: int, width: int) -> torch.Tensor:
        events.append("validation")
        return original_require_prediction(value, batch_size=batch_size, height=height, width=width)

    def timing_wrapper(invoke: object) -> object:
        events.append("before")
        result = invoke()  # type: ignore[operator]
        events.append("after")
        return result

    monkeypatch.setattr(rollout, "_require_prediction", observe_validation)
    prediction, hidden = rollout.predict_step(
        _OutputModel(torch.zeros(1, 4, 2, 3), events),
        input_value,
        model_kind="fno",
        hidden=None,
        model_call=timing_wrapper,
    )
    assert prediction.shape == (1, 4, 2, 3)
    assert hidden is None
    assert events == ["before", "model", "after", "validation"]


def test_registered_airflow_default_rejects_task_policy_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the default only from the transient TaskSpec and fail if it changes."""
    context = _context(_IncrementModel())
    monkeypatch.setattr(transient.domain.tasks.registry, "get_task", lambda _task_id: SimpleNamespace(training_airflow_source="external"))
    with pytest.raises(RuntimeError, match="comsol_reference"):
        transient.predict_transient_step(context, **_request(1))
