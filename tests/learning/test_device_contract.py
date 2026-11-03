"""Protect shared requested-versus-resolved runtime device behavior at every boundary.

Responsibilities:
  - Check requested/resolved devices, CPU silence, fallback, precision, and immutable resume
    admission

Design principles:
  - Explicit hardware failure never silently falls back or mutates saved runs

This module does NOT:
  - Validate model numerics or select scheduler queues
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support import configs, run_state

from src import common, experiments, learning

if TYPE_CHECKING:
    from pathlib import Path

_MOCK_CUDA_INDEX = 2


def _forbid_cuda_query(*_args: Any, **_kwargs: Any) -> Any:
    """Fail if a CPU-only path touches a CUDA availability/property API."""
    message = "CPU policy queried CUDA"
    raise AssertionError(message)


def _hide_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CUDA deterministically unavailable at the shared resolver seam."""
    monkeypatch.setattr(learning.device.torch.cuda, "is_available", lambda: False)


def _file_inventory(root: Path) -> dict[str, bytes]:
    """Return relative file contents below one mutation boundary."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _mock_cuda_resolution() -> learning.device.DeviceResolution:
    """Return an indexed CUDA runtime fact without querying physical hardware."""
    metadata = learning.device.DeviceRuntimeMetadata(
        requested_policy="cuda",
        resolved_device="cuda:2",
        device_type="cuda",
        pytorch_version=str(torch.__version__),
        cuda_index=_MOCK_CUDA_INDEX,
        cuda_device_name="Mock GPU 2",
    )
    return learning.device.DeviceResolution(
        requested_policy="cuda",
        device=torch.device("cuda:2"),
        device_type="cuda",
        cuda_index=_MOCK_CUDA_INDEX,
        metadata=metadata,
    )


def test_cpu_resolution_is_immutable_serializable_and_cuda_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve explicit CPU while making every CUDA availability/property probe fail.

    The immutable result must serialize requested/resolved CPU facts without CUDA
    fields, protecting genuinely CUDA-silent CPU operation.
    """
    for name in ("is_available", "current_device", "get_device_name"):
        monkeypatch.setattr(learning.device.torch.cuda, name, _forbid_cuda_query)

    resolution = learning.device.resolve_device("cpu", path="run.device")
    metadata = resolution.as_dict()

    assert resolution.requested_policy == "cpu"
    assert resolution.device == torch.device("cpu")
    assert resolution.device_type == "cpu"
    assert resolution.cuda_index is None
    assert metadata["requested_policy"] == "cpu"
    assert metadata["resolved_device"] == "cpu"
    assert not {"cuda_index", "cuda_device_name", "cuda_runtime_version"}.intersection(metadata)
    json.dumps(metadata)


def test_auto_resolution_falls_back_to_cpu_only_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve ``auto`` with CUDA hidden and property probes forbidden.

    Metadata must preserve the requested policy but record concrete CPU, separating
    user intent from the runtime fact used by training.
    """
    _hide_cuda(monkeypatch)
    monkeypatch.setattr(learning.device.torch.cuda, "current_device", _forbid_cuda_query)
    monkeypatch.setattr(learning.device.torch.cuda, "get_device_name", _forbid_cuda_query)

    resolution = learning.device.resolve_device("auto", path="run.device")

    assert resolution.requested_policy == "auto"
    assert resolution.device == torch.device("cpu")
    assert resolution.as_dict()["requested_policy"] == "auto"
    assert resolution.as_dict()["resolved_device"] == "cpu"


def test_explicit_unavailable_cuda_raises_project_error_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Request strict CUDA while the shared resolver reports it unavailable.

    Resolution must raise the project error with semantic path context and never
    fall back to CPU, preserving the meaning of an explicit hardware requirement.
    """
    _hide_cuda(monkeypatch)

    with pytest.raises(
        learning.device.DeviceResolutionError,
        match=r"run\.device requested 'cuda'.*never falls back",
    ):
        learning.device.resolve_device("cuda", path="run.device")


def test_invalid_device_values_fail_at_the_exact_semantic_path() -> None:
    """
    Vary an unknown string, indexed device, boolean, number, and null.

    Every invalid family must fail at ``run.device`` while the base YAML remains
    fixed, proving config accepts only ``auto``, ``cuda``, or ``cpu`` strings.
    """
    invalid_values: tuple[Any, ...] = ("unsupported", "cuda:0", True, 0, None)
    for invalid in invalid_values:
        raw = configs.direct_config()
        raw["run"]["device"] = invalid
        with pytest.raises(
            experiments.config.loader.ConfigError,
            match=r"run\.device must be exactly one of: auto, cuda, cpu",
        ):
            configs.resolve(raw)


def test_cpu_mixed_precision_is_rejected_before_scaler_construction() -> None:
    """
    Validate mixed precision against one resolved CPU device.

    The boundary must reject it before scaler construction because the maintained
    training contract implements AMP only for CUDA.
    """
    resolution = learning.device.resolve_device("cpu")
    with pytest.raises(learning.device.DeviceResolutionError, match="CPU autocast is not supported"):
        learning.device.validate_mixed_precision_device(True, resolution)


def test_cuda_determinism_policy_is_operation_aware_and_never_mutates_config() -> None:
    """Reject known strict CUDA conflicts while admitting supported effective operations."""
    cuda = _mock_cuda_resolution()
    uno_raw = configs.direct_config(model_kind="uno", physics_enabled=False)
    uno_raw["run"]["deterministic"] = True
    uno = configs.resolve(uno_raw)
    with pytest.raises(
        learning.device.DeviceResolutionError,
        match=r"(?s)UNO 2D resampling.*bicubic.*run\.deterministic: false.*best-effort CUDA reproducibility",
    ):
        experiments.run.validate_deterministic_model_device_policy(uno, cuda)
    assert uno["run"]["deterministic"] is True

    spectral_raw = configs.direct_config(model_kind="fno", physics_enabled=True)
    spectral_raw["run"]["deterministic"] = True
    spectral_raw["loss"]["physics"]["derivatives"] = {
        "kind": "spectral",
        "extension": "fourier_continuation",
    }
    spectral = configs.resolve(spectral_raw)
    experiments.run.validate_deterministic_model_device_policy(spectral, cuda)
    assert spectral["run"]["deterministic"] is True

    for config in (uno, spectral):
        config["run"]["deterministic"] = False
        experiments.run.validate_deterministic_model_device_policy(config, cuda)
        config["run"]["deterministic"] = True
        experiments.run.validate_deterministic_model_device_policy(config, learning.device.resolve_device("cpu"))

    data_only = configs.resolve(
        configs.direct_config(model_kind="fno", physics_enabled=False),
    )
    physical = configs.resolve(
        configs.direct_config(model_kind="fno", physics_enabled=True),
    )
    experiments.run.validate_deterministic_model_device_policy(data_only, cuda)
    experiments.run.validate_deterministic_model_device_policy(physical, cuda)


@pytest.mark.parametrize("saved_config_resume", [False, True])
def test_strict_cuda_does_not_mutate_an_existing_run_that_would_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_config_resume: bool,
) -> None:
    """
    Continue an existing interrupted run with strict CUDA unavailable.

    Rerunning the authored config and resuming from the saved config must both fail on
    device resolution; the byte inventory and lease state stay unchanged, proving device
    resolution precedes every mutation of an existing run.
    """
    _hide_cuda(monkeypatch)
    training_root = tmp_path / "training"
    monkeypatch.setenv("STORAGE_ROOT", str(training_root))
    output_root = tmp_path / "outputs"
    config = configs.resolve(configs.direct_config(), output_root=output_root)
    run_dir = common.paths.resolve_config_run_dir(config)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    before = _file_inventory(run_dir)

    request: dict[str, Any] = (
        {"config_path": common.paths.resolve_run_config_path(run_dir), "resume": run_dir}
        if saved_config_resume
        else {"config_path": configs.write_config(tmp_path, configs.direct_config(), canonical_root=tmp_path), "output_root": output_root}
    )
    with pytest.raises(learning.device.DeviceResolutionError):
        experiments.run.run_experiment(device="cuda", **request)

    assert _file_inventory(run_dir) == before
    assert not (training_root / ".state").exists()
    assert not list(run_dir.rglob("*.lock"))
