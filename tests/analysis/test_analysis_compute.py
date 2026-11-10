"""Scientific and routing checks for bounded Evaluation analysis reductions.

Responsibilities:
  - Check full-grid error reduction, explicit device selection, and CUDA resource fallback

Design principles:
  - Missing support differs from invalid finite predictions; programming errors propagate

This module does NOT:
  - Benchmark GPU throughput or alter analysis device policy
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from src.analysis.evaluation import evaluation_transient_artifact as transient_artifact
from src.analysis.evaluation.evaluation_plot import evaluation_plot_transient as transient_plots
from src.analysis.presentation import analysis_compute as compute


def test_cpu_error_batch_preserves_full_grid_support_and_invalid_finite_values() -> None:
    """A finite unphysical value contributes while absent support does not."""
    reference = np.zeros((2, 2, 2), dtype=np.float64)
    prediction = np.array(([[1.0, -2.0], [3.0, 99.0]], [[-1.0, 4.0], [0.0, 8.0]]))
    support = np.array(([[True, True], [False, False]], [[True, False], [True, True]]))
    result = compute.reduce_error_batch(reference, prediction, support, device=compute.resolve_analysis_device("cpu"))
    np.testing.assert_array_equal(result.absolute_sum, [[2.0, 2.0], [0.0, 8.0]])
    np.testing.assert_array_equal(result.signed_sum, [[0.0, -2.0], [0.0, 8.0]])
    np.testing.assert_array_equal(result.squared_sum, [[2.0, 4.0], [0.0, 64.0]])
    np.testing.assert_array_equal(result.support_count, [[2, 1], [1, 1]])
    assert result.device.backend == "cpu"


def test_nonfinite_required_support_is_rejected_but_missing_support_is_not() -> None:
    """Only explicit absent support permits a nonfinite stored value."""
    reference = np.zeros((1, 1, 2), dtype=np.float64)
    prediction = np.array([[[np.nan, 2.0]]], dtype=np.float64)
    allowed = compute.reduce_error_batch(reference, prediction, np.array([[[False, True]]]), device=compute.resolve_analysis_device("cpu"))
    np.testing.assert_array_equal(allowed.support_count, [[0, 1]])
    with pytest.raises(ValueError, match="non-finite"):
        compute.reduce_error_batch(reference, prediction, np.array([[[True, True]]]), device=compute.resolve_analysis_device("cpu"))
    with pytest.raises(ValueError, match="summary is non-finite"):
        compute.reduce_error_batch(
            np.zeros((1, 1, 1), dtype=np.float64),
            np.full((1, 1, 1), 1e200, dtype=np.float64),
            np.ones((1, 1, 1), dtype=bool),
            device=compute.resolve_analysis_device("cpu"),
        )


def test_device_selection_is_explicit_and_auto_falls_back_without_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CPU, auto without Slurm and explicit-CUDA failure are deterministic."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert compute.resolve_analysis_device("cpu").logical_device == "cpu"
    assert compute.resolve_analysis_device("auto").backend == "cpu"
    assert "no Slurm GPU allocation" in compute.resolve_analysis_device("auto").reason
    assert "CUDA when a Slurm-allocated device is visible" in compute.selection_status("auto")
    with pytest.raises(RuntimeError, match="Slurm-allocated"):
        compute.resolve_analysis_device("cuda")
    with pytest.raises(ValueError, match="cpu, auto"):
        compute.resolve_analysis_device("gpu")


def test_auto_without_visible_cuda_falls_back_to_cpu_and_explicit_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under Slurm without a visible device auto reports CPU and explicit CUDA fails."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")

    class NoCuda:
        class cuda:  # noqa: N801
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(compute, "_torch_cuda", lambda: NoCuda)
    assert "no visible accessible CUDA device" in compute.resolve_analysis_device("auto").reason
    with pytest.raises(RuntimeError, match="visible accessible"):
        compute.resolve_analysis_device("cuda")


def test_auto_selects_visible_cuda_and_admission_leaves_cuda_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto picks a visible working device for any workload; early admission never imports torch."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def synchronize(_index: int) -> None:
            return None

    class FakeTorch:
        cuda = FakeCuda()
        float64 = object()
        inference_mode = staticmethod(nullcontext)

        @staticmethod
        def ones(*_args: object, **_kwargs: object) -> float:
            return 1.0

    def forbidden() -> None:
        msg = "admission must not touch CUDA"
        raise AssertionError(msg)

    monkeypatch.setattr(compute, "_torch_cuda", forbidden)
    compute.admit_analysis_request("auto")
    compute.AnalysisComputePreference("auto").set_requested("auto")
    monkeypatch.setattr(compute, "_torch_cuda", lambda: FakeTorch)
    device = compute.resolve_analysis_device("auto")
    assert (device.backend, device.logical_device) == ("cuda", "cuda:0")


def test_workspace_compute_preference_validates_then_notifies_once() -> None:
    """Changing one route notifies mounted views only after admission."""
    preference = compute.AnalysisComputePreference("auto")
    observed: list[str] = []

    def rerender() -> None:
        observed.append(preference.requested)

    preference.subscribe(rerender)
    preference.set_requested("cpu")
    preference.set_requested("cpu")
    with pytest.raises(ValueError, match="cpu, auto"):
        preference.set_requested("gpu")
    assert observed == ["cpu"]
    assert preference.requested == "cpu"


def test_auto_probe_does_not_hide_nonresource_cuda_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected CUDA probe errors propagate before any numerical output."""
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    class FakeTorch:
        cuda = FakeCuda()
        float64 = object()
        inference_mode = staticmethod(nullcontext)

        @staticmethod
        def ones(*_args: object, **_kwargs: object) -> None:
            msg = "kernel contract bug"
            raise RuntimeError(msg)

    monkeypatch.setattr(compute, "_torch_cuda", lambda: FakeTorch)
    with pytest.raises(RuntimeError, match=r"Auto CUDA.*kernel contract bug"):
        compute.resolve_analysis_device("auto")


def test_auto_resource_fallback_discards_gpu_result_and_reports_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known resource failure retries once on CPU with visible provenance."""
    selected = compute.AnalysisDevice("auto", "cuda", "cuda:0", "test route")

    def out_of_memory(*_args: object) -> None:
        msg = "CUDA out of memory"
        raise RuntimeError(msg)

    monkeypatch.setattr(compute, "_cuda_error_batch", out_of_memory)
    values = np.ones((1, 1, 1), dtype=np.float64)
    result = compute.reduce_error_batch(values, values + 2.0, np.ones_like(values, dtype=bool), device=selected)
    assert result.device.backend == "cpu"
    assert "fallback" in result.device.reason
    np.testing.assert_array_equal(result.absolute_sum, [[2.0]])
    explicit = compute.AnalysisDevice("cuda", "cuda", "cuda:0", "test route")
    with pytest.raises(RuntimeError, match="out of memory"):
        compute.reduce_error_batch(values, values + 2.0, np.ones_like(values, dtype=bool), device=explicit)


def test_auto_never_hides_programming_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected execution errors are not treated as resource failures."""
    selected = compute.AnalysisDevice("auto", "cuda", "cuda:0", "test route")

    def broken(*_args: object) -> None:
        msg = "shape contract bug"
        raise RuntimeError(msg)

    monkeypatch.setattr(compute, "_cuda_error_batch", broken)
    values = np.ones((1, 1, 1), dtype=np.float64)
    with pytest.raises(RuntimeError, match="shape contract bug"):
        compute.reduce_error_batch(values, values, np.ones_like(values, dtype=bool), device=selected)


def test_actual_cuda_batch_matches_cpu_when_one_allocated_device_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the implemented float64 kernel and support on allocated CUDA."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("No CUDA device is visible in this allocated process.")
    # The suite isolates the inherited job identity, so the allocation is declared explicitly.
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    reference = np.arange(24, dtype=np.float64).reshape(3, 2, 4) / 4.0
    prediction = reference + np.array((1.0, -2.0, 0.5), dtype=np.float64)[:, None, None]
    support = np.ones_like(reference, dtype=bool)
    support[1, 0, 2] = False
    cpu = compute.reduce_error_batch(reference, prediction, support, device=compute.resolve_analysis_device("cpu"))
    gpu = compute.reduce_error_batch(reference, prediction, support, device=compute.resolve_analysis_device("cuda:0"))
    assert gpu.device.logical_device == "cuda:0"
    np.testing.assert_array_equal(gpu.support_count, cpu.support_count)
    np.testing.assert_allclose(gpu.absolute_sum, cpu.absolute_sum, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gpu.signed_sum, cpu.signed_sum, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gpu.squared_sum, cpu.squared_sum, rtol=1e-14, atol=1e-14)
    large_reference = np.zeros((32, 251, 401), dtype=np.float64)
    large_prediction = np.ones_like(large_reference)
    large_support = np.ones_like(large_reference, dtype=bool)
    auto = compute.reduce_error_batch(large_reference, large_prediction, large_support, device=compute.resolve_analysis_device("auto"))
    assert auto.device.backend == "cuda"
    np.testing.assert_array_equal(auto.absolute_sum, np.full((251, 401), 32.0))
    np.testing.assert_array_equal(auto.support_count, np.full((251, 401), 32))


def test_transient_aggregate_auto_retains_full_grid_and_reports_cpu_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small exact-time aggregate uses all supported cells before display."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    x, y = np.meshgrid(np.array((0.0, 1.0)), np.array((0.0, 1.0)))
    static = np.zeros((len(transient_artifact.STATIC_ORDER), 2, 2), dtype=np.float64)
    static[0] = x
    static[1] = y
    reference = np.zeros((2, len(transient_artifact.STATE_ORDER), 2, 2), dtype=np.float64)

    def record(error: float, support: np.ndarray) -> dict[str, object]:
        prediction = reference.copy()
        prediction[1, 0] = error
        return {
            "mode": "autonomous_full",
            "requested_horizon": "full",
            "physical_times": np.array((0.0, 1.0)),
            "reference_states": reference,
            "predicted_states": prediction,
            "spatial_mask": support,
            "static_conditioning": static,
        }

    evidence: dict[str, str] = {}
    result = transient_plots._aggregate_snapshot_evidence(
        (record(1.0, np.ones((2, 2), dtype=bool)), record(3.0, np.array([[True, False], [True, True]]))),
        state_field=transient_artifact.STATE_ORDER[0],
        physical_time=1.0,
        compute_device="auto",
        device_evidence=evidence,
    )
    np.testing.assert_allclose(result[2], [[2.0, 1.0], [2.0, 2.0]])
    assert "resolved cpu" in evidence["route"]
