"""Verify direct timing, the steady speed-ups as ratios of sums, strict persistence and cache independence.

Responsibilities:
  - Check direct inference timing, paired steady speed-ups, solver hardware, strict schemas and atomic persistence

Design principles:
  - Missing COMSOL evidence stays unavailable and counted; scientific identity excludes timing

This module does NOT:
  - Parse live solver logs or benchmark production models
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

from src import analysis
from src.analysis.artifacts import analysis_artifact_performance as artifact_performance

if TYPE_CHECKING:
    from pathlib import Path


class _RecordingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.inference_modes: list[bool] = []
        self.batch_sizes: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.batch_sizes.append(int(value.shape[0]))
        return value[:, :1] * self.weight


def _neural_runtime() -> dict[str, Any]:
    return {
        "requested_policy": "cpu",
        "resolved_device": "cpu",
        "device_type": "cpu",
        "pytorch_version": str(torch.__version__),
        "hostname": "test-host",
        "platform": "test-platform",
        "processor": "test-cpu",
        "python_version": "3.11",
        "inference_dtype": "float32",
        "torch_num_threads": 1,
    }


def _neural_cases() -> list[dict[str, Any]]:
    return [
        {"case_id": "case_0001", "source_index": 0, "batch_amortized_neural_operator_forward_s": 0.1},
        {"case_id": "case_0002", "source_index": 2, "batch_amortized_neural_operator_forward_s": 0.2},
        {"case_id": "case_0003", "source_index": 3, "batch_amortized_neural_operator_forward_s": 0.3},
    ]


def _solver_cases(*, solver: tuple[float | None, ...] = (4.0, None, 6.0)) -> dict[str, dict[str, Any]]:
    hosts = ("node-a", "node-b", "node-a")
    return {
        f"case_000{index + 1}": {"comsol_solver_s": seconds, "comsol_process_s": process, "solver_hostname": host, "solver_requested_cores": 8}
        for index, (seconds, process, host) in enumerate(zip(solver, (20.0, 30.0, 10.0), hosts, strict=True))
    }


def _comparison_arguments(*, solver_cases: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "split_role": "eval",
        "dataset_identity": {"name": "batch-a", "dataset_provenance_sha256": "dataset-digest", "effective_ordered_source_indices_sha256": "indices"},
        "model_identity": {"run_name": "run-a", "effective_config_digest": "config-digest", "best_checkpoint_sha256": "checkpoint-digest"},
        "neural_runtime": _neural_runtime(),
        "batch_size": 2,
        "cases": _neural_cases(),
        "solver_cases": _solver_cases() if solver_cases is None else solver_cases,
    }


def _comparison(*, solver_cases: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    return analysis.artifacts.timing.build_runtime_comparison(**_comparison_arguments(solver_cases=solver_cases))


def test_cpu_forward_is_direct_inference_mode_and_never_uses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that cpu forward is direct inference mode and never uses cuda."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        msg = "CPU timing accessed CUDA"
        raise AssertionError(msg)

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    model = _RecordingProjection()
    prediction, duration = analysis.artifacts.timing.measure_forward(
        model=model,
        normalized_inputs=torch.ones(3, 2, 2, 2),
        device=torch.device("cpu"),
    )
    assert prediction.shape == (3, 1, 2, 2)
    assert duration > 0.0
    assert model.training is False
    assert model.inference_modes == [True]


def test_speedups_are_ratios_of_sums_over_the_paired_cases() -> None:
    """
    Pair three measured cases with their COMSOL clocks, one of which lacks its solver time.

    The solver speed-up is (4 + 6) / (0.1 + 0.3) = 25 over the two paired cases, not the mean of the
    per-case ratios (30); the process speed-up pairs all three cases. Batch size and both hardware
    records are kept.
    """
    payload = _comparison()
    solver, process = payload["speedups"]["solver_speedup"], payload["speedups"]["process_speedup"]

    assert (solver["paired_case_count"], solver["unavailable_case_count"]) == (2, 1)
    assert solver["ratio_of_sums"] == pytest.approx(25.0, rel=1e-12)
    assert solver["ratio_of_sums"] != pytest.approx(0.5 * (4.0 / 0.1 + 6.0 / 0.3))
    assert (process["paired_case_count"], process["unavailable_case_count"]) == (3, 0)
    assert process["ratio_of_sums"] == pytest.approx(60.0 / 0.6, rel=1e-12)
    assert payload["cases"][1]["comsol_solver_s"] is None
    assert payload["measurement"]["batch_size"] == 2
    assert payload["neural_runtime"]["processor"] == "test-cpu"
    assert payload["solver_hardware"] == [
        {"hostname": "node-a", "requested_cores": 8, "case_count": 2},
        {"hostname": "node-b", "requested_cores": 8, "case_count": 1},
    ]


def test_missing_timing_is_unavailable_never_zero() -> None:
    """Generation evidence without a positive solver time pairs no case, and the speed-up is unavailable."""
    stub = SimpleNamespace(stationary_airflow_solver_seconds=None, comsol_process_seconds=0.0, solver_hostname="node-a", solver_requested_cores=4)
    projected = analysis.artifacts.timing.solver_case_timing(stub)  # type: ignore[arg-type]
    assert (projected["comsol_solver_s"], projected["comsol_process_s"]) == (None, None)

    payload = _comparison(solver_cases=_solver_cases(solver=(None, None, None)))
    solver = payload["speedups"]["solver_speedup"]
    assert (solver["paired_case_count"], solver["unavailable_case_count"]) == (0, 3)
    assert (solver["ratio_of_sums"], solver["reference_seconds_sum"], solver["surrogate_seconds_sum"]) == (None, None, None)
    with pytest.raises(ValueError, match="no Generation solver timing"):
        _comparison(solver_cases={case_id: value for case_id, value in _solver_cases().items() if case_id != "case_0002"})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["cases"][0].__setitem__("batch_amortized_neural_operator_forward_s", 0.0),
        lambda payload: payload["cases"][0].__setitem__("batch_amortized_neural_operator_forward_s", float("inf")),
        lambda payload: payload["cases"][1].__setitem__("comsol_solver_s", 0.0),
        lambda payload: payload["speedups"]["solver_speedup"].__setitem__("ratio_of_sums", 99.0),
        lambda payload: payload["solver_hardware"][0].__setitem__("case_count", 3),
        lambda payload: payload["cases"].append(copy.deepcopy(payload["cases"][0])),
        lambda payload: payload["cases"][0].pop("source_index"),
        lambda payload: payload.__setitem__("surrogate_boundary", "isolated single-case latency"),
        lambda payload: payload.__setitem__("aggregates", {}),
    ],
    ids=(
        "zero-forward",
        "infinite-forward",
        "zero-solver-time",
        "edited-speedup",
        "edited-hardware",
        "duplicate-case",
        "missing-field",
        "relabelled-boundary",
        "old-schema-field",
    ),
)
def test_runtime_comparison_rejects_invalid_edited_or_old_payloads(mutation: Any) -> None:
    """Admission rebuilds the speed-ups and the hardware census from the cases and refuses anything else."""
    payload = _comparison()
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        analysis.artifacts.timing.validate_runtime_comparison(payload)


def test_cuda_runtime_names_the_cuda_build_and_refuses_the_old_runtime_name() -> None:
    """A CUDA surrogate records the CUDA build PyTorch was compiled with; the old ``cuda_runtime_version`` field is refused."""
    cuda = {**_neural_runtime(), "resolved_device": "cuda:0", "device_type": "cuda", "cuda_index": 0, "cuda_device_name": "GPU"}
    payload = analysis.artifacts.timing.build_runtime_comparison(
        **{**_comparison_arguments(), "neural_runtime": {**cuda, "cuda_build_version": "12.1"}},
    )
    assert payload["neural_runtime"]["cuda_build_version"] == "12.1"

    payload["neural_runtime"]["cuda_runtime_version"] = payload["neural_runtime"].pop("cuda_build_version")
    with pytest.raises(ValueError, match="invalid fields"):
        analysis.artifacts.timing.validate_runtime_comparison(payload)


def test_stage_timing_stays_outside_payloads_and_its_block_is_closed(tmp_path: Path) -> None:
    """Timing merges into the stage marker atomically, leaves the payload manifest alone and must be complete."""
    artifact_root = tmp_path / "artifact"
    npz_root = artifact_root / "npz"
    npz_root.mkdir(parents=True)
    (artifact_root / "cases.parquet").write_bytes(b"scientific-table")
    (npz_root / "case_0001.npz").write_bytes(b"scientific-case")
    marker = analysis.artifacts.contracts.artifact_provenance_path(artifact_root)
    marker.write_text(json.dumps({"run": {"name": "run"}}), encoding="utf-8")
    before = analysis.artifacts.contracts.artifact_output_manifest(artifact_root)
    snapshot = artifact_performance.ArtifactPerformanceRecorder(device="cpu", dtype="float32").snapshot()

    analysis.artifacts.timing.write_stage_timing(artifact_root, {"report": _comparison()})
    partial = json.loads(marker.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="--rebuild"):
        analysis.artifacts.timing.timing_block(partial, records=frozenset({"report"}))
    with pytest.raises(ValueError, match="bounded output") as bounded:  # a bounded build takes no --rebuild
        analysis.artifacts.timing.timing_block(partial, records=frozenset({"report"}), bounded=True)
    assert "--rebuild" not in str(bounded.value)
    analysis.artifacts.timing.write_stage_timing(artifact_root, {"operational_performance": snapshot})

    document = json.loads(marker.read_text(encoding="utf-8"))
    assert document["run"] == {"name": "run"}
    report = analysis.artifacts.timing.timing_block(document, records=frozenset({"report"}))["report"]
    assert analysis.artifacts.timing.validate_runtime_comparison(report) == _comparison()
    assert not list(artifact_root.glob(".*.tmp"))
    assert analysis.artifacts.contracts.artifact_output_manifest(artifact_root) == before
