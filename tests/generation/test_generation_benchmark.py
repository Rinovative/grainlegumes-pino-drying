# ruff: noqa: S101
"""Core-benchmark resources remain outside generated scientific identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import generation

if TYPE_CHECKING:
    import pytest

_COMMIT = "a" * 40
_SMOKE_DIGEST = "b" * 64


def _repository() -> Path:
    """Return the repository root used by the maintained benchmark request."""
    return Path(__file__).resolve().parents[2]


def _suite() -> Any:
    """Load the benchmark request without requiring native COMSOL executables."""
    return generation.benchmark.load_core_benchmark_suite(
        _repository() / "configs/generation/benchmarks/transient_core_scaling/suite.yaml",
        require_executable=False,
    )


def test_resource_change_preserves_case_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep scheduler resources operational while generated science stays stable."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", _COMMIT)
    suite = _suite()
    baseline_variant = suite.variants[0]
    replacement_cores = (
        baseline_variant.cores_per_case + 1 if baseline_variant.cores_per_case < suite.cores_per_node else baseline_variant.cores_per_case - 1
    )
    changed_variant = replace(baseline_variant, cores_per_case=replacement_cores)

    baseline = generation.cases.case.generate_case_input_bundle(
        suite.case_config,
        suite.case_index,
        tmp_path / "baseline",
    )
    changed = generation.cases.case.generate_case_input_bundle(
        suite.case_config,
        suite.case_index,
        tmp_path / "changed-resource",
    )

    assert suite.execution_id(changed_variant) != suite.execution_id(baseline_variant)
    assert generation.benchmark.core_benchmark_run_id(
        suite,
        git_commit=_COMMIT,
        smoke_gate_digest=_SMOKE_DIGEST,
    )
    assert changed.case_input_id == baseline.case_input_id
    assert changed.simulation_case_id == baseline.simulation_case_id
    assert changed.case_payload == baseline.case_payload
