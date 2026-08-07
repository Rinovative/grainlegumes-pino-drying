# ruff: noqa: S101, PLR2004
"""CPU resource-plan, node-worker, and scheduler-command contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from src import generation
from src.generation.cli import cli_generation

if TYPE_CHECKING:
    from pathlib import Path


def test_resource_equations_and_invalid_allocations() -> None:
    """Protect all hard resource equations at their authoritative owning layer."""
    plan = generation.cluster.build_resource_plan(
        max_nodes=4,
        cases_per_node=4,
        cores_per_case=8,
        max_parallel_cases=10,
        cores_per_node=32,
        remaining_cases=100,
    )
    assert plan.effective_parallel_cases == 10
    assert plan.effective_nodes == 3
    with pytest.raises(ValueError, match="cores_per_node"):
        generation.cluster.build_resource_plan(
            max_nodes=4,
            cases_per_node=5,
            cores_per_case=8,
            max_parallel_cases=16,
            cores_per_node=32,
            remaining_cases=100,
        )
    with pytest.raises(ValueError, match=r"max_nodes \* cases_per_node"):
        generation.cluster.build_resource_plan(
            max_nodes=2,
            cases_per_node=2,
            cores_per_case=8,
            max_parallel_cases=5,
            cores_per_node=32,
            remaining_cases=10,
        )
    with pytest.raises(ValueError, match="max_nodes"):
        generation.cluster.build_resource_plan(
            max_nodes=0,
            cases_per_node=1,
            cores_per_case=1,
            max_parallel_cases=1,
            cores_per_node=32,
            remaining_cases=1,
        )


def test_multi_worker_batch_and_scheduler_dry_run(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect bounded multi-wave claims, duplicate-free publication, global cap, and Slurm dry-run."""
    config_path, _template = generation_config_factory(
        case_indices=[1, 2, 3, 4, 5, 6],
        executable=fake_comsol,
        scheduler_kind="slurm",
    )
    config = generation.config.load_generation_config(config_path)
    storage = tmp_path / "storage"
    tracker = tmp_path / "tracker.json"
    monkeypatch.setenv("FAKE_COMSOL_TRACKER", str(tracker))
    monkeypatch.setenv("FAKE_COMSOL_DELAY", "0.05")
    plan = generation.cluster.build_resource_plan(
        max_nodes=2,
        cases_per_node=2,
        cores_per_case=1,
        max_parallel_cases=3,
        cores_per_node=32,
        remaining_cases=6,
    )
    assert plan.effective_nodes == 2
    results = generation.cluster.run_local_batch(
        config,
        config.case_indices,
        plan=plan,
        storage_root=storage,
        work_root=tmp_path / "work",
    )
    assert len(results) == 2
    assert sorted(path.name for path in (storage / "01_generation" / "processed" / config.batch_id).iterdir()) == [
        f"case_{index:04d}" for index in config.case_indices
    ]
    tracker_state = json.loads(tracker.read_text(encoding="utf-8"))
    assert tracker_state["starts"] == 6
    assert tracker_state["maximum"] <= 3
    generation.runtime.validate_terminal_batch(config, storage_root=storage)

    command = generation.cluster.build_slurm_submission_command(config, config.case_indices, plan=plan)
    assert command[0] == "sbatch"
    assert "--nodes=1" in command
    assert "--array=0-1%2" in command
    assert "--cpus-per-task=2" in command
    assert not any("-nn" in argument for argument in command)
    assert all(flag in command[-1] for flag in ("--max-nodes", "--cases-per-node", "--cores-per-case", "--max-parallel-cases"))


def test_production_cli_requires_all_resource_limits(generation_config_factory: Any) -> None:
    """Protect mandatory production flags at the thin parser boundary."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    with pytest.raises(SystemExit) as error:
        cli_generation.main(["print-submit", str(config_path), "--max-nodes", "1"])
    assert error.value.code == 2
