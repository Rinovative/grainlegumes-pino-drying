"""Protect labeled subseeds and pre-construction process reproducibility controls.

Responsibilities:
  - Check stable labeled subseeds, process RNGs, deterministic controls, and model initialization
    order

Design principles:
  - CPU-safe fixtures test reproducibility within the current library/runtime boundary

This module does NOT:
  - Promise bitwise equality across hardware versions or restore checkpoint RNGs
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from src import experiments, learning
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum, RolloutCurriculumState

if TYPE_CHECKING:
    from pathlib import Path


def test_labeled_subseeds_are_stable_distinct_and_reproducible() -> None:
    """
    Derive the full seed plan and the same labels in reverse call order.

    Values must be identical and pairwise distinct, protecting component streams
    from accidental dependence on orchestration call sequence.
    """
    first = experiments.run.build_seed_plan(19)
    reverse = {label: experiments.run.derive_subseed(19, label) for label in reversed(tuple(first))}

    assert first == reverse
    assert len(first) == len(set(first.values()))


def test_curriculum_draws_follow_the_derived_seed_of_each_arm() -> None:
    """
    Draw rollout lengths from the seed plan's A0 and B curriculum seeds.

    The same run seed reproduces its draws, another run seed and the other arm draw
    different sequences, so replicates do not share one rollout-length sequence.
    """
    curriculum = RolloutCurriculum(lengths=(2, 4, 8, 16, 32), milestone_fractions=(0.0, 0.2, 0.4, 0.6, 0.8))

    def draws(run_seed: int, label: str) -> list[int]:
        state = RolloutCurriculumState.create(curriculum, seed=experiments.run.build_seed_plan(run_seed)[label])
        return [state.select(progress=1.0, available_length=32, batch_size=1)[0] for _ in range(32)]

    assert draws(19, "curriculum_a0") == draws(19, "curriculum_a0")
    assert draws(19, "curriculum_a0") != draws(19, "curriculum_b")
    assert draws(19, "curriculum_a0") != draws(23, "curriculum_a0")


def test_process_seed_reproduces_python_numpy_torch_and_model_init() -> None:
    """
    Seed CPU execution twice around Python, NumPy, Torch, and model initialization.

    Every produced value and weight must repeat exactly, establishing the process
    reproducibility boundary used before component construction.
    """
    seed = experiments.run.derive_subseed(11, "model_init")
    experiments.run.seed_process(seed, device=torch.device("cpu"))
    first = (
        random.random(),  # noqa: S311 -- deterministic training-seed test exercises process-global Python RNG
        float(np.random.random()),  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
        torch.rand(3),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )
    experiments.run.seed_process(seed, device=torch.device("cpu"))
    second = (
        random.random(),  # noqa: S311 -- deterministic training-seed test exercises process-global Python RNG
        float(np.random.random()),  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
        torch.rand(3),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])


def test_run_deterministic_controls_torch_settings() -> None:
    """
    Toggle deterministic orchestration on and then off in one CPU process.

    Torch algorithm and cuDNN flags must follow the setting in both directions,
    proving the persisted flag controls implemented runtime behavior.
    """
    experiments.run.configure_determinism(True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark

    experiments.run.configure_determinism(False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark


class _Processor:
    """Minimal serializable data processor for construction-order testing."""

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a small serializable state."""
        return {"value": torch.tensor(1)}


def test_model_subseed_is_applied_immediately_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Record process seeds while a stub loader is built before a deliberately stopped model build.

    The final seed must be the labeled ``model_init`` subseed and the run must fail
    cleanly, protecting model weights from RNG consumed during data construction.
    """
    config: dict[str, Any] = {
        "task": "synthetic",
        "run": {"name": "run", "seed": 23, "deterministic": True, "device": "cpu"},
        "training": {"epochs": 1},
        "tracking": {"wandb": {"mode": "disabled"}},
        "evaluation": {
            "objective": {
                "id": "objective",
                "kind": "objective",
                "space": "normalized",
                "fields": ["value"],
                "reduction": "element_mean",
                "direction": "minimize",
            }
        },
        "model": {"kind": "synthetic"},
        "paths": {"output_root": str(tmp_path)},
    }
    # The stub config has no resolved identity to register; this test only observes seeding order.
    monkeypatch.setattr(experiments.registry, "reserve", lambda *_args: nullcontext())
    run_dir = experiments.run.prepare_fresh_run(config, run_dir=tmp_path / "run")
    seed_calls: list[int] = []
    expected = experiments.run.build_seed_plan(23)

    def capture_seed(seed: int, *, device: torch.device) -> None:
        assert device == torch.device("cpu")
        seed_calls.append(seed)

    monkeypatch.setattr(experiments.run, "seed_process", capture_seed)
    # The stub config carries no data, optimizer or loss sections for the start block, which consumes no randomness.
    monkeypatch.setattr(experiments.run.console.ConsoleReporter, "startup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        experiments.run.config_loader,
        "create_dataloaders_from_config",
        # The construction-order test stays isolated from Dataset identity: the loaders hand over an opaque scaling artifact.
        lambda *_args, **_kwargs: {
            "data_processor": _Processor(),
            "scaling_artifact": {"state": _Processor().state_dict()},
            "split_indices": {},
        },
    )
    monkeypatch.setattr(
        experiments.run.config_loader,
        "validate_resolved_task_contract",
        lambda _config: object(),
    )

    class ConstructionReachedError(RuntimeError):
        """Stop the orchestration immediately after the ordering assertion."""

    def build_model(_config: dict[str, Any], *, device: torch.device) -> torch.nn.Module:
        assert device == torch.device("cpu")
        assert seed_calls[-1] == expected["model_init"]
        raise ConstructionReachedError

    monkeypatch.setattr(learning.models.factory, "build_model", build_model)
    with pytest.raises(ConstructionReachedError):
        experiments.run.execute_prepared_run(
            config,
            run_dir=run_dir,
            persisted_config=config,
            device_resolution=learning.device.resolve_device("cpu"),
        )

    assert seed_calls == [expected["process"], expected["model_init"]]
    assert experiments.run.read_run_summary(run_dir)["status"] == "failed"
