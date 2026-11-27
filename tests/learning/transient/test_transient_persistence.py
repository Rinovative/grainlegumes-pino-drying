"""Protect transient adapter checkpoint restore and handoff state.

Responsibilities:
  - Check exact restore, late-failure rollback, and handoff state of common and adapter continuation state

Design principles:
  - Late restore failures roll back common and adapter state transactionally

This module does NOT:
  - Allocate stage runs or test the epoch history, which the shared training-history tests cover
"""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src import learning
from src.datasets.runtime.dataset_runtime_units import UnitSampler

_WORK = {"optimizer_steps": 1, "gradient_predictions": 2, "units": 2, "update_distinct_cases": 2}
_UPDATE_WORK = [[2, 2, 2]]


class _StatefulLoss(nn.Module):
    """Keep a small serializable loss state for restore assertions."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(value))


class _Adapter:
    """Expose controlled adapter state, including a late restore failure."""

    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, Any]:
        """Return the adapter's complete compact continuation state."""
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore one valid state or deliberately reject a test sentinel."""
        if state.get("fail"):
            message = "adapter restore failure"
            raise RuntimeError(message)
        value = state.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            message = "adapter state is invalid"
            raise TypeError(message)
        self.value = value


def _identity(label: str = "transient-persistence") -> dict[str, Any]:
    """Build the complete checkpoint identity required by strict tests."""

    def digest(name: str) -> str:
        return hashlib.sha256(f"{label}:{name}".encode()).hexdigest()

    return {
        "identity_contract_sha256": learning.training.checkpoint.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "transient_drying",
        "task_contract_digest": digest("task"),
        "effective_config_digest": digest("config"),
        "dataset_ids": {"train": "synthetic-train", "material_ood": "synthetic-material-ood", "parameter_ood": "synthetic-parameter-ood"},
        "dataset_fingerprints": {"train": digest("train"), "material_ood": digest("material-ood"), "parameter_ood": digest("parameter-ood")},
        "split_contract_digest": digest("split"),
        "split_membership_digests": {
            "train": digest("train-membership"),
            "validation": digest("validation-membership"),
            "id_test": digest("id-test-membership"),
            "material_ood": digest("material-ood-membership"),
            "parameter_ood": digest("parameter-ood-membership"),
        },
        "normalizer_sha256": digest("normalizer"),
        "numerical_environment": learning.training.checkpoint.numerical_environment_identity(),
        "objective": {"id": "mse", "kind": "mse", "space": "normalized", "fields": ["w"], "reduction": "element_mean", "direction": "minimize"},
    }


def _validated_budget(point: int) -> learning.training.budget.StageBudget:
    """Return a 20-unit budget validated at the given point, with deterministic clock seconds."""
    budget = learning.training.budget.StageBudget(unit="units", total=20)
    budget.record_group(units=point, supervised_transitions=3 * point, train_seconds=0.5 * point)
    budget.record_validation()
    return budget


def _components(seed: int, *, epochs: int = 1) -> tuple[nn.Module, AdamW, _StatefulLoss, DataLoader[Any]]:
    """Build compact stateful CPU components whose unit sampler has committed ``epochs`` complete epochs."""
    torch.manual_seed(seed)
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters(), lr=0.01)
    loss = _StatefulLoss(float(seed))
    sampler = UnitSampler(unit_kind="transition", cases=(0, 0, 1, 1), keys=None, units_per_update=2, memory_batch=2, seed=seed + 1)
    loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4, 1)), batch_sampler=sampler)
    for epoch in range(epochs):
        sampler.begin_epoch(epoch)
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            torch.mean((model(x) - y).square()).backward()
            optimizer.step()
            sampler.advance(len(x))
    return model, optimizer, loss, loader


def _sampler_state(loader: DataLoader[Any]) -> dict[str, Any]:
    """Return the persisted state of a loader's unit sampler."""
    sampler = loader.batch_sampler
    assert isinstance(sampler, UnitSampler)
    return sampler.state_dict()


def _assert_equal(left: Any, right: Any) -> None:
    """Compare nested checkpoint state exactly."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right, strict=True):
            _assert_equal(item_left, item_right)
    else:
        assert left == right


def test_v2_adapter_restore_is_exact_and_late_failure_rolls_back_every_state() -> None:
    """Restore adapter state and roll all mutable/runtime state back after adapter failure."""
    identity = _identity("restore")
    source_model, source_optimizer, source_loss, source_loader = _components(10)
    source_adapter = _Adapter(3)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        payload_kind="full",
        identity=identity,
        completed_epoch=1,
        global_step=1,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        loss=source_loss,
        best_metric=0.5,
        best_epoch=1,
        objective_history=[{"point": 1, "epoch": 1, "objective_id": "mse", "value": 0.5, "bound_violations": {}}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work=_WORK,
        update_work=_UPDATE_WORK,
        budget=_validated_budget(1),
        adapter=source_adapter,
    )
    model, optimizer, loss, loader = _components(20)
    adapter = _Adapter(9)
    budget = learning.training.budget.StageBudget(unit="units", total=20)
    restored = learning.training.checkpoint.restore_checkpoint(
        payload,
        expected_identity=identity,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss=loss,
        train_loader=loader,
        budget=budget,
        adapter=adapter,
        adapter_expected=True,
    )
    assert budget.state_dict() == _validated_budget(1).state_dict()
    assert restored["global_step"] == 1
    assert adapter.value == source_adapter.value
    _assert_equal(model.state_dict(), source_model.state_dict())
    _assert_equal(_sampler_state(loader), _sampler_state(source_loader))

    failing = copy.deepcopy(payload)
    failing["adapter_state_dict"] = {"value": 7, "fail": True}
    # Move the live budget away from the saved state so a partial restore would show.
    budget.record_group(units=3, supervised_transitions=9, train_seconds=1.0)
    before = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "loss": copy.deepcopy(loss.state_dict()),
        "loader": _sampler_state(loader),
        "adapter": adapter.state_dict(),
        "budget": budget.state_dict(),
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
        "torch": torch.get_rng_state().clone(),
    }
    with pytest.raises(RuntimeError, match="adapter restore failure"):
        learning.training.checkpoint.restore_checkpoint(
            failing,
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss=loss,
            train_loader=loader,
            budget=budget,
            adapter=adapter,
            adapter_expected=True,
        )
    _assert_equal(before["budget"], budget.state_dict())
    _assert_equal(before["model"], model.state_dict())
    _assert_equal(before["optimizer"], optimizer.state_dict())
    _assert_equal(before["loss"], loss.state_dict())
    _assert_equal(before["loader"], _sampler_state(loader))
    _assert_equal(before["adapter"], adapter.state_dict())
    _assert_equal(before["python"], random.getstate())
    _assert_equal(before["numpy"], np.random.get_state())  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
    assert torch.equal(before["torch"], torch.get_rng_state())


def test_handoff_restore_preserves_complete_common_state_and_fresh_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue common optimization/RNG state while the target stage owns fresh adapter state."""
    identity = _identity("handoff")
    source_model, source_optimizer, source_loss, source_loader = _components(30, epochs=2)
    source_adapter = _Adapter(4)
    payload = learning.training.checkpoint.make_checkpoint(
        role="best",
        payload_kind="full",
        identity=identity,
        completed_epoch=2,
        global_step=1,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        loss=source_loss,
        best_metric=0.2,
        best_epoch=2,
        objective_history=[{"point": 2, "epoch": 2, "objective_id": "mse", "value": 0.2, "bound_violations": {}}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work=_WORK,
        update_work=_UPDATE_WORK,
        budget=_validated_budget(2),
        adapter=source_adapter,
    )
    # A best never carries schedule state, so a continuation cannot inherit A0's schedule.
    assert payload["scheduler_state_dict"] is None
    model, optimizer, loss, loader = _components(40, epochs=0)
    target_adapter = _Adapter(9)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="schedule state"):
        learning.training.checkpoint.restore_handoff_checkpoint(
            {**payload, "scheduler_state_dict": {"kind": "cosine", "start_rate": 1.0e-3}},
            expected_source_identity=identity,
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_loader=loader,
        )
    _assert_equal(before, model.state_dict())

    restored = learning.training.checkpoint.restore_handoff_checkpoint(
        payload,
        expected_source_identity=identity,
        model=model,
        optimizer=optimizer,
        loss=loss,
        train_loader=loader,
    )

    _assert_equal(model.state_dict(), source_model.state_dict())
    _assert_equal(optimizer.state_dict(), source_optimizer.state_dict())
    _assert_equal(loss.state_dict(), source_loss.state_dict())
    # The continuation adopts only the teacher's generator stream; its own order starts at a fresh epoch.
    target, source = _sampler_state(loader), _sampler_state(source_loader)
    assert target["generator_state"] == source["generator_state"]
    assert (target["epoch"], target["permutation"], target["cursor"]) == (-1, [], 0)
    assert target_adapter.value == 9
    assert restored["source_global_step"] == 1
    assert restored["restored_common_state"]["global_step"] == 1

    # Another runtime is refused, and so is another container: the handoff runs only where the teacher was produced, so
    # either is the training environment's error, which a rerun in the producing environment passes.
    for key, value in (("numpy", "different-producing-runtime"), ("sif_sha256", "a" * 64)):
        incompatible_runtime = {**identity["numerical_environment"], key: value}
        before = copy.deepcopy(model.state_dict())
        monkeypatch.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda runtime=incompatible_runtime: runtime)
        with pytest.raises(learning.training.checkpoint.ResumeEnvironmentError, match=f"differs in {key}"):
            learning.training.checkpoint.restore_handoff_checkpoint(
                payload,
                expected_source_identity=identity,
                model=model,
                optimizer=optimizer,
                loss=loss,
                train_loader=loader,
            )
        _assert_equal(before, model.state_dict())


def test_handoff_restore_rejects_an_evaluation_best_before_mutation() -> None:
    """A best without continuation state can never seed a B/A1 handoff."""
    identity = _identity("evaluation-handoff")
    source_model, source_optimizer, source_loss, source_loader = _components(50)
    payload = learning.training.checkpoint.make_checkpoint(
        role="best",
        payload_kind="evaluation",
        identity=identity,
        completed_epoch=1,
        global_step=1,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        loss=source_loss,
        best_metric=0.3,
        best_epoch=1,
        objective_history=[{"point": 1, "epoch": 1, "objective_id": "mse", "value": 0.3, "bound_violations": {}}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work=_WORK,
        update_work=_UPDATE_WORK,
        budget=_validated_budget(1),
        adapter=_Adapter(5),
    )
    assert set(payload).isdisjoint({"optimizer_state_dict", "python_rng_state", "train_sampler_state_dict", "next_epoch"})
    model, optimizer, loss, loader = _components(60)
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="payload_kind mismatch: expected 'full', got 'evaluation'"):
        learning.training.checkpoint.restore_handoff_checkpoint(
            payload,
            expected_source_identity=identity,
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_loader=loader,
        )
    _assert_equal(before, model.state_dict())
