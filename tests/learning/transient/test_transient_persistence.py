# ruff: noqa: NPY002, S101, PLR2004
"""Protect transient completed-history admission and v2 adapter rollback."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src import common, learning
from src.learning.transient import learning_transient_history as history


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
    """Build the complete checkpoint identity required by strict v2 tests."""

    def digest(name: str) -> str:
        return hashlib.sha256(f"{label}:{name}".encode()).hexdigest()

    return {
        "identity_contract_sha256": learning.training.checkpoint.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "transient_drying",
        "task_contract_digest": digest("task"),
        "effective_config_digest": digest("config"),
        "resume_contract_digest": digest("resume"),
        "dataset_ids": {"train": "synthetic-train", "ood": "synthetic-ood"},
        "dataset_fingerprints": {"train": digest("train"), "ood": digest("ood")},
        "split_contract_digest": digest("split"),
        "split_membership_digests": {
            "train": digest("train-membership"),
            "scaling_train_one_step": digest("scaling-membership"),
            "evaluation": digest("evaluation-membership"),
            "id_test": digest("id-test-membership"),
            "ood": digest("ood-membership"),
        },
        "normalizer_sha256": digest("normalizer"),
        "objective": {"id": "mse", "kind": "mse", "space": "normalized", "fields": ["w"], "reduction": "element_mean", "direction": "minimize"},
    }


def _history_payload(identity: dict[str, Any], *, epochs: int, task: str = "transient_drying", run_name: str = "run") -> dict[str, Any]:
    """Build one strict contiguous history payload."""
    return {
        "schema_version": 1,
        "task": task,
        "run_name": run_name,
        "checkpoint_identity_digest": common.serialization.canonical_json_sha256(identity),
        "epochs": [{"epoch": epoch, "loss": 1.0 / epoch} for epoch in range(1, epochs + 1)],
    }


def _write_history(tmp_path: Any, payload: dict[str, Any]) -> None:
    """Write one deliberately test-owned history file."""
    history.history_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("payload", "completed_epoch", "error"),
    [
        (None, 1, FileNotFoundError),
        ({"not": "history"}, 1, ValueError),
        ("wrong_identity", 1, ValueError),
        ("gapped", 2, ValueError),
        ("ahead", 1, ValueError),
    ],
)
def test_completed_history_admission_is_exact_and_read_only(
    tmp_path: Any,
    payload: dict[str, Any] | str | None,
    completed_epoch: int,
    error: type[Exception],
) -> None:
    """Reject absent, malformed, foreign, gapped, and ahead completed evidence without repair."""
    identity = _identity()
    if payload == "wrong_identity":
        payload = _history_payload(_identity("other"), epochs=1)
    elif payload == "gapped":
        payload = _history_payload(identity, epochs=2)
        payload["epochs"][1]["epoch"] = 3
    elif payload == "ahead":
        payload = _history_payload(identity, epochs=2)
    if payload is not None:
        assert isinstance(payload, dict)
        _write_history(tmp_path, payload)
    path = history.history_path(tmp_path)
    before = path.read_bytes() if path.exists() else None

    with pytest.raises(error):
        history.validate_completed_history(
            tmp_path,
            task="transient_drying",
            run_name="run",
            checkpoint_identity=identity,
            completed_epoch=completed_epoch,
        )

    assert (path.read_bytes() if path.exists() else None) == before


def test_reconcile_history_repairs_only_one_ahead_crash_record(tmp_path: Any) -> None:
    """Retain exact history, truncate one crash-ahead record, and fail closed otherwise."""
    identity = _identity()
    exact = _history_payload(identity, epochs=2)
    _write_history(tmp_path, exact)
    assert history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2) == exact

    ahead = _history_payload(identity, epochs=3)
    _write_history(tmp_path, ahead)
    reconciled = history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2)
    assert len(reconciled["epochs"]) == 2
    assert len(json.loads(history.history_path(tmp_path).read_text(encoding="utf-8"))["epochs"]) == 2

    _write_history(tmp_path, _history_payload(identity, epochs=4))
    with pytest.raises(ValueError, match="more than one"):
        history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2)
    _write_history(tmp_path, _history_payload(_identity("other"), epochs=2))
    with pytest.raises(ValueError, match="identity"):
        history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2)


def _components(seed: int) -> tuple[nn.Module, AdamW, _StatefulLoss, DataLoader[Any]]:
    """Build compact stateful CPU components for exact v2 restore."""
    torch.manual_seed(seed)
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters(), lr=0.01)
    loss = _StatefulLoss(float(seed))
    generator = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4, 1)), batch_size=2, shuffle=True, generator=generator)
    x, y = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)
    torch.mean((model(x) - y).square()).backward()
    optimizer.step()
    return model, optimizer, loss, loader


def _loader_generator_state(loader: DataLoader[Any]) -> torch.Tensor:
    """Return one explicitly configured loader generator state."""
    generator = loader.generator
    assert generator is not None
    return generator.get_state()


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
        identity=identity,
        completed_epoch=1,
        global_step=1,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        scaler=None,
        amp_enabled=False,
        loss=source_loss,
        best_metric=0.5,
        best_epoch=1,
        objective_history=[{"epoch": 1, "objective_id": "mse", "value": 0.5}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        adapter=source_adapter,
    )
    model, optimizer, loss, loader = _components(20)
    adapter = _Adapter(9)
    restored = learning.training.checkpoint.restore_checkpoint(
        payload,
        expected_identity=identity,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        amp_enabled=False,
        loss=loss,
        train_loader=loader,
        adapter=adapter,
        adapter_expected=True,
    )
    assert restored["global_step"] == 1
    assert adapter.value == source_adapter.value
    _assert_equal(model.state_dict(), source_model.state_dict())

    failing = copy.deepcopy(payload)
    failing["adapter_state_dict"] = {"value": 7, "fail": True}
    before = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "loss": copy.deepcopy(loss.state_dict()),
        "loader": _loader_generator_state(loader).clone(),
        "adapter": adapter.state_dict(),
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }
    with pytest.raises(RuntimeError, match="adapter restore failure"):
        learning.training.checkpoint.restore_checkpoint(
            failing,
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            amp_enabled=False,
            loss=loss,
            train_loader=loader,
            adapter=adapter,
            adapter_expected=True,
        )
    _assert_equal(before["model"], model.state_dict())
    _assert_equal(before["optimizer"], optimizer.state_dict())
    _assert_equal(before["loss"], loss.state_dict())
    assert torch.equal(before["loader"], _loader_generator_state(loader))
    _assert_equal(before["adapter"], adapter.state_dict())
    _assert_equal(before["python"], random.getstate())
    _assert_equal(before["numpy"], np.random.get_state())
    assert torch.equal(before["torch"], torch.get_rng_state())


def test_handoff_restore_preserves_complete_common_state_and_fresh_adapter() -> None:
    """Continue common optimization/RNG state while the target stage owns fresh adapter state."""
    identity = _identity("handoff")
    source_model, source_optimizer, source_loss, source_loader = _components(30)
    source_adapter = _Adapter(4)
    payload = learning.training.checkpoint.make_checkpoint(
        role="best",
        identity=identity,
        completed_epoch=2,
        global_step=5,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        scaler=None,
        amp_enabled=False,
        loss=source_loss,
        best_metric=0.2,
        best_epoch=2,
        objective_history=[{"epoch": 2, "objective_id": "mse", "value": 0.2}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        adapter=source_adapter,
    )
    model, optimizer, loss, loader = _components(40)
    target_adapter = _Adapter(9)

    restored = learning.training.checkpoint.restore_handoff_checkpoint(
        payload,
        expected_source_identity=identity,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        amp_enabled=False,
        loss=loss,
        train_loader=loader,
    )

    _assert_equal(model.state_dict(), source_model.state_dict())
    _assert_equal(optimizer.state_dict(), source_optimizer.state_dict())
    _assert_equal(loss.state_dict(), source_loss.state_dict())
    assert torch.equal(_loader_generator_state(loader), _loader_generator_state(source_loader))
    assert target_adapter.value == 9
    assert restored["source_global_step"] == 5
    assert restored["restored_common_state"]["global_step"] == 5
