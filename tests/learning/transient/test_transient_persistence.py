"""Protect transient completed-history admission and adapter rollback.

Responsibilities:
  - Check completed history, ahead-of-checkpoint reconciliation, live non-finite metric classification, exact restore, and handoff state

Design principles:
  - Late restore failures roll back common and adapter state transactionally

This module does NOT:
  - Allocate stage runs or discard incompatible history
"""

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
    """Build the complete checkpoint identity required by strict tests."""

    def digest(name: str) -> str:
        return hashlib.sha256(f"{label}:{name}".encode()).hexdigest()

    return {
        "identity_contract_sha256": learning.training.checkpoint.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "transient_drying",
        "task_contract_digest": digest("task"),
        "effective_config_digest": digest("config"),
        "resume_contract_digest": digest("resume"),
        "dataset_ids": {"train": "synthetic-train", "additional_evaluation": "synthetic-additional-evaluation"},
        "dataset_fingerprints": {"train": digest("train"), "additional_evaluation": digest("additional-evaluation")},
        "split_contract_digest": digest("split"),
        "split_membership_digests": {
            "train": digest("train-membership"),
            "validation": digest("validation-membership"),
            "id_test": digest("id-test-membership"),
            "additional_evaluation": digest("additional-evaluation-membership"),
        },
        "normalizer_sha256": digest("normalizer"),
        "numerical_environment": learning.training.checkpoint.numerical_environment_identity(),
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


def test_reconcile_history_truncates_records_ahead_of_the_checkpoint_only(tmp_path: Any) -> None:
    """Retain exact history, truncate epochs trained after the restored last, and fail closed when history is behind."""
    identity = _identity()
    exact = _history_payload(identity, epochs=2)
    _write_history(tmp_path, exact)
    assert history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2) == exact

    _write_history(tmp_path, _history_payload(identity, epochs=4))
    reconciled = history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2)
    assert reconciled == exact
    assert json.loads(history.history_path(tmp_path).read_text(encoding="utf-8")) == exact

    _write_history(tmp_path, _history_payload(identity, epochs=1))
    with pytest.raises(ValueError, match="behind"):
        history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2)
    _write_history(tmp_path, _history_payload(_identity("other"), epochs=2))
    with pytest.raises(ValueError, match="identity"):
        history.reconcile_history(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=identity, completed_epoch=2)


@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")])
def test_live_callback_classifies_nonfinite_metric_as_numerical_failure_without_persisting(tmp_path: Any, nonfinite: float) -> None:
    """Raise FloatingPointError for a non-finite live metric while malformed values stay persistence errors."""
    write_epoch = history.make_epoch_state_callback(tmp_path, task="transient_drying", run_name="run", checkpoint_identity=_identity())
    write_epoch(1, {"loss": 0.5, "id/physical/w_gr_rmse": 0.1})

    with pytest.raises(FloatingPointError, match=r"id/physical/w_gr_rmse.*epoch 2"):
        write_epoch(2, {"loss": 0.4, "id/physical/w_gr_rmse": nonfinite})
    with pytest.raises(ValueError, match="finite numeric evidence"):
        write_epoch(2, {"loss": 0.4, "id/physical/w_gr_rmse": "not-a-number"})  # type: ignore[dict-item]

    assert len(json.loads(history.history_path(tmp_path).read_text(encoding="utf-8"))["epochs"]) == 1


def _components(seed: int) -> tuple[nn.Module, AdamW, _StatefulLoss, DataLoader[Any]]:
    """Build compact stateful CPU components for exact restore."""
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
        objective_history=[{"epoch": 1, "objective_id": "mse", "value": 0.5}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work={"optimizer_steps": 1, "gradient_predictions": 2},
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
            adapter=adapter,
            adapter_expected=True,
        )
    _assert_equal(before["model"], model.state_dict())
    _assert_equal(before["optimizer"], optimizer.state_dict())
    _assert_equal(before["loss"], loss.state_dict())
    assert torch.equal(before["loader"], _loader_generator_state(loader))
    _assert_equal(before["adapter"], adapter.state_dict())
    _assert_equal(before["python"], random.getstate())
    _assert_equal(before["numpy"], np.random.get_state())  # noqa: NPY002 -- checkpoint and seeding tests exercise the process-global RNG state
    assert torch.equal(before["torch"], torch.get_rng_state())


def test_handoff_restore_preserves_complete_common_state_and_fresh_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue common optimization/RNG state while the target stage owns fresh adapter state."""
    identity = _identity("handoff")
    source_model, source_optimizer, source_loss, source_loader = _components(30)
    source_adapter = _Adapter(4)
    payload = learning.training.checkpoint.make_checkpoint(
        role="best",
        payload_kind="full",
        identity=identity,
        completed_epoch=2,
        global_step=5,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        loss=source_loss,
        best_metric=0.2,
        best_epoch=2,
        objective_history=[{"epoch": 2, "objective_id": "mse", "value": 0.2}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work={"optimizer_steps": 1, "gradient_predictions": 2},
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

    incompatible_runtime = copy.deepcopy(identity["numerical_environment"])
    incompatible_runtime["numpy"] = "different-producing-runtime"
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: incompatible_runtime)
    with pytest.raises(ValueError, match="identity is incompatible"):
        learning.training.checkpoint.restore_handoff_checkpoint(
            payload,
            expected_source_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=None,
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
        global_step=2,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=None,
        loss=source_loss,
        best_metric=0.3,
        best_epoch=1,
        objective_history=[{"epoch": 1, "objective_id": "mse", "value": 0.3}],
        train_loader=source_loader,
        runtime_device=torch.device("cpu"),
        training_work={"optimizer_steps": 1, "gradient_predictions": 2},
        adapter=_Adapter(5),
    )
    assert set(payload).isdisjoint({"optimizer_state_dict", "python_rng_state", "train_loader_generator_state", "next_epoch"})
    model, optimizer, loss, loader = _components(60)
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="payload_kind mismatch: expected 'full', got 'evaluation'"):
        learning.training.checkpoint.restore_handoff_checkpoint(
            payload,
            expected_source_identity=identity,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss=loss,
            train_loader=loader,
        )
    _assert_equal(before, model.state_dict())
