"""Protect the epoch record a divergence writes right after a validation inside its epoch.

Responsibilities:
  - Check that the epoch record of a divergence in the first group after a validation inside the epoch holds the
    training values of the stretch that ended at the same step, in one attempt and after a resume at that
    validation's last checkpoint, and that the finished history validates
  - Check that an epoch end that committed updates after such a validation keeps its own training values

Design principles:
  - Real CPU training loops through the real history writer, reconciled to ``last`` on resume like run admission;
    the loss turns non-finite at a fixed update

This module does NOT:
  - Cover checkpoint cadence or history admission; the checkpoint-resume and history tests do
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from src import learning
from src.datasets.runtime.dataset_runtime_units import UnitSampler

if TYPE_CHECKING:
    from pathlib import Path

_CASES = 12
_OBJECTIVE = {"id": "mse", "kind": "mse", "space": "normalized", "fields": ["first", "second"], "reduction": "element_mean", "direction": "minimize"}


class _Cases(Dataset[dict[str, torch.Tensor]]):
    """Twelve fixed two-feature regression cases."""

    def __init__(self) -> None:
        self.x = torch.linspace(-1.0, 1.0, 2 * _CASES).reshape(_CASES, 2)
        self.y = self.x.flip(1)

    def __len__(self) -> int:
        return _CASES

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


class _ConstantMSE:
    """Report one constant normalized objective, so the first validation is eligible and the best."""

    id = "mse"
    space = "normalized"

    def reset(self) -> None:
        """Hold no accumulated state."""

    def update(self, pred: torch.Tensor, target: torch.Tensor, *, space: str, batch_index: int) -> None:
        """Accept every normalized batch without accumulating."""
        del pred, target, space, batch_index

    def compute(self) -> float:
        """Return the constant objective."""
        return 0.5

    def components(self) -> dict[str, float]:
        """Return no components."""
        return {}


class _LossDivergingAtUpdate(nn.Module):
    """Per-case MSE that turns non-finite from the given update on; every update here is one microbatch."""

    def __init__(self, diverging_update: int) -> None:
        super().__init__()
        self.diverging_update = diverging_update
        self.calls = 0

    def unit_components(self, pred: torch.Tensor, *, x: torch.Tensor, y: torch.Tensor, reference: object = None) -> dict[str, torch.Tensor]:
        """Return every case's squared error, multiplied by NaN from the divergent update on."""
        del x, reference
        self.calls += 1
        total = (pred - y).square().mean(dim=1)
        return {"total": total * math.nan if self.calls >= self.diverging_update else total}


def _identity() -> dict[str, Any]:
    """Build a complete checkpoint identity of one synthetic steady run."""
    digest = {label: hashlib.sha256(label.encode()).hexdigest() for label in ("task", "config", "split", "normalizer")}
    roles = ("train", "validation", "id_test", "material_ood", "parameter_ood")
    digest.update({role: hashlib.sha256(role.encode()).hexdigest() for role in roles})
    return {
        "identity_contract_sha256": learning.training.checkpoint.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "steady_flow",
        "task_contract_digest": digest["task"],
        "effective_config_digest": digest["config"],
        "dataset_ids": {"train": "train", "material_ood": "material-ood", "parameter_ood": "parameter-ood"},
        "dataset_fingerprints": {role: digest[role] for role in ("train", "material_ood", "parameter_ood")},
        "split_contract_digest": digest["split"],
        "split_membership_digests": {role: digest[role] for role in roles},
        "normalizer_sha256": digest["normalizer"],
        "numerical_environment": learning.training.checkpoint.numerical_environment_identity(),
        "objective": _OBJECTIVE,
    }


class _StopRunError(Exception):
    """Interrupt an attempt once its first event's checkpoints and history are durable."""


def _binding() -> dict[str, Any]:
    """Return the run identity the history binds."""
    return {"task": "steady_flow", "run_name": "run", "checkpoint_identity": _identity()}


def _train(run_dir: Path, loss: nn.Module, *, epochs: int = 2, resume: bool = False, event_callback: Any = None) -> dict[str, Any]:
    """Train three groups of four cases per epoch for an ``epochs`` budget through the real history writer, reconciled on resume."""
    binding = _binding()
    last_path = run_dir / "last_checkpoint.pt"
    counted: dict[str, Any] = {"completed_epoch": 0, "validations": [], "global_step": 0}
    if resume:
        saved = learning.training.checkpoint.load_checkpoint(
            last_path,
            expected_identity=binding["checkpoint_identity"],
            expected_role="last",
            payload_kind="full",
            scheduler_expected=False,
            require_best=False,
        )
        counted = {
            "completed_epoch": learning.training.checkpoint.recorded_history_epochs(saved),
            "validations": saved["objective_history"],
            "global_step": saved["global_step"],
        }
    initial = learning.training.history.reconcile_history(run_dir, **binding, **counted)
    model = nn.Linear(2, 2)
    return learning.training.loop.train_loop(
        config={
            "task": "steady_flow",
            "run": {"device": "cpu"},
            "training": {"budget_epochs": epochs},
            "evaluation": {"objective": _OBJECTIVE},
            "tracking": {"wandb": {"monitor": {"enabled": False, "max_cases": 1}}},
        },
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.05),
        train_loader=DataLoader(
            _Cases(), batch_sampler=UnitSampler(unit_kind="case", cases=range(_CASES), keys=None, units_per_update=4, memory_batch=4, seed=7)
        ),
        eval_loader=DataLoader(_Cases(), batch_size=4, shuffle=False),
        train_loss=loss,
        eval_metrics={"mse": _ConstantMSE()},
        save_dir=run_dir,
        resume_from=last_path if resume else None,
        event_callback=event_callback,
        epoch_state_callback=learning.training.history.make_epoch_state_callback(run_dir, **binding, initial_history=initial),
        checkpoint_identity=binding["checkpoint_identity"],
        budget=learning.training.budget.StageBudget(unit="units", total=learning.training.budget.passes_budget(epochs, _CASES)),
    )


def _stop(_step: int, _values: dict[str, float]) -> None:
    """Interrupt the attempt after its first event."""
    raise _StopRunError


@pytest.mark.parametrize("resume", [False, True], ids=("in-one-attempt", "after-a-resume-at-that-validation"))
def test_a_divergence_right_after_a_validation_inside_the_epoch_records_the_last_committed_training(tmp_path: Path, resume: bool) -> None:
    """
    Three groups of four cases make one epoch; the first reaches point 3 and validates, the next one diverges.

    With ``resume`` the first attempt stops after that validation published ``last`` and the resumed one diverges in
    its first group. Either way the stretch after the validation commits no group, so the epoch record repeats the
    training values of the stretch that ended at the same step, which the validation record holds, and the training
    curve keeps its loss.
    """
    if resume:
        with pytest.raises(_StopRunError):
            _train(tmp_path, _LossDivergingAtUpdate(2), event_callback=_stop)
    result = _train(tmp_path, _LossDivergingAtUpdate(1 if resume else 2), resume=resume)

    assert result["status"] == "diverged"
    last = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    recorded = learning.training.history.validate_completed_history(
        tmp_path,
        **_binding(),
        completed_epoch=learning.training.checkpoint.recorded_history_epochs(last),
        validations=last["objective_history"],
    )
    ((epoch_record,), (validation,)) = recorded["epochs"], recorded["validations"]
    assert validation["validation/point"] == 3.0
    assert epoch_record["global_step"] == validation["global_step"] == 1.0
    assert epoch_record["train/loss_total"] == validation["train/loss_total"]
    assert epoch_record["train/updates"] == validation["train/updates"] == 1.0


def test_an_epoch_end_that_trained_after_a_validation_inside_the_epoch_keeps_its_own_training(tmp_path: Path) -> None:
    """
    A twelve-epoch budget validates point 1 after the second group; the third group ends epoch 1 without a validation.

    That epoch end committed an update after the validation, so its record keeps its own stretch's training values.
    """

    def stop_at_the_first_epoch_end(step: int, _values: dict[str, float]) -> None:
        if step == 3:
            raise _StopRunError

    # The run stops after its third update, before the loss would diverge.
    with pytest.raises(_StopRunError):
        _train(tmp_path, _LossDivergingAtUpdate(4), epochs=12, event_callback=stop_at_the_first_epoch_end)

    persisted = json.loads(learning.training.history.history_path(tmp_path).read_text(encoding="utf-8"))
    ((epoch_record,), (validation,)) = persisted["epochs"], persisted["validations"]
    assert (validation["validation/point"], validation["global_step"], epoch_record["global_step"]) == (1.0, 2.0, 3.0)
    assert (validation["train/updates"], epoch_record["train/updates"]) == (2.0, 1.0)
    assert epoch_record["train/loss_total"] != validation["train/loss_total"]
