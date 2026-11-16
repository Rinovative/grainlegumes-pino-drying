"""Protect the durable training history shared by both tasks.

Responsibilities:
  - Check completed-history admission and its binding to the checkpoint's objective history or a trial's points,
    ahead-of-checkpoint reconciliation of epoch and validation records, and live non-finite metric classification

Design principles:
  - Admission is exact and read-only; only reconciliation against a restored last checkpoint rewrites the file

This module does NOT:
  - Run training; the loop resume tests and the steady and transient lifecycle tests cover the producer wiring
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from src import common
from src.learning.training import learning_training_history as history

# Epoch ``e`` ends at optimizer step ``_EPOCH_STEPS * e``; validation point ``p`` of epoch 1 lies at step ``p``.
_EPOCH_STEPS = 100


def _identity(label: str = "history") -> dict[str, Any]:
    """Build a small test-owned checkpoint identity; the history binds only its canonical digest."""
    return {"task": "steady_flow", "effective_config_digest": hashlib.sha256(label.encode()).hexdigest()}


def _objective_history(points: tuple[int, ...]) -> list[dict[str, Any]]:
    """Build the checkpoint objective history of eligible validations in epoch 1, one per point."""
    return [{"point": point, "epoch": 1, "objective_id": "mse", "value": 0.1 * point, "bound_violations": {}} for point in points]


def _history_payload(identity: dict[str, Any], *, epochs: int, points: tuple[int, ...] = ()) -> dict[str, Any]:
    """Build one strict history payload: contiguous epoch records and one eligible record per validation point in epoch 1."""
    return {
        "task": "steady_flow",
        "run_name": "run",
        "checkpoint_identity_digest": common.serialization.canonical_json_sha256(identity),
        "epochs": [{"epoch": epoch, "global_step": _EPOCH_STEPS * epoch, "loss": 1.0 / epoch} for epoch in range(1, epochs + 1)],
        "validations": [
            {"epoch": 1, "global_step": point, "validation/point": point, "validation/eligible": 1.0, "id/mse": 0.1 * point} for point in points
        ],
    }


def _write_history(tmp_path: Any, payload: dict[str, Any]) -> None:
    """Write one deliberately test-owned history file."""
    history.history_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")


def _defective_payload(defect: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Return the completed history of one defect against a checkpoint that counts two epochs and the points 5 and 10."""
    payload = _history_payload(identity, epochs=2, points=(5, 10))
    validations, epochs = payload["validations"], payload["epochs"]
    if defect == "missing":
        return None
    if defect == "schema":
        return {"not": "history"}
    if defect == "wrong_identity":
        return _history_payload(_identity("other"), epochs=2, points=(5, 10))
    if defect == "gapped":
        epochs[1]["epoch"] = 3
    elif defect == "descending_points":
        validations.reverse()
    elif defect == "validation_after_the_running_epoch":
        # Two completed epochs make epoch 3 the running one; a validation can belong to no later epoch.
        validations[1].update({"epoch": 4, "global_step": 3 * _EPOCH_STEPS})
    elif defect == "ahead":
        return _history_payload(identity, epochs=2, points=(5, 10, 15))
    elif defect == "non_selected_value":
        # The selected point 5 keeps its value; only the other validation's objective is stale.
        validations[1]["id/mse"] = 0.75
    elif defect == "validation_epoch":
        # Ordered and inside epoch 2, but the checkpoint validated point 10 in epoch 1.
        validations[1].update({"epoch": 2, "global_step": 150})
    elif defect == "eligibility":
        validations[1]["validation/eligible"] = 0.0
    elif defect == "epoch_steps_not_increasing":
        epochs[1]["global_step"] = epochs[0]["global_step"]
    elif defect == "validation_step_after_its_epoch":
        validations[1]["global_step"] = _EPOCH_STEPS + 1
    elif defect == "validation_steps_not_increasing":
        validations[1]["global_step"] = validations[0]["global_step"]
    else:
        raise AssertionError(defect)
    return payload


@pytest.mark.parametrize(
    ("defect", "error", "match"),
    [
        ("missing", FileNotFoundError, "history.json"),
        ("schema", ValueError, "schema"),
        ("wrong_identity", ValueError, "identity"),
        ("gapped", ValueError, "epoch records"),
        ("descending_points", ValueError, "validation records"),
        ("validation_after_the_running_epoch", ValueError, "validation records"),
        ("ahead", ValueError, "counts"),
        ("non_selected_value", ValueError, r"record 2 .*id/mse"),
        ("validation_epoch", ValueError, r"record 2 .*epoch"),
        ("eligibility", ValueError, r"record 2 .*eligibility"),
        ("epoch_steps_not_increasing", ValueError, "epoch end steps"),
        ("validation_step_after_its_epoch", ValueError, "validation steps"),
        ("validation_steps_not_increasing", ValueError, "validation steps"),
    ],
)
def test_completed_history_admission_is_exact_bound_and_read_only(tmp_path: Any, defect: str, error: type[Exception], match: str) -> None:
    """Reject absent, malformed, foreign, unordered, misplaced, ahead, or checkpoint-disagreeing evidence without repair."""
    identity = _identity()
    payload = _defective_payload(defect, identity)
    if payload is not None:
        _write_history(tmp_path, payload)
    path = history.history_path(tmp_path)
    before = path.read_bytes() if path.exists() else None

    with pytest.raises(error, match=match):
        history.validate_completed_history(
            tmp_path, task="steady_flow", run_name="run", checkpoint_identity=identity, completed_epoch=2, validations=_objective_history((5, 10))
        )

    assert (path.read_bytes() if path.exists() else None) == before


def test_completed_history_binds_ineligible_points_and_a_trial_binds_its_points_only(tmp_path: Any) -> None:
    """An ineligible entry binds its eligibility without a value; a trial's bare budget points bind no value at all."""
    identity = _identity()
    binding: dict[str, Any] = {"task": "steady_flow", "run_name": "run", "checkpoint_identity": identity, "completed_epoch": 2}
    payload = _history_payload(identity, epochs=2, points=(5, 10))
    payload["validations"][1].update({"validation/eligible": 0.0, "id/mse": 0.75})
    _write_history(tmp_path, payload)
    entries = _objective_history((5, 10))
    entries[1]["value"] = None

    assert history.validate_completed_history(tmp_path, **binding, validations=entries)["validations"] == payload["validations"]
    assert history.validate_completed_history(tmp_path, **binding, validations=(5, 10))["validations"] == payload["validations"]
    with pytest.raises(ValueError, match=r"record 2 .*validation point"):
        history.validate_completed_history(tmp_path, **binding, validations=(5, 11))


def test_reconcile_history_truncates_records_ahead_of_the_checkpoint_only(tmp_path: Any) -> None:
    """Retain exact history, truncate records after the restored last, and fail closed when history is behind or disagrees."""
    identity = _identity()
    binding = {"task": "steady_flow", "run_name": "run", "checkpoint_identity": identity}
    entries = _objective_history((5, 10))
    exact = _history_payload(identity, epochs=2, points=(5, 10))
    _write_history(tmp_path, exact)
    assert history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries) == exact

    _write_history(tmp_path, _history_payload(identity, epochs=4, points=(5, 10, 15)))
    reconciled = history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)
    assert reconciled == exact
    assert json.loads(history.history_path(tmp_path).read_text(encoding="utf-8")) == exact

    for behind in (_history_payload(identity, epochs=1, points=(5, 10)), _history_payload(identity, epochs=2, points=(5,))):
        _write_history(tmp_path, behind)
        with pytest.raises(ValueError, match="behind"):
            history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)
    _write_history(tmp_path, _history_payload(identity, epochs=2, points=(5, 9)))
    with pytest.raises(ValueError, match="validation point"):
        history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)
    stale = _history_payload(identity, epochs=2, points=(5, 10))
    stale["validations"][0]["id/mse"] = 0.75
    _write_history(tmp_path, stale)
    with pytest.raises(ValueError, match="id/mse"):
        history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)
    _write_history(tmp_path, _history_payload(_identity("other"), epochs=2, points=(5, 10)))
    with pytest.raises(ValueError, match="identity"):
        history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)


def test_live_callback_admits_each_record_in_step_order(tmp_path: Any) -> None:
    """An epoch record must not end before the validation its own event recorded; nothing is persisted then."""
    write_event = history.make_epoch_state_callback(tmp_path, task="steady_flow", run_name="run", checkpoint_identity=_identity())
    write_event(1, {"global_step": 4.0, "loss": 0.5}, {"global_step": 4.0, "validation/point": 5.0, "id/mse": 0.1})

    with pytest.raises(ValueError, match="epoch end steps"):
        write_event(2, {"global_step": 7.0, "loss": 0.4}, {"global_step": 8.0, "validation/point": 10.0, "id/mse": 0.1})

    persisted = json.loads(history.history_path(tmp_path).read_text(encoding="utf-8"))
    assert (len(persisted["epochs"]), len(persisted["validations"])) == (1, 1)


@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")])
def test_live_callback_classifies_nonfinite_metric_as_numerical_failure_without_persisting(tmp_path: Any, nonfinite: float) -> None:
    """Raise FloatingPointError for a non-finite live metric while malformed values stay persistence errors."""
    write_event = history.make_epoch_state_callback(tmp_path, task="steady_flow", run_name="run", checkpoint_identity=_identity())
    write_event(1, {"global_step": 4.0, "loss": 0.5}, {"global_step": 4.0, "validation/point": 5.0, "id/physical/p_rmse": 0.1})

    with pytest.raises(FloatingPointError, match=r"id/physical/p_rmse.*epoch 2"):
        write_event(2, {"global_step": 8.0, "loss": 0.4}, {"global_step": 8.0, "validation/point": 10.0, "id/physical/p_rmse": nonfinite})
    with pytest.raises(ValueError, match="finite numeric evidence"):
        write_event(2, {"global_step": 8.0, "loss": 0.4}, {"global_step": 8.0, "validation/point": 10.0, "id/physical/p_rmse": "not-a-number"})  # type: ignore[dict-item]

    persisted = json.loads(history.history_path(tmp_path).read_text(encoding="utf-8"))
    assert (len(persisted["epochs"]), len(persisted["validations"])) == (1, 1)
