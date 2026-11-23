"""Protect the durable training history shared by both tasks.

Responsibilities:
  - Check completed-history admission and its binding to the checkpoint's objective history or a trial's points,
    the read-only resume admission and ahead-of-checkpoint reconciliation of epoch and validation records whose
    counted records end exactly at the checkpoint's global step, and live non-finite metric classification
  - Check the one guarded float conversion of read and live values, counts integral as written, and the refusal of
    a linked history file

Design principles:
  - Admission is exact and read-only; only reconciliation against a restored last checkpoint rewrites the file,
    and only to drop records after the epochs it counts

This module does NOT:
  - Run training; the loop resume tests and the steady and transient lifecycle tests cover the producer wiring
"""

from __future__ import annotations

import hashlib
import json
import sys
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


def _write_history(tmp_path: Any, payload: dict[str, Any] | str) -> None:
    """Write one deliberately test-owned history file, a payload or its raw JSON text."""
    history.history_path(tmp_path).write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def _defective_payload(defect: str, identity: dict[str, Any]) -> dict[str, Any] | str | None:
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
    elif defect == "fractional_epoch_step":
        epochs[1]["global_step"] = 2 * _EPOCH_STEPS + 0.5
    elif defect == "step_fractional_below_float_precision":
        # The literal rounds to the float 10.0, so only the decimal text shows the fraction.
        validations[1]["global_step"] = 10.25
        return json.dumps(payload).replace("10.25", "10.0000000000000001")
    elif defect == "integer_beyond_float_range":
        epochs[1]["loss"] = 10**400
    elif defect == "number_beyond_decimal_range":
        # The exponent exceeds the exact decimal parser's range, which raises decimal.InvalidOperation.
        epochs[1]["loss"] = 7.75
        return json.dumps(payload).replace("7.75", "1e999999999999999999999")
    elif defect == "epochs_not_a_list":
        payload["epochs"] = {"1": epochs[0]}
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
        ("fractional_epoch_step", ValueError, "integral"),
        ("step_fractional_below_float_precision", ValueError, "integral"),
        ("integer_beyond_float_range", ValueError, "finite numeric evidence"),
        ("number_beyond_decimal_range", ValueError, "representable numbers"),
        ("epochs_not_a_list", ValueError, "record lists"),
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


def test_a_finished_history_holds_no_validation_after_its_last_epoch(tmp_path: Any) -> None:
    """A finished run has no running epoch, so a validation after its last epoch record is refused, also for a trial's bare points."""
    identity = _identity()
    payload = _history_payload(identity, epochs=2, points=(5, 10))
    payload["validations"][1].update({"epoch": 3, "global_step": 2 * _EPOCH_STEPS + 10})
    _write_history(tmp_path, payload)

    with pytest.raises(ValueError, match="after its last completed epoch"):
        history.validate_completed_history(
            tmp_path, task="steady_flow", run_name="run", checkpoint_identity=identity, completed_epoch=2, validations=(5, 10)
        )


def test_reconcile_history_truncates_records_ahead_of_the_checkpoint_only(tmp_path: Any) -> None:
    """Retain exact history, truncate records after the counted epochs, and fail closed on a record before the last it lacks."""
    identity = _identity()
    # The last counts two epochs and ends at the second one's end step.
    binding = {"task": "steady_flow", "run_name": "run", "checkpoint_identity": identity, "global_step": 2 * _EPOCH_STEPS}
    entries = _objective_history((5, 10))
    exact = _history_payload(identity, epochs=2, points=(5, 10))
    _write_history(tmp_path, exact)
    assert history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries) == exact

    # Epochs 3 and 4 and a validation inside epoch 3 come after the two epochs the last counts: they are trained again.
    ahead = _history_payload(identity, epochs=4, points=(5, 10))
    ahead["validations"].append({"epoch": 3, "global_step": 250, "validation/point": 25, "validation/eligible": 1.0, "id/mse": 2.5})
    _write_history(tmp_path, ahead)
    before = history.history_path(tmp_path).read_bytes()
    # The resume admission keeps them; only reconciliation, the attempt's one writer, truncates them.
    assert history.validate_resume_history(tmp_path, **binding, completed_epoch=2, validations=entries) == ahead
    assert history.history_path(tmp_path).read_bytes() == before
    reconciled = history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)
    assert reconciled == exact
    assert json.loads(history.history_path(tmp_path).read_text(encoding="utf-8")) == exact
    # A last inside epoch 1 at point 10 counts no epoch record; an epoch end at its step 10 committed no update after it
    # (a divergence in the next group) and is trained again.
    diverged = _history_payload(identity, epochs=0, points=(5, 10))
    diverged["epochs"] = [{"epoch": 1, "global_step": 10, "loss": 1.0}]
    _write_history(tmp_path, diverged)
    inside = history.reconcile_history(tmp_path, **{**binding, "global_step": 10}, completed_epoch=0, validations=entries)
    assert inside == _history_payload(identity, epochs=0, points=(5, 10))
    # A fresh run counts no record, at step 0: what an attempt recorded before its first last is trained again.
    fresh = history.reconcile_history(tmp_path, **{**binding, "global_step": 0}, completed_epoch=0, validations=[])
    assert fresh == _history_payload(identity, epochs=0)

    # Point 15 lies inside epoch 1, before the last that counts two epochs, which has no entry for it: never truncated.
    _write_history(tmp_path, _history_payload(identity, epochs=4, points=(5, 10, 15)))
    before = history.history_path(tmp_path).read_bytes()
    with pytest.raises(ValueError, match="before the restored last checkpoint"):
        history.reconcile_history(tmp_path, **binding, completed_epoch=2, validations=entries)
    assert history.history_path(tmp_path).read_bytes() == before

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


@pytest.mark.parametrize(
    ("completed_epoch", "points", "global_step", "epochs"),
    [
        # The second epoch the last counts ends at step 200, after the last's step.
        (2, (5, 10), 150, 2),
        # A last inside epoch 1 at step 7 counts point 10, which the history places at step 10.
        (0, (5, 10), 7, 0),
        # A last that closed epoch 1 at step 101 counts an epoch record ending before it, at step 100.
        (1, (5, 10), 101, 1),
        # A last inside epoch 1 at point 10, step 10, counts only point 5 at step 5 and would truncate point 10.
        (0, (5,), 10, 0),
        # A last that counts no record claims three committed updates; only a fresh run counts none, at step 0.
        (0, (), 3, 0),
        # A last inside epoch 3 at its point 25, step 250, counts one epoch record, but the history ends epoch 2 before it.
        (1, (5, 10, 25), 250, 2),
    ],
    ids=(
        "counted-epoch-after",
        "counted-validation-after",
        "counted-epoch-before",
        "counted-validation-before",
        "nothing-counted-after-updates",
        "uncounted-epoch-before",
    ),
)
def test_reconcile_history_binds_records_to_the_checkpoint_step(
    tmp_path: Any, completed_epoch: int, points: tuple[int, ...], global_step: int, epochs: int
) -> None:
    """The newest record the last counts lies exactly at its global step and none it does not count before it; otherwise nothing is written."""
    identity = _identity()
    binding: dict[str, Any] = {"task": "steady_flow", "run_name": "run", "checkpoint_identity": identity}
    payload = _history_payload(identity, epochs=epochs, points=(5, 10))
    entries = _objective_history(points)
    if 25 in points:
        # Point 25 is validated inside epoch 3, after the history's second epoch end.
        payload["validations"].append({"epoch": 3, "global_step": 250, "validation/point": 25, "validation/eligible": 1.0, "id/mse": 2.5})
        entries[-1]["epoch"] = 3
    _write_history(tmp_path, payload)
    before = history.history_path(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="global step"):
        history.reconcile_history(tmp_path, **binding, completed_epoch=completed_epoch, validations=entries, global_step=global_step)
    assert history.history_path(tmp_path).read_bytes() == before


def test_read_and_live_values_share_one_guarded_float_conversion(tmp_path: Any) -> None:
    """An integer that rounds to a finite float is evidence; one beyond the float range is a ValueError when read or live."""
    identity = _identity()
    payload = _history_payload(identity, epochs=2, points=(5, 10))
    payload["epochs"][1]["loss"] = int(sys.float_info.max) + 1
    _write_history(tmp_path, payload)
    admitted = history.validate_completed_history(
        tmp_path, task="steady_flow", run_name="run", checkpoint_identity=identity, completed_epoch=2, validations=_objective_history((5, 10))
    )
    assert admitted["epochs"][1]["loss"] == sys.float_info.max

    live = tmp_path / "live"
    live.mkdir()
    write_event = history.make_epoch_state_callback(live, task="steady_flow", run_name="run", checkpoint_identity=identity)
    with pytest.raises(ValueError, match="finite numeric evidence"):
        write_event(1, {"global_step": 4.0, "loss": 10**400}, None)
    assert not history.history_path(live).exists()


def test_a_linked_history_file_is_refused(tmp_path: Any) -> None:
    """A history.json that is a symbolic link is no run-local evidence, whether it points at a valid history or nowhere."""
    identity = _identity()
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_history_payload(identity, epochs=2, points=(5, 10))), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history.history_path(run_dir).symlink_to(target)
    binding: dict[str, Any] = {"task": "steady_flow", "run_name": "run", "checkpoint_identity": identity}

    with pytest.raises(ValueError, match="link"):
        history.validate_completed_history(run_dir, **binding, completed_epoch=2, validations=_objective_history((5, 10)))
    with pytest.raises(ValueError, match="link"):
        history.reconcile_history(run_dir, **binding, completed_epoch=2, validations=_objective_history((5, 10)), global_step=2 * _EPOCH_STEPS)
    # A dangling link is refused too, also where a missing file would start an empty history.
    target.unlink()
    with pytest.raises(ValueError, match="link"):
        history.validate_completed_history(run_dir, **binding, completed_epoch=0, validations=[])
    with pytest.raises(ValueError, match="link"):
        history.reconcile_history(run_dir, **binding, completed_epoch=0, validations=[], global_step=0)


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
