"""Protect explicit matched branches, supervised-transition budgets and receipts, and exact cursor resume.

Responsibilities:
  - Check matched branches, immutable receipts, exact group cursors, caps, and recovery reporting
  - Check B's K x N and A1's receipt budgets, corrupted receipts, and the one GPU model of a B/A1 pair
  - Check that a receipt's validation schedule holds the point each count reached and B's checkpoint work at each step
  - Check that A1 cuts its last group to commit exactly B's transitions and validates at exactly the points B reached,
    also after a resume inside its stretch, and that only an A1 finishes with a replay or checkpoints a sampler stop
  - Check terminal group cursors, the history count of a divergence inside an epoch, and finalize-only terminal resume
  - Check that an unsuccessful run's epoch history holds exactly the records its last checkpoint counts, none without one
  - Check that a stretch without a committed step adds no point to the tracked step history
  - Check that only the matched A0 best keeps training state, and partial-epoch evaluation bests
  - Check that completed terminal controller and curriculum values are valid and equal the final history record

Design principles:
  - Committed compute and unconsumed samples determine restart boundaries

This module does NOT:
  - Choose scientific budget targets or run production matched experiments
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from support import configs
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, TensorDataset

from src import analysis, common, datasets
from src.experiments import experiments_matched_budget as budget
from src.experiments import experiments_run as run_service
from src.experiments import experiments_run_identity as identity
from src.experiments.cli import cli_train
from src.learning.training import learning_training_adapter as adapter_contract
from src.learning.training import learning_training_budget as stage_budget
from src.learning.training import learning_training_checkpoint as checkpoint_service
from src.learning.training import learning_training_history as epoch_history
from src.learning.training import learning_training_loop as loop
from src.learning.transient import learning_transient_adapter as transient_adapter
from src.learning.transient import learning_transient_curriculum as curriculum
from src.learning.transient.learning_transient_handoff import HANDOFF_DIRECTORY

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _train_loader(values: TensorDataset, *, seed: int, unit_kind: datasets.runtime.units.UnitKind = "trajectory", k: int = 1) -> DataLoader[Any]:
    """Return a loader of ``k``-unit updates (single-unit by default) driven by its unit sampler."""
    units = datasets.runtime.units
    sampler = units.UnitSampler(
        unit_kind=unit_kind, cases=range(len(values)), keys=None, units_per_update=k, memory_batch=1, seed=seed, full_set_transitions=len(values)
    )
    return DataLoader(values, batch_sampler=sampler)


class _Adapter:
    """Compute one squared-output unit loss and count the committed optimizer groups."""

    def __init__(self) -> None:
        self.groups = 0
        self.samples: list[int] = []

    def begin_group(self, group: datasets.runtime.units.UnitGroup, *, progress: float) -> None:
        del group
        assert 0.0 <= progress <= 1.0

    def unit_losses(
        self, model: torch.nn.Module, raw_batch: tuple[torch.Tensor], loss: torch.nn.Module, *, positions: tuple[int, ...]
    ) -> Iterator[adapter_contract.UnitLosses]:
        del loss
        batch = raw_batch[0]
        self.samples.append(int(batch[0, 0].item()))
        value = model(batch).square().flatten(1).mean(dim=1)
        yield adapter_contract.UnitLosses(components={"total": value}, supervised_transitions=len(positions), forward_transitions=len(positions))

    def record_optimizer_work(self, work: Any) -> None:
        assert work.microbatches > 0
        self.groups += 1


class _FailingAdapter(_Adapter):
    """Fail in the second microbatch of a group, after the first one went through forward and backward."""

    def __init__(self, error: type[Exception] = RuntimeError) -> None:
        super().__init__()
        self.error = error

    def unit_losses(
        self, model: torch.nn.Module, raw_batch: tuple[torch.Tensor], loss: torch.nn.Module, *, positions: tuple[int, ...]
    ) -> Iterator[adapter_contract.UnitLosses]:
        if self.samples:
            message = "synthetic failure in the second microbatch"
            raise self.error(message)
        yield from super().unit_losses(model, raw_batch, loss, positions=positions)


class _LoopAdapter(_Adapter):
    """Persist enough fake adapter state to exercise the real matched loop; the loop owns the stage budget."""

    def prepare_batch(self, raw: tuple[torch.Tensor], *, device: torch.device, training: bool) -> torch.Tensor:
        del training
        return raw[0].to(device)

    def state_dict(self) -> dict[str, Any]:
        return {"groups": self.groups}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.groups = state["groups"]

    def telemetry_state(self) -> dict[str, float]:
        return {"groups": float(self.groups)}

    def record_validation_work(self, seconds: float) -> None:
        del seconds

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        assert metric >= 0
        assert epoch_index >= 0

    def validate_terminal_state(self, *, best_metric: float, best_epoch: int) -> None:
        assert best_metric >= 0
        assert best_epoch >= 1

    def evaluation_step(self, model: torch.nn.Module, batch: torch.Tensor) -> adapter_contract.AdapterEvaluationViews:
        prediction = model(batch).reshape(-1, 1, 1, 1, 1)
        target = torch.zeros_like(prediction)
        return adapter_contract.AdapterEvaluationViews(
            normalized_prediction=prediction,
            normalized_target=target,
            physical_prediction=prediction,
            physical_target=target,
            one_step_normalized_prediction=prediction,
            one_step_physical_prediction=prediction,
            sample_count=len(batch),
            processed_target_transitions=len(batch),
            case_complete=torch.ones(len(batch), dtype=torch.bool),
            bound_violations={},
        )


class _WindowedLoopAdapter(_LoopAdapter):
    """Train each trajectory as two windows within its one update, as for RNO trajectories."""

    def unit_losses(
        self, model: torch.nn.Module, raw_batch: tuple[torch.Tensor], loss: torch.nn.Module, *, positions: tuple[int, ...]
    ) -> Iterator[adapter_contract.UnitLosses]:
        del loss
        for _window in range(2):
            value = 0.5 * model(raw_batch[0]).square().flatten(1).mean(dim=1)
            yield adapter_contract.UnitLosses(components={"total": value}, supervised_transitions=len(positions), forward_transitions=len(positions))


class _DivergingLoopAdapter(_LoopAdapter):
    """Diverge in the second update group, after the first one committed."""

    def unit_losses(
        self, model: torch.nn.Module, raw_batch: tuple[torch.Tensor], loss: torch.nn.Module, *, positions: tuple[int, ...]
    ) -> Iterator[adapter_contract.UnitLosses]:
        if self.samples:
            message = "synthetic non-finite loss"
            raise FloatingPointError(message)
        yield from super().unit_losses(model, raw_batch, loss, positions=positions)


class _ScaledMSELoss(torch.nn.MSELoss):
    """Accept the run's Train scaling binding without using it."""

    def set_scaling(self, scaling: object) -> None:
        del scaling


class _Metric:
    id = "mse"
    space = "physical"

    def reset(self) -> None:
        self.values: list[float] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, *, space: str, batch_index: int) -> None:
        del space, batch_index
        self.values.extend((pred - target).square().flatten().tolist())

    def compute(self) -> float:
        return sum(self.values) / len(self.values)

    def components(self) -> dict[str, float]:
        return {}


def _loop_identity() -> dict[str, Any]:
    def digest(name: str) -> str:
        return hashlib.sha256(name.encode()).hexdigest()

    objective = {"id": "mse", "kind": "mse", "space": "normalized", "fields": ["state"], "reduction": "element_mean", "direction": "minimize"}
    return {
        "identity_contract_sha256": checkpoint_service.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "transient_drying",
        "task_contract_digest": digest("task"),
        "effective_config_digest": digest("config"),
        "dataset_ids": {"train": "synthetic-train", "material_ood": "synthetic-material-ood", "parameter_ood": "synthetic-parameter-ood"},
        "dataset_fingerprints": {"train": digest("train"), "material_ood": digest("material-ood"), "parameter_ood": digest("parameter-ood")},
        "split_contract_digest": digest("split"),
        "split_membership_digests": {
            "train": digest("train-membership"),
            "validation": digest("validation-membership"),
            "id_test": digest("id-membership"),
            "material_ood": digest("material-ood-membership"),
            "parameter_ood": digest("parameter-ood-membership"),
        },
        "normalizer_sha256": digest("normalizer"),
        "numerical_environment": checkpoint_service.numerical_environment_identity(),
        "objective": objective,
    }


def test_cli_stage_parser_rejects_duplicates_and_preserves_canonical_order() -> None:
    assert cli_train._parse_stage_selection("a1,a0") == ("a0", "a1")
    assert cli_train._parse_stage_selection("all") is None
    with pytest.raises(ValueError, match="distinct"):
        cli_train._parse_stage_selection("a0,a0")
    with pytest.raises(ValueError, match="distinct"):
        cli_train._parse_stage_selection("a2")


def test_parent_identity_binds_the_b_budget(tmp_path: Path) -> None:
    """B's budget K x N follows the one K, A0's budget epochs, so another K is another parent."""
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    original = configs.resolve_plan(configs.transient_matched_config())
    first = identity.build_transient_experiment_record(*original.stages[:2], stage_a1=original.a1, config_path=config_path)
    revised = configs.transient_matched_config()
    revised["training"]["a0"]["budget_epochs"] += 1
    changed_budget = configs.resolve_plan(revised)
    third = identity.build_transient_experiment_record(*changed_budget.stages[:2], stage_a1=changed_budget.a1, config_path=config_path)
    assert third["parent_identity_sha256"] != first["parent_identity_sha256"]


# The full Train set's supervised transitions N of these synthetic B summaries, and the one GPU model of their pair.
_N = 5
_PAIR_GPU = {"device_type": "cuda", "cuda_device_name": "NVIDIA RTX 6000 Ada Generation"}
# A stand-in receipt of a B that trained on the CPU the synthetic plans run on, which A1's device check reads.
_CPU_RECEIPT = {"test": "receipt", "device": {"device_type": "cpu", "cuda_device_name": None}}


def _b_total() -> int:
    """Return the test plan's B budget K x N, K being A0's budget epochs."""
    return int(configs.resolve_plan(configs.transient_matched_config()).a0["training"]["budget_epochs"]) * _N


def _completed_b_summary(*, total: int, spent: int, outcome: str = "completed", eligible: bool = True) -> dict[str, Any]:
    """Return the receipt-relevant part of a finished B summary whose budget counts supervised transitions."""
    terminal_budget = stage_budget.StageBudget(unit="supervised_transitions", total=total)
    if spent:
        terminal_budget.record_group(units=3, supervised_transitions=spent, train_seconds=0.5)
        terminal_budget.record_validation()
    return {
        "runtime_sessions": [dict(_PAIR_GPU)],
        "training_units": {"full_set_transitions": _N},
        "status": "completed" if eligible else "unsuccessful",
        "training_outcome": {
            "status": outcome,
            "budget_progress": terminal_budget.progress,
            "eligible_points": int(eligible),
            "divergence": "loss is non-finite" if outcome == "diverged" else None,
        },
        "terminal_controller": {"post_handoff_optimizer_steps": 3 if spent else 0},
        "terminal_budget": terminal_budget.state_dict(),
        "validation_plan_deviation": {"skipped_points": terminal_budget.skipped_points},
        "terminal_epoch": 2 if spent else 0,
        "best_epoch": 1 if eligible else None,
        **({"best_checkpoint_sha256": "c" * 64} if eligible else {}),
        "last_checkpoint_sha256": "d" * 64 if spent else None,
    }


def _finished(summary: dict[str, Any], run_dir: Path | None = None, *, update_work: list[list[int]] | None = None) -> dict[str, Any]:
    """
    Return B as validate_finished_run admits it: run directory, summary, the terminal budget the one outcome rule admitted, last checkpoint.

    The last checkpoint's ``update_work`` defaults to one update of B's terminal spend.
    """
    target = ("supervised_transitions", summary["terminal_budget"]["total"], None)
    terminal_budget = run_service.validate_training_outcome(summary, lifecycle=summary["status"], target=target)
    rows = [[3, 1, terminal_budget.supervised_transitions]] if update_work is None else update_work
    return {"run_dir": run_dir, "summary": summary, "terminal_budget": terminal_budget, "last_checkpoint": {"update_work": rows}}


def _b_history(run_dir: Path, validations: list[list[int]]) -> Path:
    """Write the part of B's history its receipt reads: every validation's point, committed supervised transitions and step 1, 2, ..."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"validation/point": float(point), "budget/supervised_transitions": float(count), "global_step": float(step)}
        for step, (point, count) in enumerate(validations, start=1)
    ]
    epoch_history.history_path(run_dir).write_text(json.dumps({"validations": records}), encoding="utf-8")
    return run_dir


def _receipt(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Return the receipt of a hand-made B summary whose one validation lies at its terminal spend."""
    spent = summary["terminal_budget"]
    _b_history(run_dir, [[spent["validated_point"], spent["supervised_transitions"]]] if spent["supervised_transitions"] else [])
    b_config = configs.resolve_plan(configs.transient_matched_config()).b
    return budget.build_b_receipt(
        parent_identity_sha256="a" * 64, a0_handoff_sha256="b" * 64, b_config=b_config, finished_b=_finished(summary, run_dir)
    )


@pytest.mark.parametrize(
    ("outcome", "eligible", "spent"),
    [("diverged", True, 6), ("diverged", False, 6), ("no_eligible_checkpoint", False, 21), ("diverged", False, 0)],
    ids=("diverged-with-best", "diverged-without-best", "no-eligible-checkpoint", "diverged-in-first-group"),
)
def test_receipt_records_b_actual_transitions_and_end_status_for_every_end_of_b(tmp_path: Path, outcome: str, eligible: bool, spent: int) -> None:
    """B diverging at 30 % of its budget, in its first group, or ending without an eligible checkpoint still gives A1 its transitions."""
    receipt = _receipt(_completed_b_summary(total=20, spent=spent, outcome=outcome, eligible=eligible), tmp_path)

    assert (receipt["budget_unit"], receipt["supervised_transitions"]) == ("supervised_transitions", spent)
    assert (receipt["train_seconds"], receipt["device"]) == (0.5 if spent else 0.0, _PAIR_GPU)
    assert receipt["b_status"] == outcome
    assert (receipt["b_budget_total"], receipt["b_budget_progress"]) == (20, pytest.approx(min(1.0, spent / 20)))
    assert (receipt["b_eligible_points"], receipt["b_validations"]) == (int(eligible), [[min(20, spent), spent]] if spent else [])
    assert (receipt["best_checkpoint_sha256"] is not None) is eligible
    assert (receipt["last_checkpoint_sha256"] is not None) is (spent > 0)
    assert budget.validate_b_receipt(receipt) == receipt
    # A1's budget is exactly B's actual transitions, whatever its own Train set; zero is a budget complete from the start.
    plan = configs.resolve_plan(configs.transient_matched_config())
    a1_budget = run_service._stage_budget(plan.a1, training_units={"full_set_transitions": 999}, matched_receipt=receipt)
    assert (a1_budget.unit, a1_budget.total, a1_budget.replay) == ("supervised_transitions", spent, receipt["b_validations"])
    assert a1_budget.complete is (spent == 0)


@pytest.mark.parametrize(
    ("validations", "update_work"),
    [([[6, 6]], [[2, 1, 5], [1, 1, 1]]), ([[5, 5]], [[3, 1, 5]]), ([[6, 6], [6, 6]], [[3, 1, 6]]), ([[5.5, 6]], [[3, 1, 6]])],
    ids=("count-not-work-at-step", "transitions-not-work", "step-beyond-work", "non-integral-point"),
)
def test_a_receipt_count_must_be_bs_checkpoint_work_at_its_validation_step(
    tmp_path: Path, validations: list[list[int]], update_work: list[list[int]]
) -> None:
    """B diverged after 6 transitions; a count other than its checkpoint work at its step, or other total work, gives no receipt."""
    finished = _finished(_completed_b_summary(total=20, spent=6, outcome="diverged"), _b_history(tmp_path, validations), update_work=update_work)
    b_config = configs.resolve_plan(configs.transient_matched_config()).b

    with pytest.raises(ValueError, match="checkpoint work"):
        budget.build_b_receipt(parent_identity_sha256="a" * 64, a0_handoff_sha256="b" * 64, b_config=b_config, finished_b=finished)


def test_only_an_a1_finishes_with_a_replay_and_only_with_its_receipts_schedule(tmp_path: Path) -> None:
    """
    The one outcome rule compares the terminal budget's unit, total and replay with the stage's target.

    A finished A1 replays exactly its leaf receipt's schedule, so another schedule or none is refused; every other
    stage replays nothing, so a finished B whose terminal budget carries a replay is refused.
    """
    receipt = _receipt(_completed_b_summary(total=20, spent=6, outcome="diverged"), tmp_path / "b")
    a1_dir = tmp_path / "a1"
    budget.publish_b_receipt(_b_history(a1_dir, []), receipt)
    plan = configs.resolve_plan(configs.transient_matched_config())
    a1_budget = run_service._stage_budget(plan.a1, training_units={}, matched_receipt=receipt)
    a1_budget.record_group(units=3, supervised_transitions=6, train_seconds=0.5)
    a1_budget.record_validation()
    summary = {
        "terminal_controller": {"matched_receipt_sha256": receipt["receipt_sha256"]},
        "terminal_budget": a1_budget.state_dict(),
        "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
        "validation_plan_deviation": {"skipped_points": a1_budget.skipped_points},
    }

    target = run_service._finished_stage_target(a1_dir, plan.a1, summary)
    assert target == ("supervised_transitions", 6, [[6, 6]])
    assert run_service.validate_training_outcome(summary, lifecycle="completed", target=target).replay == [[6, 6]]
    for replay in ([[1, 1]], None):
        other = stage_budget.StageBudget(unit="supervised_transitions", total=6, replay=replay).state_dict()
        with pytest.raises(run_service.RunLifecycleError, match="differs"):
            run_service.validate_training_outcome({**summary, "terminal_budget": other}, lifecycle="completed", target=target)
    b_summary = _completed_b_summary(total=_b_total(), spent=_b_total())
    b_summary["terminal_budget"]["replay"] = [[20, _b_total()]]
    with pytest.raises(run_service.RunLifecycleError, match="differs"):
        run_service.validate_training_outcome(
            b_summary, lifecycle="completed", target=run_service._finished_stage_target(tmp_path, plan.b, b_summary)
        )


def _redigested(receipt: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Return a receipt with changed fields and a digest recomputed over them, so only the field rules can refuse it."""
    payload = {key: value for key, value in {**receipt, **changes}.items() if key != "receipt_sha256"}
    return {**payload, "receipt_sha256": common.serialization.canonical_json_sha256(payload)}


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (lambda receipt: {**receipt, "supervised_transitions": receipt["supervised_transitions"] + 1}, "digest mismatch"),
        (lambda receipt: _redigested(receipt, budget_unit="train_seconds"), "budget unit"),
        (lambda receipt: _redigested({key: value for key, value in receipt.items() if key != "budget_unit"}), "budget unit"),
        (lambda receipt: _redigested(receipt, supervised_transitions=21.0), "supervised_transitions"),
        (lambda receipt: _redigested(receipt, supervised_transitions=0), "zero for a B that diverged"),
        (lambda receipt: _redigested(receipt, train_seconds=float("nan")), "train_seconds"),
        (lambda receipt: _redigested(receipt, device={"device_type": "cuda", "cuda_device_name": ""}), "Device evidence"),
        (lambda receipt: _redigested(receipt, device={**_PAIR_GPU, "cuda_index": 0}), "device must be exactly"),
        (lambda receipt: _redigested(receipt, b_status=None), "end status"),
        (lambda receipt: _redigested({key: value for key, value in receipt.items() if key != "optimizer_steps"}), "keys"),
        (lambda receipt: _redigested(receipt, dataset_identity={"train_dataset": "d13"}), "dataset_identity"),
        (lambda receipt: _redigested(receipt, b_budget_progress=0.0), "progress"),
        (lambda receipt: _redigested(receipt, b_budget_total=22), "progress"),
        (lambda receipt: _redigested(receipt, b_eligible_points=2), "validation schedule"),
        (lambda receipt: _redigested(receipt, b_validations=[[20, 20]]), "validation schedule"),
        (lambda receipt: _redigested(receipt, b_validations=[[20, 21, 0]]), "malformed"),
        (lambda receipt: _redigested(receipt, b_validations=[[20, 22]]), "malformed"),
        (lambda receipt: _redigested(receipt, b_validations=[[1, 10], [20, 21]]), "contradicts"),
        (
            lambda receipt: _redigested(
                receipt,
                b_status="diverged",
                supervised_transitions=0,
                b_budget_progress=0.3,
                b_eligible_points=0,
                optimizer_steps=0,
                best_epoch=None,
                best_checkpoint_sha256=None,
                last_checkpoint_sha256=None,
            ),
            "progress",
        ),
        (
            lambda receipt: _redigested(
                receipt,
                b_status="diverged",
                supervised_transitions=0,
                b_budget_progress=0.0,
                b_eligible_points=1,
                optimizer_steps=0,
                best_epoch=1,
                best_checkpoint_sha256="a" * 64,
                last_checkpoint_sha256=None,
            ),
            "committed no group",
        ),
    ],
    ids=(
        "altered-count",
        "seconds-unit",
        "no-unit",
        "fractional-count",
        "zero-without-divergence",
        "nan-seconds",
        "gpu-without-model",
        "extra-device-field",
        "no-b-status",
        "missing-field",
        "malformed-field",
        "completed-without-progress",
        "completed-below-b-budget",
        "eligible-beyond-schedule",
        "schedule-before-end",
        "schedule-shape",
        "schedule-count-beyond-transitions",
        "schedule-point-not-reached-at-its-count",
        "zero-transitions-with-progress",
        "zero-transitions-with-best-checkpoint",
    ),
)
def test_a_corrupted_receipt_is_rejected_before_any_a1_budget(
    tmp_path: Path, corrupt: Callable[[dict[str, Any]], dict[str, Any]], match: str
) -> None:
    """An altered, seconds-era or malformed receipt never becomes A1's budget; the digest binds every field."""
    receipt = corrupt(_receipt(_completed_b_summary(total=20, spent=21), tmp_path))

    with pytest.raises(ValueError, match=match):
        budget.validate_b_receipt(receipt)


def test_a_b_a1_pair_trains_on_one_gpu_model_while_a0_may_use_any(tmp_path: Path) -> None:
    """
    A1 runs on its receipt's GPU model and a resumed B on its first attempt's; a fresh B and every A0 bind none.

    The receipt names one model only: a B whose attempts ran on several is refused before its receipt exists.
    """
    other_gpu = {"device_type": "cuda", "cuda_device_name": "NVIDIA RTX A6000"}

    def on(device: dict[str, Any]) -> Any:
        return SimpleNamespace(as_dict=lambda: {**device, "cuda_index": 1})

    plan = configs.resolve_plan(configs.transient_matched_config())
    receipt = _receipt(_completed_b_summary(total=20, spent=21), tmp_path)
    run_service._require_pair_device(plan.a1, on(_PAIR_GPU), sessions=[], matched_receipt=receipt)
    run_service._require_pair_device(plan.b, on(other_gpu), sessions=[], matched_receipt=None)
    run_service._require_pair_device(plan.b, on(_PAIR_GPU), sessions=[{**_PAIR_GPU, "cuda_index": 0}], matched_receipt=None)
    run_service._require_pair_device(plan.a0, on(other_gpu), sessions=[dict(_PAIR_GPU)], matched_receipt=None)
    for config, sessions, matched in ((plan.a1, [], receipt), (plan.b, [dict(_PAIR_GPU)], None)):
        with pytest.raises(run_service.RunLifecycleError, match="one GPU model"):
            run_service._require_pair_device(config, on(other_gpu), sessions=sessions, matched_receipt=matched)
    with pytest.raises(run_service.RunLifecycleError, match="receipt"):
        run_service._require_pair_device(plan.a1, on(_PAIR_GPU), sessions=[], matched_receipt=None)
    with pytest.raises(ValueError, match="several devices"):
        budget.pair_device([dict(_PAIR_GPU), other_gpu])


def test_an_a1_refused_on_another_gpu_model_publishes_no_b_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A0 and B finished without B's receipt; an A1 on another GPU model is refused before that receipt is published."""
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    stages: dict[str, dict[str, Any]] = {}
    published: list[Path] = []

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = config["training"]["comparison_arm"]
        stages[name] = config
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        run_service.config_loader.save_yaml(config, common.paths.resolve_run_config_path(run_dir))
        if name == "a0":
            manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{}", encoding="utf-8")
        result = {"status": "completed", "best_epoch": 1, "best_metric": 0.5}
        return {"run_dir": run_dir, "result": result, "device_resolution": kwargs["device_resolution"]}

    receipt = {"test": "receipt", "device": {"device_type": "cuda", "cuda_device_name": "NVIDIA RTX A6000"}}
    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_service, "validate_finished_run", lambda _run_dir: {"config": stages["b"], "summary": {}})
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: dict(receipt))
    monkeypatch.setattr(budget, "publish_b_receipt", lambda run_dir, _receipt: published.append(run_dir))
    monkeypatch.setattr(budget, "load_b_receipt", lambda *_args, **_kwargs: dict(receipt))
    assert cli_train.main([str(config_path), "--train-stages", "a0,b"]) == 0
    published.clear()  # B ended without its receipt, as after an interruption between B's end and its publication.

    assert cli_train.main([str(config_path), "--train-stages", "a1"]) == 1
    assert (published, list(stages)) == ([], ["a0", "b"])


def test_a_resumed_b_on_another_gpu_model_is_refused_before_its_writer_lock_exists() -> None:
    """A resumed B is checked against its first attempt's GPU model by a read before the writer lease creates its lock file."""
    plan = configs.resolve_plan(configs.transient_matched_config())
    run_dir = run_service.prepare_fresh_run(plan.b)
    lock = run_service._run_writer_lock_path(run_dir)
    lock.unlink()
    attempt = dict.fromkeys(run_service._ATTEMPT_KEYS)
    summary = {**run_service.read_run_summary(run_dir), "status": "interrupted", "runtime_sessions": [{**_PAIR_GPU, "attempt": attempt}]}
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)

    with pytest.raises(run_service.RunLifecycleError, match="one GPU model"):
        run_service._run_resolved_experiment(dict(plan.b), config_path="b.yaml", resume=run_dir)
    assert not lock.exists()


def test_one_outcome_rule_admits_an_unspent_budget_only_for_a_divergence() -> None:
    """The training outcome rule every finished run passes: complete budget unless diverged, progress as recorded."""
    with pytest.raises(run_service.RunLifecycleError, match="incomplete terminal budget"):
        _finished(_completed_b_summary(total=20, spent=6))
    with pytest.raises(run_service.RunLifecycleError, match="not allowed"):
        _finished(_completed_b_summary(total=20, spent=21, outcome="no_eligible_checkpoint", eligible=True))
    lying = _completed_b_summary(total=20, spent=6, outcome="diverged")
    lying["training_outcome"]["budget_progress"] = 1.0
    with pytest.raises(run_service.RunLifecycleError, match="progress"):
        _finished(lying)


def _finished_b_leaf(summary: dict[str, Any]) -> Path:
    """Return a prepared B leaf, at its derived destination, whose summary records the given finished training end."""
    run_dir = run_service.prepare_fresh_run(configs.resolve_plan(configs.transient_matched_config()).b)
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), {**run_service.read_run_summary(run_dir), **summary})
    return run_dir


def _no_group_evidence(stage: str) -> dict[str, Any]:
    """Return the terminal evidence the run writes for a stage that ended before its first committed group."""
    adapter: Any = object.__new__(transient_adapter.TransientTrainingAdapter)
    if stage == "b":
        handoff = curriculum.TeacherHandoffIdentity(
            source_run_name="a0",
            source_checkpoint_sha256="e" * 64,
            source_scaling_sha256="c" * 64,
            task_contract_sha256="d" * 64,
            tensorizer_sha256="f" * 64,
            model_kind="fno",
            input_profile="canonical_physics_complete",
        )
        controller = curriculum.MatchedComputeController(arm="B", stage="stage_b_self_fed", config_digest="a" * 64, teacher_handoff=handoff)
        rollout = curriculum.RolloutCurriculum(lengths=(2,), milestone_fractions=(0.0,))
        adapter.curriculum_state = curriculum.RolloutCurriculumState.create(rollout, seed=0)
    else:
        controller = curriculum.MatchedComputeController(arm="A0", stage="stage_a_teacher_forcing", config_digest="a" * 64)
        adapter.curriculum_state = None
    adapter.spec = SimpleNamespace(controller=controller)
    return run_service._terminal_adapter_evidence(adapter)


@pytest.mark.parametrize("total", ["configured", "zero", "other-n"])
def test_a_finished_b_is_admitted_only_with_its_configured_budget_total(total: str) -> None:
    """B diverged in its first group: its terminal total is K x N of its own Train set; zero or another N is refused."""
    totals = {"configured": _b_total(), "zero": 0, "other-n": _b_total() + 1}
    summary = _completed_b_summary(total=totals[total], spent=0, outcome="diverged", eligible=False)
    run_dir = _finished_b_leaf({**summary, **_no_group_evidence("b")})

    if total == "configured":
        assert run_service.validate_finished_run(run_dir)["terminal_budget"].total == _b_total()
    else:
        with pytest.raises(run_service.RunLifecycleError, match="configured"):
            run_service.validate_finished_run(run_dir)


def test_an_unsuccessful_run_binds_its_last_checkpoint_and_its_budget_to_its_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    B diverged at 30 %: its last checkpoint is loaded like a completed run's.

    A budget other than the summary's, a checkpoint without the divergence the summary claims,
    one whose history holds an eligible point the summary does not, or one whose terminal
    controller (which binds a matched stage's receipt) differs from the summary's, is refused,
    and so is a history without the partial epoch's record.
    """
    total = _b_total()
    controller = dict.fromkeys(run_service._TERMINAL_CONTROLLER_KEYS) | {"arm": "B", "post_handoff_optimizer_steps": 3}
    curriculum = dict.fromkeys(run_service._TERMINAL_CURRICULUM_KEYS, 1)
    summary = {
        **_completed_b_summary(total=total, spent=6, outcome="diverged", eligible=False),
        "terminal_controller": controller,
        "terminal_curriculum": curriculum,
    }
    run_dir = _finished_b_leaf(summary)
    last = common.paths.resolve_last_checkpoint_file(run_dir)
    last.write_bytes(b"last checkpoint")
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {**run_service.read_run_summary(run_dir), "last_checkpoint_sha256": common.serialization.file_sha256(last)},
    )
    history = [{"point": 1, "epoch": 1, "objective_id": "objective", "value": None, "bound_violations": {}}]
    saved_identity = {"effective_config_digest": "e" * 64}
    admitted = {
        "identity": saved_identity,
        "completed_epoch": 1,
        "train_sampler_state_dict": {"cursor": 1, "size": 3},
        "budget_state_dict": dict(summary["terminal_budget"]),
        "divergence": "loss is non-finite",
        "objective_history": history,
        "adapter_state_dict": {"controller": dict(controller), "curriculum": dict(curriculum)},
    }
    # The divergence inside epoch 1 wrote that partial epoch's records, its validation point's too, before its terminal last.
    _write_history(run_dir, saved_identity, epochs=1, validations=history)
    saved = [admitted]
    loads: list[dict[str, Any]] = []
    monkeypatch.setattr(run_service, "_saved_run_identity", lambda *_args, **_kwargs: ({}, None, saved_identity))

    def load_last(path: Path, **kwargs: Any) -> dict[str, Any]:
        loads.append({"path": path, **kwargs})
        return saved[0]

    monkeypatch.setattr(run_service.learning.training.checkpoint, "load_checkpoint", load_last)

    finished = run_service.validate_finished_run(run_dir)
    # B's receipt reads the admitted last checkpoint's committed work, also for a B that ended unsuccessful.
    assert (finished["terminal_budget"].supervised_transitions, finished["last_checkpoint"]) == (6, admitted)
    assert (loads[0]["path"], loads[0]["expected_role"], loads[0]["expected_identity"]) == (last, "last", {"effective_config_digest": "e" * 64})
    for changed, match in (
        ({"budget_state_dict": {**admitted["budget_state_dict"], "supervised_transitions": 5}}, "disagrees with its last checkpoint"),
        ({"divergence": None}, "divergence"),
        ({"objective_history": [{**history[0], "value": 0.5}]}, "eligible"),
        ({"adapter_state_dict": {"controller": {**controller, "matched_receipt_sha256": "f" * 64}, "curriculum": curriculum}}, "terminal controller"),
    ):
        saved[0] = {**admitted, **changed}
        with pytest.raises(run_service.RunLifecycleError, match=match):
            run_service.validate_finished_run(run_dir)
    # Without the partial epoch's record, or without its validation record, the history disagrees with what its last counts.
    saved[0] = admitted
    for epochs, recorded in ((0, history), (1, [])):
        _write_history(run_dir, saved_identity, epochs=epochs, validations=recorded)
        with pytest.raises(run_service.RunLifecycleError, match="history"):
            run_service.validate_finished_run(run_dir)


def _write_history(run_dir: Path, checkpoint_identity: dict[str, Any], *, epochs: int, validations: list[dict[str, Any]] | None = None) -> None:
    """Write a B leaf's history through the producer: reset to no record, then one record per objective-history entry and per epoch."""
    binding = {"task": "transient_drying", "run_name": str(configs.resolve_plan(configs.transient_matched_config()).b["run"]["name"])}
    initial = epoch_history.reconcile_history(
        run_dir, **binding, checkpoint_identity=checkpoint_identity, completed_epoch=0, validations=[], global_step=0
    )
    write = epoch_history.make_epoch_state_callback(run_dir, **binding, checkpoint_identity=checkpoint_identity, initial_history=initial)
    entries = validations or []
    for step, entry in enumerate(entries, start=1):
        value = {} if entry["value"] is None else {f"id/{entry['objective_id']}": entry["value"]}
        record = {"global_step": float(step), "validation/point": float(entry["point"]), "validation/eligible": float(bool(value))}
        write(entry["epoch"], None, {**record, **value})
    for epoch in range(1, epochs + 1):
        write(epoch, {"global_step": float(len(entries) + epoch), "train/loss_total": 0.5}, None)


@pytest.mark.parametrize("records", [None, 0, 1], ids=("no-history", "empty-history", "one-record"))
def test_an_unsuccessful_run_without_a_checkpoint_holds_no_history_record(monkeypatch: pytest.MonkeyPatch, records: int | None) -> None:
    """
    B diverged in its first group, so it wrote no history record and no checkpoint.

    Its history is absent, or empty after a restart reset a record of an attempt that crashed before its
    first checkpoint; a record is refused.
    """
    run_dir = _finished_b_leaf({**_completed_b_summary(total=_b_total(), spent=0, outcome="diverged", eligible=False), **_no_group_evidence("b")})
    identity = {"effective_config_digest": "e" * 64}
    monkeypatch.setattr(run_service, "_saved_run_identity", lambda *_args, **_kwargs: ({}, None, identity))
    if records is not None:
        _write_history(run_dir, identity, epochs=records)

    if records:
        with pytest.raises(run_service.RunLifecycleError, match="history"):
            run_service.validate_finished_run(run_dir)
    else:
        assert run_service.validate_finished_run(run_dir)["terminal_budget"].spent == 0


def test_budget_receipt_is_immutable_and_binds_exact_a0_origin(tmp_path: Path) -> None:
    b_config = configs.resolve_plan(configs.transient_matched_config()).b
    digest_a = "a" * 64
    digest_b = "b" * 64
    summary = _completed_b_summary(total=20, spent=21)
    finished = _finished(summary, _b_history(tmp_path / "b", [[20, 21]]))
    receipt = budget.build_b_receipt(parent_identity_sha256=digest_a, a0_handoff_sha256=digest_b, b_config=b_config, finished_b=finished)
    assert (receipt["protocol"], receipt["budget_unit"], receipt["supervised_transitions"]) == ("matched_a1_b", "supervised_transitions", 21)
    path = budget.publish_b_receipt(tmp_path, receipt)
    assert budget.publish_b_receipt(tmp_path, receipt) == path
    assert budget.load_b_receipt(path, parent_identity_sha256=digest_a, a0_handoff_sha256=digest_b, b_run_name=b_config["run"]["name"]) == receipt
    with pytest.raises(ValueError, match="selected A0/B"):
        budget.load_b_receipt(path, parent_identity_sha256="e" * 64, a0_handoff_sha256=digest_b, b_run_name=b_config["run"]["name"])
    with pytest.raises(FileExistsError, match="differs"):
        budget.publish_b_receipt(tmp_path, _redigested(receipt, train_seconds=0.75))


def test_each_stage_starts_its_own_budget_a0_in_full_set_units_b_in_k_times_n_a1_in_the_receipt_transitions() -> None:
    """
    A0 converts its K budget epochs at the full Train set's units, B the same K at its N transitions, A1 takes B's receipt.

    With unequal trajectory lengths N is their sum, not units x rollout: 7 units of a Train set of 23
    transitions give A0 K x 7 units and B K x 23 transitions.
    """
    plan = configs.resolve_plan(configs.transient_matched_config())
    k = plan.a0["training"]["budget_epochs"]
    units = {"full_set_units": 7, "full_set_transitions": 23}
    a0 = run_service._stage_budget(plan.a0, training_units=units, matched_receipt=None)
    b = run_service._stage_budget(plan.b, training_units=units, matched_receipt=None)
    a1 = run_service._stage_budget(plan.a1, training_units=units, matched_receipt={"supervised_transitions": 31, "b_validations": [[20, 31]]})

    assert (a0.unit, a0.total) == ("units", k * 7)
    assert (b.unit, b.total) == ("supervised_transitions", k * 23)
    assert (a1.unit, a1.total, a1.replay) == ("supervised_transitions", 31, [[20, 31]])
    with pytest.raises(run_service.RunLifecycleError, match="receipt"):
        run_service._stage_budget(plan.a1, training_units=units, matched_receipt=None)
    with pytest.raises(run_service.RunLifecycleError, match="full Train set count"):
        run_service._stage_budget(plan.b, training_units={"full_set_units": 7}, matched_receipt=None)


@pytest.mark.parametrize(("a1_status", "a1_best", "expected_exit"), [("no_eligible_checkpoint", None, 2), ("diverged", 1, 0)])
def test_the_exit_code_follows_the_terminal_stage_of_a_matched_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, a1_status: str, a1_best: int | None, expected_exit: int
) -> None:
    """A final A1 without an eligible checkpoint exits 2 even without artifacts; a diverged A1 with a result exits 0."""
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    stages: dict[str, dict[str, Any]] = {}

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = config["training"]["comparison_arm"]
        stages[name] = config
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        if name == "a0":
            manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{}", encoding="utf-8")
        status, best = (a1_status, a1_best) if name == "a1" else ("completed", 1)
        outcome = {"status": status, "budget_progress": 1.0, "eligible_points": int(best is not None), "divergence": None}
        result = {"status": status, "best_epoch": best, "best_metric": None if best is None else 0.5, "training_outcome": outcome}
        return {"run_dir": run_dir, "result": result, "device_resolution": kwargs["device_resolution"]}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_service, "validate_completed_run", lambda _run_dir: {"config": stages["b"], "summary": {"train_seconds": 1.0}})
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: dict(_CPU_RECEIPT))
    monkeypatch.setattr(budget, "publish_b_receipt", lambda *_args: None)
    monkeypatch.setattr(budget, "load_b_receipt", lambda *_args, **_kwargs: {"test": "receipt"})

    assert cli_train.main([str(config_path)]) == expected_exit


def test_a_diverged_a0_with_an_eligible_checkpoint_is_the_branch_point_of_b_and_a1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A0 diverged at 40 % of its budget with its best eligible checkpoint: it is admitted, and B and A1 start from its handoff.

    The status line names A1, the last stage; A0's stage line keeps its divergence visible.
    """
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    stages: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    handoffs: list[str] = []

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = config["training"]["comparison_arm"]
        stages[name] = config
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        if name == "a0":
            manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{}", encoding="utf-8")
        divergence = "non-finite loss" if name == "a0" else None
        outcomes[name] = {"status": "diverged" if divergence else "completed", "budget_progress": 0.4 if divergence else 1.0, "eligible_points": 1}
        outcome = {**outcomes[name], "divergence": divergence}
        result = {"status": outcome["status"], "best_epoch": 1, "best_metric": 0.5, "training_outcome": outcome}
        return {"run_dir": run_dir, "result": result, "device_resolution": kwargs["device_resolution"]}

    def completed(run_dir: Path) -> dict[str, Any]:
        name = next(name for name, config in stages.items() if run_service._stage_destination(config) == run_dir)
        return {"config": stages[name], "summary": {"training_outcome": outcomes[name]}}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "validate_completed_run", completed)
    monkeypatch.setattr(
        run_service.transient_handoff,
        "validate_stage_a_handoff",
        lambda *_args, expected_source_run_name, **_kwargs: handoffs.append(expected_source_run_name),
    )
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: dict(_CPU_RECEIPT))
    monkeypatch.setattr(budget, "publish_b_receipt", lambda *_args: None)
    monkeypatch.setattr(budget, "load_b_receipt", lambda *_args, **_kwargs: {"test": "receipt"})
    plans: list[dict[str, Any]] = []
    run_plan = run_service.run_experiment
    monkeypatch.setattr(run_service, "run_experiment", lambda *args, **kwargs: plans.append(run_plan(*args, **kwargs)) or plans[-1])

    assert cli_train.main([str(config_path)]) == 0
    assert list(stages) == ["a0", "b", "a1"]
    assert (plans[0]["stage_outcomes"]["a0"]["result"]["status"], plans[0]["unrun_stages"]) == ("diverged", ())
    assert set(handoffs) == {stages["a0"]["run"]["name"]}
    out = capsys.readouterr().out
    assert "Training status: completed" in out
    assert re.search(r"Stage a0 run directory: .* \| diverged at 40\.0 % of the budget", out)


@pytest.mark.parametrize("a0_status", ["diverged", "no_eligible_checkpoint"])
def test_an_a0_without_an_eligible_checkpoint_stops_the_plan_with_its_own_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], a0_status: str
) -> None:
    """A0 without an eligible checkpoint is no branch point: B and A1 never start, and the requested pair exits 2."""
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    executed: list[str] = []

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        executed.append(config["training"]["comparison_arm"])
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        divergence = "non-finite loss" if a0_status == "diverged" else None
        outcome = {"status": a0_status, "budget_progress": 0.4 if divergence else 1.0, "eligible_points": 0, "divergence": divergence}
        result = {"status": a0_status, "best_epoch": None, "best_metric": None, "training_outcome": outcome}
        return {"run_dir": run_dir, "result": result, "device_resolution": kwargs["device_resolution"]}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: pytest.fail("A0 admitted as a branch point"))
    outcomes: list[dict[str, Any]] = []
    run_plan = run_service.run_experiment
    monkeypatch.setattr(run_service, "run_experiment", lambda *args, **kwargs: outcomes.append(run_plan(*args, **kwargs)) or outcomes[-1])

    assert cli_train.main([str(config_path)]) == 2
    assert f"Training status: {a0_status}" in capsys.readouterr().out
    assert executed == ["a0"]
    assert (tuple(outcomes[0]["stage_outcomes"]), outcomes[0]["training_status"], outcomes[0]["unrun_stages"]) == (("a0",), a0_status, ("b", "a1"))


@pytest.mark.parametrize("resumable", [True, False])
@pytest.mark.parametrize("interrupted", [False, True])
def test_a1_failure_or_interrupt_reports_exact_leaf_and_only_verified_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resumable: bool,
    interrupted: bool,
) -> None:
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    executed: list[str] = []
    stages: dict[str, dict[str, Any]] = {}
    failure = KeyboardInterrupt("A1 synthetic worker interrupt") if interrupted else ValueError("A1 synthetic worker failure")

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = config["training"]["comparison_arm"]
        stages[name] = config
        executed.append(name)
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        if name == "a0":
            manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{}", encoding="utf-8")
        if name == "a1":
            raise failure
        return {"run_dir": run_dir, "result": {"status": "completed", "best_epoch": 1}, "device_resolution": kwargs["device_resolution"]}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_service, "validate_completed_run", lambda _run_dir: {"config": stages["b"], "summary": {"train_seconds": 1.0}})
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: dict(_CPU_RECEIPT))
    monkeypatch.setattr(budget, "publish_b_receipt", lambda *_args: None)
    monkeypatch.setattr(budget, "load_b_receipt", lambda *_args, **_kwargs: {"test": "receipt"})
    inspected: list[Path] = []

    def fake_recovery(run_dir: Path, _requested: dict[str, Any], *, config_path: Path | str) -> tuple[bool, str]:
        assert Path(config_path) == config_path_outer
        inspected.append(run_dir)
        return resumable, "validated saved checkpoint" if resumable else "no valid checkpoint"

    config_path_outer = config_path
    monkeypatch.setattr(run_service, "_matched_failed_stage_recovery", fake_recovery)
    expected_exit = 130 if interrupted else 1
    assert cli_train.main([str(config_path)]) == expected_exit
    assert executed == ["a0", "b", "a1"]
    expected_leaf = run_service._stage_destination(stages["a1"])
    assert inspected == [expected_leaf]
    stderr = capsys.readouterr().err
    status = "interrupted" if interrupted else "failed"
    assert f"Matched stage a1 {status}. Run directory: {expected_leaf}" in stderr
    if interrupted:
        assert "Training interrupted." in stderr
    else:
        assert "A1 synthetic worker failure" in stderr
    expected_command = f"python -m src.experiments.cli.cli_train {config_path} --train-stages a1"
    if resumable:
        assert f"it continues this stage: {expected_command}" in stderr
        assert "--resume" not in stderr
    else:
        assert "not eligible for checkpoint resume" in stderr
        assert "--resume" not in stderr
        assert "Inspect or repair the persisted state" in stderr


def test_failed_stage_recovery_checks_saved_checkpoint_and_rejects_missing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    requested = run_service._with_output_root(configs.resolve_plan(configs.transient_matched_config()).a1, tmp_path / "outputs")
    run_dir = run_service._stage_destination(requested)
    run_dir.mkdir(parents=True)
    run_service.config_loader.save_yaml(requested, common.paths.resolve_run_config_path(run_dir))
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"status": "interrupted", "completed_epoch": 1},
    )
    for path in (
        common.paths.resolve_split_indices_path(run_dir),
        common.paths.resolve_normalizer_path(run_dir),
        common.paths.resolve_last_checkpoint_file(run_dir),
        common.paths.resolve_best_checkpoint_file(run_dir),
    ):
        path.touch()
    monkeypatch.setattr(run_service, "_read_saved_data_artifacts", lambda *_args: (SimpleNamespace(as_payload=dict), {}))
    monkeypatch.setattr(run_service, "_validate_saved_data_contract", lambda *_args: None)
    monkeypatch.setattr(checkpoint_service, "build_checkpoint_identity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(checkpoint_service, "recorded_history_epochs", lambda _payload: 1)
    validated: list[tuple[str, bool]] = []

    def load_valid_checkpoint(path: Path, *, expected_role: str, resume: bool = False, **_kwargs: Any) -> dict[str, Any]:
        assert path.is_file()
        validated.append((expected_role, resume))
        # The last closed epoch 1 at its one committed update.
        return {"best_metric": 0.1, "best_epoch": 1, "completed_epoch": 1, "objective_history": [], "global_step": 1}

    def record_epoch_end(step: float) -> None:
        binding: dict[str, Any] = {"task": "transient_drying", "run_name": str(requested["run"]["name"]), "checkpoint_identity": {}}
        epoch_history.make_epoch_state_callback(run_dir, **binding)(1, {"global_step": step, "train/loss_total": 0.5}, None)

    monkeypatch.setattr(checkpoint_service, "load_checkpoint", load_valid_checkpoint)
    record_epoch_end(1.0)
    assert run_service._matched_failed_stage_recovery(run_dir, requested, config_path=config_path)[0]
    # The last is checked as the continuation source, the best only as evidence to read.
    assert validated == [("last", True), ("best", False)]
    # A history whose counted epoch ends after the last's step would be refused by the resume, so it is not offered.
    record_epoch_end(2.0)
    assert not run_service._matched_failed_stage_recovery(run_dir, requested, config_path=config_path)[0]
    common.paths.resolve_last_checkpoint_file(run_dir).unlink()
    eligible, reason = run_service._matched_failed_stage_recovery(run_dir, requested, config_path=config_path)
    assert not eligible
    assert "Missing resume state" in reason


def test_a1_prerequisite_failure_never_advises_checkpoint_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    stages: dict[str, dict[str, Any]] = {}
    receipt_reads = 0

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = config["training"]["comparison_arm"]
        stages[name] = config
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        if name == "a0":
            manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{}", encoding="utf-8")
        return {"run_dir": run_dir, "result": {"status": "completed", "best_epoch": 1}, "device_resolution": kwargs["device_resolution"]}

    def fake_receipt(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        nonlocal receipt_reads
        receipt_reads += 1
        if receipt_reads == 2:
            msg = "B receipt became invalid"
            raise ValueError(msg)
        return {"test": "receipt"}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_service, "validate_completed_run", lambda _run_dir: {"config": stages["b"], "summary": {"train_seconds": 1.0}})
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: dict(_CPU_RECEIPT))
    monkeypatch.setattr(budget, "publish_b_receipt", lambda *_args: None)
    monkeypatch.setattr(budget, "load_b_receipt", fake_receipt)
    monkeypatch.setattr(
        run_service,
        "_matched_failed_stage_recovery",
        lambda *_args, **_kwargs: pytest.fail("A1 resume inspection must not run after prerequisite failure"),
    )
    assert cli_train.main([str(config_path)]) == 1
    stderr = capsys.readouterr().err
    assert "Matched stage a1 failed" in stderr
    assert "B receipt became invalid" in stderr
    assert "Prerequisite admission failed before this stage executed" in stderr
    assert "--resume" not in stderr


def test_short_last_group_commits_once_and_excludes_loader_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two units under k = 3 form one short group of two microbatches: two transitions committed, its seconds without both loads."""
    monkeypatch.setattr(loop.time, "perf_counter", _Clock())
    dataset = TensorDataset(torch.arange(2, dtype=torch.float32).reshape(-1, 1))
    sampler = datasets.runtime.units.UnitSampler(
        unit_kind="window", cases=(0, 1), keys=None, units_per_update=3, memory_batch=1, seed=15, full_set_transitions=2
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    sampler.begin_epoch(0)
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.001)
    adapter = _Adapter()
    transition_budget = stage_budget.StageBudget(unit="supervised_transitions", total=100)
    result = loop.train_update_groups(
        model, loader, optimizer, torch.nn.MSELoss(), torch.device("cpu"), adapter, budget=transition_budget, adapter=adapter
    )
    assert result["optimizer_steps"] == 1
    assert result["train/last_group_units"] == 2
    assert sampler.cursor == 2
    assert (transition_budget.units, transition_budget.supervised_transitions) == (2, 2)
    # Clock reads of 0.25 each: the group starts at read 2 (0.5) and ends after its step at read 7 (1.75), less two loads of 0.25.
    assert transition_budget.train_seconds == pytest.approx(1.75 - 0.5 - 0.5)


@pytest.mark.parametrize("error", [RuntimeError, FloatingPointError])
def test_a_group_an_exception_or_a_divergence_ends_counts_in_the_attempt_and_never_in_the_budget(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """The optimisation seconds of a group up to its exception or divergence, without loader waits, go to the attempt; nothing is committed."""
    monkeypatch.setattr(loop.time, "perf_counter", _Clock())
    dataset = TensorDataset(torch.arange(3, dtype=torch.float32).reshape(-1, 1))
    sampler = datasets.runtime.units.UnitSampler(
        unit_kind="window", cases=(0, 1, 2), keys=None, units_per_update=3, memory_batch=1, seed=15, full_set_transitions=3
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    sampler.begin_epoch(0)
    model = torch.nn.Linear(1, 1)
    budget = stage_budget.StageBudget(unit="supervised_transitions", total=100)
    attempt = loop.AttemptWork()

    def train() -> dict[str, Any]:
        return loop.train_update_groups(
            model,
            loader,
            SGD(model.parameters(), lr=0.001),
            torch.nn.MSELoss(),
            torch.device("cpu"),
            _FailingAdapter(error),
            budget=budget,
            attempt=attempt,
        )

    if error is FloatingPointError:
        # A divergence ends the stretch without raising; its group is attempted, never committed.
        result = train()
        assert result["divergence"] == "synthetic failure in the second microbatch"
        assert result["optimizer_steps"] == 0
    else:
        with pytest.raises(RuntimeError, match="second microbatch"):
            train()

    # Clock reads of 0.25 each: the group starts at read 2 (0.5), two loads take 0.25 each, and the exception is timed at read 7 (1.75).
    assert attempt.interrupted_group_seconds == pytest.approx(1.75 - 0.5 - 0.5)
    assert (budget.train_seconds, budget.units, budget.supervised_transitions, sampler.cursor) == (0.0, 0, 0, 0)


def _record_history(events: list[tuple[str, int]]) -> Callable[[int, dict[str, float] | None, dict[str, float] | None], None]:
    """Record the loop's history writes in their order: a validation record by its point, then an epoch record by its epoch."""

    def record(epoch: int, training: dict[str, float] | None, validation: dict[str, float] | None) -> None:
        if validation is not None:
            events.append(("validation", int(validation["validation/point"])))
        if training is not None:
            events.append(("history", epoch))

    return record


def _matched_loop_config(arm: str) -> dict[str, Any]:
    """Return the resolved-config slice the matched training loop reads for one B or A1 stage."""
    return {
        "task": "transient_drying",
        "training": {"comparison_arm": arm, "matched_a1_b": {"role": arm}},
        "evaluation": {"objective": _loop_identity()["objective"]},
        "tracking": {"wandb": {"monitor": {"enabled": False, "max_cases": 1}}},
    }


def test_train_loop_interrupted_inside_an_epoch_resumes_same_epoch_then_selects_terminal_partial_a1(tmp_path: Path) -> None:
    """
    Interrupt an A1 inside its first epoch, then resume the same epoch on the same identity to its budget end.

    The budget of two supervised transitions puts point 10 after the first group, which validates and publishes
    best and last before its observers, where the interruption strikes; the resumed second
    group reaches point 20.
    """
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    config = _matched_loop_config("a1")
    identity = _loop_identity()
    events: list[tuple[str, int]] = []

    def observe(step: int, *, interrupt: bool) -> None:
        events.append(("observers", step))
        if interrupt:
            raise KeyboardInterrupt

    def run(*, resume: bool) -> dict[str, Any]:
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():  # a fixed start: each group moves the output towards the zero target, so the terminal point is the best
            model.weight.fill_(1.0)
            model.bias.zero_()
        optimizer = SGD(model.parameters(), lr=0.001)
        train_loader = _train_loader(values, seed=21)
        eval_loader = DataLoader(values, batch_size=1, shuffle=False, num_workers=0)
        return loop.train_loop(
            config=config,
            device=torch.device("cpu"),
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_loss=torch.nn.MSELoss(),
            eval_metrics={"mse": _Metric()},
            save_dir=tmp_path,
            resume_from=tmp_path / "last_checkpoint.pt" if resume else None,
            checkpoint_identity=identity,
            adapter=_LoopAdapter(),
            epoch_state_callback=_record_history(events),
            event_callback=lambda step, _metrics: observe(step, interrupt=not resume),
            budget=stage_budget.StageBudget(unit="supervised_transitions", total=2),
        )

    with pytest.raises(KeyboardInterrupt):
        run(resume=False)
    # Point 10 inside epoch 1 is recorded before its last and observers.
    assert events == [("validation", 10), ("observers", 1)]
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert (saved["train_sampler_state_dict"]["epoch"], saved["train_sampler_state_dict"]["cursor"], saved["next_epoch"]) == (0, 1, 1)
    # The interrupted partial epoch has no history record yet, so recovery replays it.
    assert checkpoint_service.recorded_history_epochs(saved) == 0
    second = run(resume=True)
    assert second["status"] == "completed"
    assert second["best_epoch"] == 1
    assert [(entry["point"], entry["epoch"]) for entry in second["objective_history"]] == [(10, 1), (20, 1)]
    # The budget end inside epoch 1 writes its validation and epoch records before the event observers.
    assert events == [("validation", 10), ("observers", 1), ("validation", 20), ("history", 1), ("observers", 2)]
    assert [entry["point"] for entry in saved["objective_history"]] == [10]
    terminal = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert terminal["train_sampler_state_dict"]["cursor"] == 2
    assert terminal["train_sampler_state_dict"]["size"] == 3
    assert checkpoint_service.recorded_history_epochs(terminal) == 1
    # The budget-ended partial-epoch A1 best keeps evaluation state at the epoch and step it was evaluated.
    best = checkpoint_service.load_checkpoint(
        tmp_path / "best_checkpoint.pt",
        expected_identity=identity,
        expected_role="best",
        payload_kind=checkpoint_service.best_payload_kind(config),
        scheduler_expected=False,
        require_best=True,
        adapter_expected=True,
    )
    assert best["payload_kind"] == "evaluation"
    assert {"next_epoch", "optimizer_state_dict", "train_sampler_state_dict"}.isdisjoint(best)
    assert (best["completed_epoch"], best["best_epoch"], best["global_step"]) == (1, 1, terminal["global_step"])


@pytest.mark.parametrize(("mode", "kind"), [("matched_a1_b", "full"), ("stage_a_only", "evaluation")])
def test_only_the_matched_a0_best_keeps_its_training_state_for_handoff(tmp_path: Path, mode: str, kind: str) -> None:
    """The matched A0 best is the teacher handoff source; a stage-A-only A0 never branches and keeps evaluation state."""
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    config = {
        "task": "transient_drying",
        "training": {"budget_epochs": 1, "comparison_arm": "a0", "stage_schedule": {"mode": mode}},
        "evaluation": {"objective": _loop_identity()["objective"]},
        "tracking": {"wandb": {"monitor": {"enabled": False, "max_cases": 1}}},
    }
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    result = loop.train_loop(
        config=config,
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=_train_loader(values, seed=21),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        checkpoint_identity=_loop_identity(),
        adapter=_LoopAdapter(),
        budget=stage_budget.StageBudget(unit="units", total=3),
    )

    assert result["status"] == "completed"
    assert result["global_step"] == 3
    best = torch.load(tmp_path / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    assert best["payload_kind"] == kind == checkpoint_service.best_payload_kind(config)
    assert checkpoint_service.is_teacher_handoff_source(config) is (kind == "full")
    assert ("optimizer_state_dict" in best) is (kind == "full")
    assert "adapter_state_dict" in best
    if kind == "full":
        target = torch.nn.Linear(1, 1)
        target_optimizer = SGD(target.parameters(), lr=0.001, momentum=0.9)
        restored = checkpoint_service.restore_handoff_checkpoint(
            best,
            expected_source_identity=_loop_identity(),
            model=target,
            optimizer=target_optimizer,
            loss=torch.nn.MSELoss(),
            train_loader=_train_loader(values, seed=5),
        )
        assert restored["source_global_step"] == best["global_step"]
        assert all(torch.equal(value, best["model_state_dict"][name]) for name, value in target.state_dict().items())
        assert target_optimizer.state_dict()["state"] == best["optimizer_state_dict"]["state"]


def test_a_transition_budget_validates_at_its_points_inside_epochs_and_records_the_skipped_ones(tmp_path: Path) -> None:
    """
    A B budget of 4 supervised transitions: every one-transition group crosses five of the 20 points.

    Each group validates once, right after it, also inside an epoch of three groups; the
    other crossed points are recorded as skipped, and the fourth group ends the budget at point 20.
    """
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    model = torch.nn.Linear(1, 1)
    result = loop.train_loop(
        config=_matched_loop_config("b"),
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=_train_loader(values, seed=21),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        checkpoint_identity=_loop_identity(),
        adapter=_LoopAdapter(),
        budget=stage_budget.StageBudget(unit="supervised_transitions", total=4),
    )

    assert result["status"] == "completed"
    assert [(entry["point"], entry["epoch"]) for entry in result["objective_history"]] == [(5, 1), (10, 1), (15, 1), (20, 2)]
    assert (result["budget"]["units"], result["budget"]["supervised_transitions"]) == (4, 4)
    assert result["budget"]["skipped_points"] == [point for point in range(1, 20) if point % 5]
    assert result["global_step"] == 4


class _DivergingAfterGroups(_LoopAdapter):
    """Diverge in the group after ``groups`` committed ones."""

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.limit, self.begun = groups, 0

    def begin_group(self, group: datasets.runtime.units.UnitGroup, *, progress: float) -> None:
        super().begin_group(group, progress=progress)
        self.begun += 1

    def unit_losses(
        self, model: torch.nn.Module, raw_batch: tuple[torch.Tensor], loss: torch.nn.Module, *, positions: tuple[int, ...]
    ) -> Iterator[adapter_contract.UnitLosses]:
        if self.begun > self.limit:
            message = "synthetic non-finite loss"
            raise FloatingPointError(message)
        yield from super().unit_losses(model, raw_batch, loss, positions=positions)


def _matched_loop(
    run_dir: Path,
    arm: str,
    stage: stage_budget.StageBudget,
    *,
    k: int,
    adapter: _LoopAdapter | None = None,
    resume: bool = False,
    interrupt: bool = False,
) -> tuple[dict[str, Any], list[list[int]]]:
    """
    Train a B (window units) or A1 (transition units) in groups of ``k`` over ten one-transition units, writing its history.

    Returns the loop result and every validation's point and committed supervised transitions; ``interrupt`` raises
    KeyboardInterrupt in the observers of the first event, after its ``last`` was published.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    values = TensorDataset(torch.arange(1.0, 11.0).reshape(-1, 1))
    model = torch.nn.Linear(1, 1)
    validations: list[list[int]] = []

    def observe(_step: int, metrics: dict[str, float]) -> None:
        if "validation/point" in metrics:
            validations.append([int(metrics["validation/point"]), int(metrics["budget/supervised_transitions"])])
        if interrupt:
            raise KeyboardInterrupt

    result = loop.train_loop(
        config=_matched_loop_config(arm),
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=_train_loader(values, seed=21, unit_kind="window" if arm == "b" else "transition", k=k),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=run_dir,
        resume_from=run_dir / "last_checkpoint.pt" if resume else None,
        checkpoint_identity=_loop_identity(),
        adapter=adapter or _LoopAdapter(),
        epoch_state_callback=None
        if resume
        else epoch_history.make_epoch_state_callback(run_dir, task="transient_drying", run_name=arm, checkpoint_identity=_loop_identity()),
        event_callback=observe,
        budget=stage,
    )
    return result, validations


def test_a1_cuts_its_last_group_to_commit_exactly_the_receipts_transitions(tmp_path: Path) -> None:
    """
    B diverged after 3 supervised transitions; its A1 trains groups of k = 2 transition units.

    A1's second group is cut to the one remaining transition, so A1 commits exactly the receipt's 3, not 4;
    its sampler state ends inside the epoch, between group boundaries, and a finalize-only resume accepts it.
    """
    receipt = _receipt(_completed_b_summary(total=20, spent=3, outcome="diverged", eligible=False), tmp_path / "b")

    def a1_budget() -> stage_budget.StageBudget:
        return stage_budget.StageBudget(unit="supervised_transitions", total=receipt["supervised_transitions"], replay=receipt["b_validations"])

    result, _validations = _matched_loop(tmp_path / "a1", "a1", a1_budget(), k=2)

    assert (result["budget"]["units"], result["budget"]["supervised_transitions"]) == (3, 3)
    assert [row[2] for row in result["update_work"]] == [2, 1]
    last = tmp_path / "a1" / "last_checkpoint.pt"
    terminal = torch.load(last, map_location="cpu", weights_only=False)
    assert (terminal["train_sampler_state_dict"]["cursor"], terminal["train_sampler_state_dict"]["stop"]) == (3, 3)
    assert _matched_loop(tmp_path / "a1", "a1", a1_budget(), k=2, resume=True)[0]["budget"] == result["budget"]


def test_an_a1_resumed_inside_its_stretch_validates_at_the_remaining_replay_counts(tmp_path: Path) -> None:
    """
    B validated at 2 (point 2) and 5 (point 5) of its 20 and diverged at 7; its A1 trains groups of k = 2 transition units.

    The first stretch validates point 2 at 2 and publishes ``last`` with the stop at 7 and point 5 pending, where an
    interruption strikes; the resumed A1 validates point 5 at 6 and ends at 7. Only such an A1 checkpoint carries a
    stop, exactly its cursor plus the remaining transitions: another stop, or this stop in a budget that replays
    nothing (every A0 or B), is refused by the checkpoint validation that finished-run admission and resume share.
    """
    run_dir = tmp_path / "a1"

    def a1_budget() -> stage_budget.StageBudget:
        return stage_budget.StageBudget(unit="supervised_transitions", total=7, replay=[[2, 2], [5, 5]])

    with pytest.raises(KeyboardInterrupt):
        _matched_loop(run_dir, "a1", a1_budget(), k=2, interrupt=True)
    saved = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert (saved["train_sampler_state_dict"]["cursor"], saved["train_sampler_state_dict"]["stop"]) == (2, 7)

    result, validations = _matched_loop(run_dir, "a1", a1_budget(), k=2, resume=True)

    assert validations == [[5, 6]]
    assert [entry["point"] for entry in result["objective_history"]] == [2, 5]
    assert [row[2] for row in result["update_work"]] == [2, 2, 2, 1]
    admission = {"expected_identity": _loop_identity(), "expected_role": "last", "payload_kind": "full", "scheduler_expected": False}
    checkpoint_service.validate_checkpoint(saved, **admission, require_best=False, adapter_expected=True)
    no_replay = {key: value for key, value in saved["budget_state_dict"].items() if key != "replay"}
    for changed in ({"train_sampler_state_dict": {**saved["train_sampler_state_dict"], "stop": 9}}, {"budget_state_dict": no_replay}):
        with pytest.raises(ValueError, match="sampler stop"):
            checkpoint_service.validate_checkpoint({**saved, **changed}, **admission, require_best=False, adapter_expected=True)


@pytest.mark.parametrize(
    ("b_total", "diverge_after", "a1_counts"),
    [(40, None, [6, 10, 16, 20, 26, 30, 36, 40]), (150, 4, [10, 16])],
    ids=("b-completed", "b-diverged"),
)
def test_a1_replays_b_validations_with_its_own_grouping(tmp_path: Path, b_total: int, diverge_after: int | None, a1_counts: list[int]) -> None:
    """
    B trains groups of 5 transitions, A1 groups of 2: A1 validates once per recorded B validation, at its first group end at or after B's count.

    B over 40 transitions validates 8 times, at 5, 10, ..., 40, not 20 times like an A1 with its own points. B over
    150 transitions validates at 10 (point 1) and 15 (point 2) and diverges in its fifth group at 20; its A1 validates
    at 10 and 16 and ends at 20 without a validation. Both arms have the same checkpoint candidates.
    """
    adapter = None if diverge_after is None else _DivergingAfterGroups(diverge_after)
    b_result, b_validations = _matched_loop(
        tmp_path / "b", "b", stage_budget.StageBudget(unit="supervised_transitions", total=b_total), k=5, adapter=adapter
    )
    spent = b_result["budget"]["supervised_transitions"]
    summary = _completed_b_summary(total=b_total, spent=spent, outcome=b_result["training_outcome"]["status"])
    summary.update(
        terminal_budget=b_result["budget"],
        training_outcome=b_result["training_outcome"],
        validation_plan_deviation={"skipped_points": b_result["budget"]["skipped_points"]},
    )
    plan = configs.resolve_plan(configs.transient_matched_config())
    finished = _finished(summary, tmp_path / "b", update_work=b_result["update_work"])
    receipt = budget.build_b_receipt(parent_identity_sha256="a" * 64, a0_handoff_sha256="b" * 64, b_config=plan.b, finished_b=finished)
    assert receipt["b_validations"] == b_validations

    a1_budget = stage_budget.StageBudget(unit="supervised_transitions", total=receipt["supervised_transitions"], replay=receipt["b_validations"])
    a1_result, a1_validations = _matched_loop(tmp_path / "a1", "a1", a1_budget, k=2)

    assert [point for point, _count in a1_validations] == [point for point, _count in b_validations]
    assert [count for _point, count in a1_validations] == a1_counts
    assert len(a1_result["objective_history"]) == len(b_result["objective_history"])
    assert a1_result["budget"]["supervised_transitions"] == spent


def _windowed_terminal_loop(
    tmp_path: Path,
    *,
    resume: bool,
    events: list[tuple[str, int]],
    adapter: _LoopAdapter | None = None,
    budget_transitions: int = 2,
) -> tuple[dict[str, Any], torch.nn.Module]:
    """Run the real matched B loop; by default its budget ends with the first update group, whose two windows supervise two transitions."""
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    model = torch.nn.Linear(1, 1)
    result = loop.train_loop(
        config=_matched_loop_config("b"),
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=_train_loader(values, seed=21),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        resume_from=tmp_path / "last_checkpoint.pt" if resume else None,
        checkpoint_identity=_loop_identity(),
        adapter=_WindowedLoopAdapter() if adapter is None else adapter,
        epoch_state_callback=_record_history(events),
        event_callback=lambda step, _metrics: events.append(("observers", step)),
        budget=stage_budget.StageBudget(unit="supervised_transitions", total=budget_transitions),
    )
    return result, model


def test_a1_matched_to_a_b_that_diverged_in_its_first_group_runs_no_update(tmp_path: Path) -> None:
    """B's receipt holds 0 transitions, so A1's budget is complete from the start: no update, no validation, no checkpoint."""
    receipt = _receipt(_completed_b_summary(total=20, spent=0, outcome="diverged", eligible=False), tmp_path / "b")
    plan = configs.resolve_plan(configs.transient_matched_config())
    a1_budget = run_service._stage_budget(plan.a1, training_units={"full_set_transitions": 3}, matched_receipt=receipt)
    events: list[tuple[str, int]] = []
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    model = torch.nn.Linear(1, 1)
    result = loop.train_loop(
        config=_matched_loop_config("a1"),
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=_train_loader(values, seed=21),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        checkpoint_identity=_loop_identity(),
        adapter=_LoopAdapter(),
        epoch_state_callback=_record_history(events),
        event_callback=lambda step, _metrics: events.append(("observers", step)),
        budget=a1_budget,
    )

    assert result["status"] == "no_eligible_checkpoint"
    assert result["training_outcome"] == {"status": "no_eligible_checkpoint", "budget_progress": 1.0, "eligible_points": 0, "divergence": None}
    assert (result["global_step"], result["terminal_epoch"], result["last_checkpoint_path"]) == (0, 0, None)
    assert events == []
    assert not (tmp_path / "last_checkpoint.pt").exists()
    assert not (tmp_path / "best_checkpoint.pt").exists()


def test_budget_crossing_in_the_first_trajectory_group_publishes_its_committed_cursor(tmp_path: Path) -> None:
    """
    Complete the budget with the first update group, whose trajectory trains as two windows.

    The windows never commit loader progress on their own: the terminal checkpoints
    record the committed group (cursor 1) and stay loadable for terminal validation
    and recovery, while a cursor before any committed group stays rejected. The
    terminal epoch's history record precedes the checkpoint, so recovery keeps it.
    """
    events: list[tuple[str, int]] = []
    result, _model = _windowed_terminal_loop(tmp_path, resume=False, events=events)

    assert result["status"] == "completed"
    assert result["best_epoch"] == 1
    terminal = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert (terminal["train_sampler_state_dict"]["cursor"], terminal["next_epoch"]) == (1, 1)
    assert checkpoint_service.recorded_history_epochs(terminal) == events.count(("history", 1)) == 1
    uncommitted = copy.deepcopy(terminal)
    uncommitted["train_sampler_state_dict"]["cursor"] = 0
    torch.save(uncommitted, tmp_path / "uncommitted.pt")
    with pytest.raises(ValueError, match="committed group"):
        checkpoint_service.load_checkpoint(
            tmp_path / "uncommitted.pt",
            expected_identity=_loop_identity(),
            expected_role="last",
            payload_kind="full",
            scheduler_expected=False,
            require_best=True,
            adapter_expected=True,
        )


def test_resuming_a_terminal_budget_checkpoint_finalizes_without_optimizer_work(tmp_path: Path) -> None:
    """
    Resume from terminal checkpoints, as after an interruption before run finalization.

    Any further optimizer group would cross the completed budget, so the resume
    must restore and validate the published checkpoints without training, without
    new epoch events and without changing the restored parameters.
    """
    events: list[tuple[str, int]] = []
    first, _model = _windowed_terminal_loop(tmp_path, resume=False, events=events)
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)

    resumed, model = _windowed_terminal_loop(tmp_path, resume=True, events=events)

    assert resumed["status"] == "completed"
    assert (resumed["best_epoch"], resumed["best_metric"]) == (first["best_epoch"], first["best_metric"])
    assert resumed["objective_history"] == first["objective_history"]
    assert resumed["terminal_epoch"] == saved["completed_epoch"]
    assert events == [("validation", 20), ("history", 1), ("observers", 1)]
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved["model_state_dict"][name])


def test_terminal_budget_at_an_epoch_end_still_resumes_to_finalization(tmp_path: Path) -> None:
    """
    Complete the budget with the last optimizer group of an epoch.

    Its terminal checkpoint describes a complete epoch; the resume of an interrupted run
    finalizes it without a further group.
    """
    events: list[tuple[str, int]] = []
    _windowed_terminal_loop(tmp_path, resume=False, events=events, adapter=_LoopAdapter(), budget_transitions=3)
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)

    assert saved["train_sampler_state_dict"]["cursor"] == saved["train_sampler_state_dict"]["size"] == 3

    resumed, model = _windowed_terminal_loop(tmp_path, resume=True, events=events, adapter=_LoopAdapter(), budget_transitions=3)

    assert resumed["status"] == "completed"
    # One event per validation point (6, 13, 20), each recorded before its observers; the epoch record comes with point 20.
    assert events == [("validation", 6), ("observers", 1), ("validation", 13), ("observers", 2), ("validation", 20), ("history", 1), ("observers", 3)]
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved["model_state_dict"][name])


def test_a_divergence_inside_an_epoch_keeps_its_history_record_for_resume_reconciliation(tmp_path: Path) -> None:
    """A divergence in the second group writes the partial epoch's history record before its terminal last, so recovery counts it."""
    events: list[tuple[str, int]] = []
    result, _model = _windowed_terminal_loop(tmp_path, resume=False, events=events, adapter=_DivergingLoopAdapter(), budget_transitions=3)

    assert result["status"] == "diverged"
    terminal = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert terminal["divergence"] == "synthetic non-finite loss"
    assert (terminal["train_sampler_state_dict"]["cursor"], terminal["train_sampler_state_dict"]["size"]) == (1, 3)
    assert not checkpoint_service.records_completed_budget(terminal)
    assert checkpoint_service.recorded_history_epochs(terminal) == events.count(("history", 1)) == 1


class _FakeWandbRun:
    """Record the steps a W&B run receives; the tracker owns the strict-step contract."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.tags: tuple[str, ...] = ()
        self.steps: list[int] = []

    def define_metric(self, name: str, **kwargs: Any) -> None:
        del name, kwargs

    def log(self, data: dict[str, Any]) -> None:
        # History is logged without W&B's own step; the payload's supervised_transitions is the x-axis.
        self.steps.append(int(data["global_step"]))

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


def test_a_divergence_in_the_first_group_of_a_stretch_adds_no_tracked_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The divergence after the event at step 1 commits no step, so the enabled tracker gets no event and the stage ends diverged."""
    run = _FakeWandbRun()
    fake = SimpleNamespace(Settings=lambda **settings: settings, setup=lambda **_settings: None, init=lambda **_settings: run)
    original = run_service.tracking.importlib.import_module
    monkeypatch.setattr(run_service.tracking.importlib, "import_module", lambda name, *args: fake if name == "wandb" else original(name, *args))
    monkeypatch.setenv("WANDB_API_KEY", "test-only-wandb-key")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    raw = configs.transient_matched_config()
    raw["tracking"]["wandb"]["mode"] = "online"
    session = run_service.tracking.initialize_wandb(configs.resolve_plan(raw).b, run_dir=tmp_path / "wandb")
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    model = torch.nn.Linear(1, 1)

    result = loop.train_loop(
        config=_matched_loop_config("b"),
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=_train_loader(values, seed=21),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        checkpoint_identity=_loop_identity(),
        adapter=_DivergingLoopAdapter(),
        event_callback=session.log_event,
        budget=stage_budget.StageBudget(unit="supervised_transitions", total=3),
    )

    assert result["status"] == "diverged"
    assert run.steps == [1]


@pytest.mark.parametrize("terminal_status", ["completed", "diverged", "unsuccessful", "failed", "interrupted"])
@pytest.mark.parametrize("unpublished_prior_attempt", [False, True])
def test_attempts_record_their_seconds_and_committed_work_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    unpublished_prior_attempt: bool,
) -> None:
    """
    Record an interrupted first attempt and a resume that completes, ends unsuccessful, fails or is interrupted, each in its runtime session.

    The totals sum the attempts; after an attempt killed before its end entry they read unknown, never
    zero. A failed or interrupted resume computed groups after its newest durable state, which a later
    resume trains again, and part of the group its exception interrupted: both count in its train_seconds,
    never in the committed work. Each attempt
    starts on a fresh unit sampler, as a run start does, and records its own memory batch.
    """
    config = configs.resolve(configs.direct_config(), output_root=tmp_path)
    run_dir = run_service.prepare_fresh_run(config, run_dir=tmp_path / "run")
    scaling = SimpleNamespace(physical_state=SimpleNamespace(scale=torch.ones(1)), to=lambda _device: None)
    sampler = datasets.runtime.units.UnitSampler(unit_kind="case", cases=(0,), keys=None, units_per_update=1, memory_batch=1, seed=0)
    loaders = {
        "train": DataLoader(TensorDataset(torch.zeros(1)), batch_sampler=sampler),
        "eval": [0],
        "id_test": [0],
        "material_ood": None,
        "parameter_ood": None,
        "scaling": scaling,
        "scaling_artifact": {},
        "split_indices": {},
    }
    checkpoint_identity = {"effective_config_digest": "a" * 64}
    resume_loads: list[tuple[str, bool]] = []

    def load_last(*_args: Any, expected_role: str, resume: bool = False, **_kwargs: Any) -> dict[str, Any]:
        resume_loads.append((expected_role, resume))
        return {"objective_history": [], "global_step": 0}

    stubs = (
        (run_service, "_grid_spacing", lambda *_args: (0.003, 0.003)),
        (run_service.config_loader, "create_dataloaders_from_config", lambda *_args, **_kwargs: loaders),
        (run_service.learning.models.factory, "build_model", lambda *_args, **_kwargs: torch.nn.Linear(1, 1)),
        (run_service.learning.losses.factory, "build_training_loss", lambda *_args, **_kwargs: _ScaledMSELoss()),
        (run_service.learning.metrics.metrics, "build_evaluation_metrics", lambda *_args, **_kwargs: {}),
        (run_service.learning.training.optim, "build_scheduler", lambda *_args, **_kwargs: None),
        (run_service.learning.training.checkpoint, "build_checkpoint_identity", lambda *_args, **_kwargs: checkpoint_identity),
        # The touched checkpoints are placeholders and the stubbed loop writes no epoch record, so a resume restores none.
        (run_service.learning.training.checkpoint, "load_checkpoint", load_last),
        (run_service.learning.training.checkpoint, "recorded_history_epochs", lambda _payload: 0),
        (run_service.tracking, "build_monitor_membership", lambda *_args: None),
    )
    for target, name, stub in stubs:
        monkeypatch.setattr(target, name, stub)
    objective = run_service.config_loader.get_resolved_objective(config)
    result = {
        "status": "interrupted",
        "completed_epoch": 1,
        "global_step": 1,
        "training_work": {"optimizer_steps": 1, "gradient_predictions": 2},
        "best_epoch": 1,
        "best_metric": 0.1,
        "objective": objective,
        "terminal_epoch": 1,
        "terminal_metrics": {},
        "best_checkpoint_path": str(run_dir / "best_checkpoint.pt"),
        "last_checkpoint_path": str(run_dir / "last_checkpoint.pt"),
        "budget": {
            "unit": "units",
            "total": 3,
            "units": 1,
            "supervised_transitions": 1,
            "train_seconds": 0.25,
            "validated_point": 6,
            "skipped_points": [1, 2, 3, 4, 5],
        },
    }
    for filename in ("best_checkpoint.pt", "last_checkpoint.pt", "split_indices.pt"):
        (run_dir / filename).touch()
    clock = [100.0]
    monkeypatch.setattr(run_service.time, "monotonic", lambda: clock[0])

    def work(steps: int, seconds: float) -> dict[str, Any]:
        return {"optimizer_steps": steps, "gradient_predictions": steps, "units": steps, "update_distinct_cases": steps, "train_seconds": seconds}

    # Per attempt: the committed work it restores, the optimisation seconds it computes, and its newest durable work.
    plan = [
        (work(0, 0.0), 0.25, work(1, 0.25)),
        (work(1, 0.25), 1.5, work(3, 1.75) if terminal_status in {"completed", "diverged", "unsuccessful"} else work(2, 1.25)),
    ]

    def measured_loop(**kwargs: Any) -> dict[str, Any]:
        running = run_service.read_run_summary(run_dir)
        assert (running["elapsed_seconds"], running["attempted_train_seconds"], running["duration_coverage"]) == (None, None, "unknown_attempts")
        assert set(running["runtime_sessions"][-1]["attempt"].values()) == {None}
        at_start, computed, durable = plan.pop(0)
        kwargs["budget"].train_seconds = at_start["train_seconds"] + computed
        kwargs["attempt_work"].at_start, kwargs["attempt_work"].durable = at_start, durable
        clock[0] += 4.0 if len(plan) == 1 else 6.0
        if result["status"] in {"failed", "interrupted"}:
            kwargs["attempt_work"].interrupted_group_seconds = 0.125
        if result["status"] == "failed":
            message = "bounded training failure"
            raise RuntimeError(message)
        if result["status"] == "interrupted":
            message = "bounded training interrupt"
            raise KeyboardInterrupt(message)
        return dict(result)

    monkeypatch.setattr(run_service.learning.training.loop, "train_loop", measured_loop)
    selected_metrics = {f"selected/id/{objective['id']}": 0.1}

    def measured_selected(**_kwargs: Any) -> dict[str, Any]:
        clock[0] += 2.0
        return {"selected_epoch": 1, "selected_metrics": selected_metrics}

    monkeypatch.setattr(run_service.learning.training.loop, "evaluate_selected_checkpoint", measured_selected)
    resolution = run_service.learning.device.resolve_device("cpu")
    with pytest.raises(KeyboardInterrupt):
        run_service.execute_prepared_run(config, run_dir=run_dir, device_resolution=resolution)
    first = run_service.read_run_summary(run_dir)
    assert first["status"] == "interrupted"
    # The interrupted attempt adds the 0.125 s of the group it interrupted to the 0.25 s the budget gained.
    assert first["runtime_sessions"][0]["attempt"] | {"ended_at": None} == {
        "ended_at": None,
        "elapsed_seconds": 4.0,
        "train_seconds": 0.375,
        "work_at_start": work(0, 0.0),
        "work_at_end": work(1, 0.25),
    }
    assert first["runtime_sessions"][0]["precision"]["parameter_dtypes"] == ["float32"]
    assert (first["elapsed_seconds"], first["attempted_train_seconds"], first["duration_coverage"]) == (4.0, 0.375, "complete")
    if unpublished_prior_attempt:
        # A hard kill leaves the start entry of the "running" transition: no end entry for that attempt.
        first["runtime_sessions"][0]["attempt"] = dict.fromkeys(first["runtime_sessions"][0]["attempt"])
        first.update(elapsed_seconds=None, attempted_train_seconds=None, duration_coverage="unknown_attempts")
        common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), first)

    result.update(status=terminal_status, global_step=3, training_work={"optimizer_steps": 3, "gradient_predictions": 3})
    result["training_outcome"] = {"status": terminal_status, "budget_progress": 1.0, "eligible_points": 2, "divergence": None}
    if terminal_status == "unsuccessful":
        # The budget ended without an eligible point: the training run ends, like a completed one, with its end entry.
        result.update(status="no_eligible_checkpoint", best_epoch=None, best_metric=None)
        result["training_outcome"] = {"status": "no_eligible_checkpoint", "budget_progress": 1.0, "eligible_points": 0, "divergence": None}
    if terminal_status == "diverged":
        # A divergence at the budget end keeps the best eligible checkpoint before it: a completed run with a diverged outcome.
        result["training_outcome"] = {"status": "diverged", "budget_progress": 1.0, "eligible_points": 2, "divergence": "loss is non-finite"}
    result["budget"] = {
        **result["budget"],
        "units": 3,
        "supervised_transitions": 3,
        "train_seconds": 1.75,
        "validated_point": 20,
        "skipped_points": list(range(1, 20)),
    }
    error_type = RuntimeError if terminal_status == "failed" else KeyboardInterrupt
    arguments = {
        "run_dir": run_dir,
        "resume_from": run_dir / "last_checkpoint.pt",
        "restored_scaling_artifact": loaders["scaling_artifact"],
        "saved_split_indices": {},
        "device_resolution": resolution,
    }
    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["memory_batch"] = config["training"]["memory_batch"] + 1
    ended = terminal_status in {"completed", "diverged", "unsuccessful"}
    run_status = "completed" if terminal_status == "diverged" else terminal_status
    if run_status == "completed":
        returned = run_service.execute_prepared_run(resumed_config, **arguments)
        assert returned["operational_timing"]["selected_checkpoint_evaluation_seconds"] == 2.0
    elif terminal_status == "unsuccessful":
        run_service.execute_prepared_run(resumed_config, **arguments)
    else:
        with pytest.raises(error_type):
            run_service.execute_prepared_run(resumed_config, **arguments)
    summary = run_service.read_run_summary(run_dir)
    sessions = summary["runtime_sessions"]
    # The resumed attempt loads its last checkpoint as continuation source before it reconciles the history.
    assert resume_loads == [("last", True)]
    assert [session["memory_batch"] for session in sessions[-2:]] == [config["training"]["memory_batch"], resumed_config["training"]["memory_batch"]]
    assert summary["status"] == run_status
    second = sessions[-1]["attempt"]
    # A failure or interrupt adds the 0.125 s of the group it interrupted to the 1.5 s the budget gained.
    interrupted = 0.0 if ended else 0.125
    assert (second["elapsed_seconds"], second["train_seconds"], second["work_at_start"]) == (
        8.0 if run_status == "completed" else 6.0,
        1.5 + interrupted,
        work(1, 0.25),
    )
    known = not unpublished_prior_attempt
    assert summary["elapsed_seconds"] == (4.0 + second["elapsed_seconds"] if known else None)
    assert summary["attempted_train_seconds"] == (0.375 + 1.5 + interrupted if known else None)
    assert summary["duration_coverage"] == ("complete" if known else "unknown_attempts")
    if ended:
        # The terminal state is durable, so every computed second is committed; both ends record the stage's units.
        assert second["work_at_end"] == work(3, 1.75)
        assert summary["terminal_budget"]["train_seconds"] == 1.75
        assert summary["training_units"] == run_service.learning.training.loop.unit_evidence(sampler)
        # One result per run: its end status, budget, work and chosen checkpoint sit in its one summary.
        assert [summary[key] for key in ("training_outcome", "training_work", "best_epoch")] == [
            result[key] for key in ("training_outcome", "training_work", "best_epoch")
        ]
    if run_status == "completed":
        assert summary["operational_timing"]["selected_checkpoint_evaluation_seconds"] == 2.0
        assert summary["effective_config_digest"] == checkpoint_identity["effective_config_digest"]
    elif not ended:
        # The last 0.5 s came after the newest durable state: computed, and repeated by the next resume, never committed.
        assert second["work_at_end"] == work(2, 1.25)
        assert summary["training_work"] == {key: value for key, value in work(2, 1.25).items() if key != "train_seconds"}
    assert [event["status"] for event in summary["status_history"]][-3:] == ["interrupted", "running", run_status]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        pytest.param(lambda summary, _last: summary["terminal_budget"].update(supervised_transitions=15), "terminal budget is invalid", id="unspent"),
        pytest.param(lambda summary, _last: summary.update(validation_plan_deviation={"skipped_points": []}), "skipped", id="hidden-deviation"),
        pytest.param(lambda _summary, last: last.update(validated_point=19, skipped_points=list(range(1, 19))), "last checkpoint", id="other-last"),
    ],
)
def test_completed_summary_records_its_spent_budget_and_its_plan_deviation(change: Any, match: str) -> None:
    """A completed summary keeps the last checkpoint's spent budget and lists the crossed points that had no own validation."""
    terminal = stage_budget.StageBudget(unit="supervised_transitions", total=20)
    terminal.record_group(units=4, supervised_transitions=12, train_seconds=1.2)
    terminal.record_validation()
    terminal.record_group(units=4, supervised_transitions=9, train_seconds=0.9)
    terminal.record_validation()
    summary = run_service._terminal_budget_evidence(terminal.state_dict())
    summary["training_outcome"] = {"status": "completed", "budget_progress": 1.0, "eligible_points": 2, "divergence": None}
    last = terminal.state_dict()
    assert summary["validation_plan_deviation"] == {"skipped_points": [*range(1, 12), *range(13, 20)]}
    arguments = {"lifecycle": "completed", "target": ("supervised_transitions", 20, None)}
    run_service._validate_terminal_budget(summary, last, **arguments)

    change(summary, last)
    with pytest.raises(run_service.RunLifecycleError, match=match):
        run_service._validate_terminal_budget(summary, last, **arguments)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        pytest.param("terminal_controller", "post_handoff_optimizer_steps", None, id="null-counter"),
        pytest.param("terminal_curriculum", "draw_index", None, id="null-curriculum"),
        pytest.param("terminal_controller", "validation_seconds", 0.75, id="disagrees-with-history"),
    ],
)
def test_completed_transient_terminal_values_are_admitted_against_their_final_history_record(
    section: str, key: str, value: Any, tmp_path: Path
) -> None:
    """Completed terminal values, a B's curriculum included, must be valid and equal the final epoch record (a trial keeps no last checkpoint)."""
    handoff = curriculum.TeacherHandoffIdentity(
        source_run_name="a0",
        source_checkpoint_sha256="e" * 64,
        source_scaling_sha256="c" * 64,
        task_contract_sha256="d" * 64,
        tensorizer_sha256="f" * 64,
        model_kind="fno",
        input_profile="canonical_physics_complete",
    )
    controller = curriculum.MatchedComputeController(arm="B", stage="stage_b_self_fed", config_digest="a" * 64, teacher_handoff=handoff)
    controller.record_completed_work(microbatches=1, processed_target_transitions=4, forward_transitions=4)
    controller.record_validation_work(0.25)
    controller.record_within_budget_evaluation(0.5, epoch_index=0)
    rollout = curriculum.RolloutCurriculumState.create(curriculum.RolloutCurriculum(lengths=(2,), milestone_fractions=(0.0,)), seed=0)
    # The adapter's own telemetry names the values that the training loop writes into each epoch record.
    stand_in = SimpleNamespace(spec=SimpleNamespace(controller=controller, stage=controller.stage), curriculum_state=rollout)
    telemetry = transient_adapter.TransientTrainingAdapter.telemetry_state(cast("transient_adapter.TransientTrainingAdapter", stand_in))
    checkpoint_identity = {"effective_config_digest": "b" * 64}
    record_epoch = epoch_history.make_epoch_state_callback(tmp_path, task="transient_drying", run_name="b", checkpoint_identity=checkpoint_identity)
    record_epoch(1, {"global_step": 1.0, **{f"transient/{name}": float(item) for name, item in telemetry.items() if item is not None}}, None)
    state = controller.state_dict()
    summary = {
        "checkpoint_identity": checkpoint_identity,
        "transient_scaling": {"semantic_digest": "c" * 64, "task_contract_digest": "d" * 64},
        "terminal_controller": {name: state[name] for name in run_service._TERMINAL_CONTROLLER_KEYS},
        "terminal_curriculum": {name: getattr(rollout, name) for name in run_service._TERMINAL_CURRICULUM_KEYS},
    }
    arguments = {
        "config": {"task": "transient_drying", "run": {"name": "b"}, "training": {"stage": "b"}, "task_contract": {"digest": "d" * 64}},
        "normalizer_state": SimpleNamespace(digest="c" * 64),
        "checkpoint_identity": checkpoint_identity,
        "completed_epoch": 1,
        "validations": (),
    }
    run_service._validate_transient_summary_evidence(tmp_path, summary=summary, **arguments)

    summary[section][key] = value
    with pytest.raises(run_service.RunLifecycleError, match="terminal evidence"):
        run_service._validate_transient_summary_evidence(tmp_path, summary=summary, **arguments)


def test_a_stage_a_summary_that_records_a_curriculum_is_refused(tmp_path: Path) -> None:
    """Teacher-forced A0 and A1 draw no rollout horizon, so their summaries record no terminal curriculum."""
    summary = {"checkpoint_identity": {}, "transient_scaling": {}, "terminal_controller": {}, "terminal_curriculum": {"draw_index": 0}}
    config = {"task": "transient_drying", "run": {"name": "a0"}, "training": {"stage": "a"}, "task_contract": {"digest": "d" * 64}}

    with pytest.raises(run_service.RunLifecycleError, match="records no curriculum"):
        run_service._validate_transient_summary_evidence(
            tmp_path,
            summary=summary,
            config=config,
            normalizer_state=SimpleNamespace(digest="c" * 64),
            checkpoint_identity={},
            completed_epoch=1,
            validations=(),
        )


_TERMINAL = {"active_stage": 1, "max_horizon": 4, "draw_index": 7}


@pytest.mark.parametrize(
    ("stage", "summary_curriculum", "checkpoint_curriculum", "admitted"),
    [
        pytest.param("a", None, None, True, id="stage-a-without-curriculum"),
        pytest.param("a", None, {"curriculum": None}, False, id="stage-a-checkpoint-null-curriculum"),
        pytest.param("a", {"terminal_curriculum": None}, None, False, id="stage-a-summary-null-curriculum"),
        pytest.param("b", {"terminal_curriculum": _TERMINAL}, {"curriculum": {**_TERMINAL, "lengths": [2, 4]}}, True, id="stage-b-equal"),
        pytest.param("b", {"terminal_curriculum": _TERMINAL}, None, False, id="stage-b-checkpoint-without-curriculum"),
        pytest.param("b", {"terminal_curriculum": _TERMINAL}, {"curriculum": {**_TERMINAL, "draw_index": 8}}, False, id="stage-b-mismatch"),
    ],
)
def test_the_terminal_curriculum_check_follows_the_stage_not_the_keys_present(
    stage: str, summary_curriculum: dict[str, Any] | None, checkpoint_curriculum: dict[str, Any] | None, admitted: bool
) -> None:
    """B's summary and last checkpoint both carry equal curricula; A0 and A1 carry the key in neither, not even as null."""
    controller = dict.fromkeys(run_service._TERMINAL_CONTROLLER_KEYS, 0)
    summary = {"terminal_controller": controller, **(summary_curriculum or {})}
    last_checkpoint = {"adapter_state_dict": {"controller": controller, **(checkpoint_curriculum or {})}}
    config = {"training": {"stage": stage}}

    if admitted:
        run_service._validate_transient_terminal_state(summary, last_checkpoint, config=config)
    else:
        with pytest.raises(run_service.RunLifecycleError, match="curriculum"):
            run_service._validate_transient_terminal_state(summary, last_checkpoint, config=config)


@pytest.mark.parametrize(
    ("stage", "change", "admitted"),
    [
        pytest.param("b", None, True, id="producer-b-admitted"),
        pytest.param("b", ("terminal_curriculum", None), False, id="b-without-curriculum"),
        pytest.param("a", ("terminal_curriculum", dict.fromkeys(run_service._TERMINAL_CURRICULUM_KEYS, 0)), False, id="a0-with-curriculum"),
        pytest.param("a", None, True, id="producer-a0-admitted"),
    ],
)
def test_a_stage_without_a_last_checkpoint_is_admitted_by_its_summary_shape(stage: str, change: tuple[str, Any] | None, admitted: bool) -> None:
    """A stage that committed no group has no last checkpoint: its summary still carries B's curriculum and never A's."""
    summary = _no_group_evidence(stage)
    if change is not None:
        key, value = change
        if value is None:
            del summary[key]
        else:
            summary[key] = value
    config = {"training": {"stage": stage}}

    if admitted:
        run_service._validate_transient_terminal_state(summary, None, config=config)
    else:
        with pytest.raises(run_service.RunLifecycleError, match="curriculum"):
            run_service._validate_transient_terminal_state(summary, None, config=config)


def test_a_stopped_plan_exits_two_and_names_every_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """An A0 without an eligible checkpoint stops the plan; B and A1 never ran, the exit code is 2 and no artifact is built."""
    path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    a0, b, a1 = (tmp_path / name for name in ("a0", "b", "a1"))
    a0.mkdir()
    resolution = SimpleNamespace(device=torch.device("cpu"))
    unsuccessful = {
        "status": "no_eligible_checkpoint",
        "best_epoch": None,
        "best_metric": None,
        "training_outcome": {"status": "no_eligible_checkpoint", "budget_progress": 1.0, "eligible_points": 0, "divergence": None},
    }
    monkeypatch.setattr(
        run_service,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": a0,
            "result": unsuccessful,
            "stage_runs": {"a0": a0, "b": b, "a1": a1},
            "stage_outcomes": {"a0": {"result": unsuccessful}},
            "unrun_stages": ("b", "a1"),
            "device_resolution": resolution,
        },
    )

    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_prepared_run_artifacts",
        lambda *_args, **_kwargs: pytest.fail("training reached artifact evaluation"),
    )
    assert cli_train.main([str(path)]) == 2
    output = capsys.readouterr().out
    assert "Training status: no_eligible_checkpoint" in output
    for name in ("a0", "b", "a1"):
        assert f"Stage {name} run directory:" in output
    assert "Post-training artifacts: none" in output


@pytest.mark.parametrize("arm", ["b", "a1"])
def test_saved_effective_resume_requires_full_a0_gate_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str) -> None:
    """A saved-stage shortcut cannot bypass corrupt common checkpoint/handoff admission."""
    raw = configs.transient_matched_config()
    plan = configs.resolve_plan(raw, output_root=tmp_path)
    selected = {"a0": plan.a0, "b": plan.b, "a1": plan.a1}
    dirs = {name: common.paths.resolve_config_run_dir(config) for name, config in selected.items()}
    source = configs.write_yaml(tmp_path / "request.yaml", raw)
    record = identity.build_transient_experiment_record(plan.a0, plan.b, stage_a1=plan.a1, config_path=source)
    for name, config in selected.items():
        configs.write_yaml(dirs[name] / "config.yaml", dict(config))
    monkeypatch.setattr(identity, "load_readonly_transient_experiment_record", lambda *_args, **_kwargs: {"record": record, "output_root": tmp_path})
    monkeypatch.setattr(
        identity,
        "resolve_readonly_experiment_child",
        lambda _record, child, **_kwargs: dirs[next(key.removeprefix("stage_") for key, value in record["children"].items() if value == child)],
    )
    calls = []

    def gate(run_dir: Path, **kwargs: Any) -> None:
        calls.append((run_dir, kwargs["requested_b"]["run"]["name"]))
        message = "Corrupt common A0 checkpoint or immutable handoff"
        raise run_service.RunLifecycleError(message)

    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", gate)
    monkeypatch.setattr(run_service, "_run_resolved_experiment", lambda *_args, **_kwargs: pytest.fail("Training reached before A0 admission"))
    with pytest.raises(run_service.RunLifecycleError, match="common A0 checkpoint"):
        run_service.run_experiment(dirs[arm] / "config.yaml", resume=dirs[arm])
    assert calls == [(dirs["a0"], plan.b["run"]["name"])]
