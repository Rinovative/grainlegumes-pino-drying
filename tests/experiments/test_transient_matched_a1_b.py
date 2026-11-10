"""Protect explicit matched branches, durable timing evidence, and exact cursor resume.

Responsibilities:
  - Check matched branches, immutable receipts, exact sample cursors, caps, and recovery reporting
  - Check length-homogeneous matched batches, terminal zero cursors, and finalize-only terminal resume
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
from dataclasses import dataclass
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
from src.learning.training import learning_training_checkpoint as checkpoint_service
from src.learning.training import learning_training_loop as loop
from src.learning.transient import learning_transient_adapter as transient_adapter
from src.learning.transient import learning_transient_curriculum as curriculum
from src.learning.transient import learning_transient_history as transient_history
from src.learning.transient.learning_transient_handoff import HANDOFF_DIRECTORY

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class _Step:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    sample_count: int
    processed_target_transitions: int
    forward_transitions: int
    consumed_items: int


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


class _Adapter:
    gradient_accumulation_steps = 1

    def __init__(self) -> None:
        self.train_seconds = 0.0
        self.samples: list[int] = []
        self.cursors: list[int] = []

    def prepare_batch(self, raw: tuple[torch.Tensor], *, device: torch.device, training: bool) -> torch.Tensor:
        assert training
        return raw[0].to(device)

    def training_steps(self, model: torch.nn.Module, batch: torch.Tensor, loss: torch.nn.Module) -> Iterator[_Step]:
        del loss
        self.samples.append(int(batch[0, 0].item()))
        value = model(batch).square().mean()
        yield _Step(value, {"total": value}, 1, 1, 1, int(batch.shape[0]))

    def record_optimizer_work(self, work: Any) -> None:
        assert work.train_seconds is not None
        assert work.train_seconds > 0
        assert work.batch_cursor is not None
        self.train_seconds += work.train_seconds
        self.cursors.append(work.batch_cursor)

    def budget_state(self) -> dict[str, float]:
        return {"train_seconds": self.train_seconds}

    def should_stop(self) -> bool:
        return False


class _LoopAdapter(_Adapter):
    """Persist enough fake controller state to exercise the real matched loop."""

    def __init__(self, budget_groups: int = 2) -> None:
        super().__init__()
        self.epoch_index = -1
        self.budget_complete = False
        self.budget_groups = budget_groups

    def begin_epoch(self, *, epoch_index: int, total_epochs: int) -> None:
        del total_epochs
        self.epoch_index = epoch_index

    def prepare_batch(self, raw: tuple[torch.Tensor], *, device: torch.device, training: bool) -> torch.Tensor:
        del training
        return raw[0].to(device)

    def record_optimizer_work(self, work: Any) -> None:
        # Like the real controller, no optimizer work is admitted after the budget boundary.
        if self.budget_complete:
            msg = "optimizer work after the budget boundary"
            raise RuntimeError(msg)
        super().record_optimizer_work(work)
        self.budget_complete = len(self.cursors) >= self.budget_groups

    def should_stop(self) -> bool:
        return self.budget_complete

    def budget_state(self) -> dict[str, float]:
        return {"train_seconds": self.train_seconds, "planned_train_seconds": 10.0}

    def state_dict(self) -> dict[str, Any]:
        # The completion flag sits in the controller state, as in the real transient adapter.
        return {
            "epoch_index": self.epoch_index,
            "train_seconds": self.train_seconds,
            "cursors": list(self.cursors),
            "controller": {"budget_complete": self.budget_complete},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch_index = state["epoch_index"]
        self.train_seconds = state["train_seconds"]
        self.cursors = list(state["cursors"])
        self.budget_complete = state["controller"]["budget_complete"]

    def telemetry_state(self) -> dict[str, float]:
        return {"train_seconds": self.train_seconds}

    def record_validation_work(self, seconds: float) -> None:
        del seconds

    def record_within_budget_evaluation(self, metric: float, *, epoch_index: int) -> None:
        assert metric >= 0
        assert epoch_index == self.epoch_index

    def evaluates_at_budget_points(self) -> bool:
        return False

    def budget_evaluation_due(self) -> bool:
        return False

    def record_budget_evaluation(self) -> None:
        return None

    def validate_terminal_state(self, *, best_metric: float, best_epoch: int) -> None:
        assert self.budget_complete
        assert best_metric >= 0
        assert best_epoch == 1

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
        )


class _PointsLoopAdapter(_LoopAdapter):
    """Report budget points as due only on chosen epochs, like a measured B/A1 controller."""

    def __init__(self, *, budget_groups: int, due_epochs: set[int]) -> None:
        super().__init__(budget_groups=budget_groups)
        self.due_epochs = due_epochs
        self.marked: list[int] = []

    def evaluates_at_budget_points(self) -> bool:
        return True

    def budget_evaluation_due(self) -> bool:
        return self.epoch_index + 1 in self.due_epochs

    def record_budget_evaluation(self) -> None:
        self.marked.append(self.epoch_index + 1)

    def validate_terminal_state(self, *, best_metric: float, best_epoch: int) -> None:
        assert self.budget_complete
        assert best_metric >= 0
        assert best_epoch in self.marked


class _EpochBudgetLoopAdapter(_LoopAdapter):
    """Count optimizer groups like an epoch-budget A0 controller, which measures no cursor or train time."""

    def record_optimizer_work(self, work: Any) -> None:
        del work
        self.cursors.append(len(self.cursors) + 1)
        self.budget_complete = len(self.cursors) >= self.budget_groups


class _WindowedLoopAdapter(_LoopAdapter):
    """Train each loader batch as two windows; only the last consumes its items, as for RNO trajectories."""

    def training_steps(self, model: torch.nn.Module, batch: torch.Tensor, loss: torch.nn.Module) -> Iterator[_Step]:
        del loss
        for consumed in (0, int(batch.shape[0])):
            value = model(batch).square().mean()
            yield _Step(value, {"total": value}, 1, 1, 1, consumed)


class _Metric:
    id = "mse"
    space = "normalized"

    def reset(self) -> None:
        self.values: list[float] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, *, space: str, batch_index: int) -> None:
        del space, batch_index
        self.values.extend((pred - target).square().flatten().tolist())

    def compute(self) -> float:
        return sum(self.values) / len(self.values)


def _loop_identity() -> dict[str, Any]:
    def digest(name: str) -> str:
        return hashlib.sha256(name.encode()).hexdigest()

    objective = {"id": "mse", "kind": "mse", "space": "normalized", "fields": ["state"], "reduction": "element_mean", "direction": "minimize"}
    return {
        "identity_contract_sha256": checkpoint_service.TRAINING_IDENTITY_CONTRACT_DIGEST,
        "task": "transient_drying",
        "task_contract_digest": digest("task"),
        "effective_config_digest": digest("config"),
        "resume_contract_digest": digest("resume"),
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
    assert cli_train._parse_stage_selection("a1,a0", option="--train-stages", allow_policies=False) == ("a0", "a1")
    assert cli_train._parse_stage_selection("all", option="--train-stages", allow_policies=False) is None
    with pytest.raises(ValueError, match="distinct"):
        cli_train._parse_stage_selection("a0,a0", option="--train-stages", allow_policies=False)
    with pytest.raises(ValueError, match="distinct"):
        cli_train._parse_stage_selection("a2", option="--train-stages", allow_policies=False)


def test_single_stage_artifact_selection_rejected_before_training_even_with_artifact_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = configs.write_config(tmp_path, configs.transient_stage_a_only_config(), canonical_root=tmp_path)
    called = False

    def fake_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(run_service, "run_experiment", fake_run)
    assert cli_train.main([str(path), "--artifacts", "all"]) == 1
    assert not called
    assert "--artifacts all or a stage list applies only to matched_a1_b" in capsys.readouterr().err


def test_parent_identity_ignores_operational_caps_but_binds_science(tmp_path: Path) -> None:
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    original = configs.resolve_plan(configs.transient_matched_config())
    first = identity.build_transient_experiment_record(*original.stages[:2], stage_a1=original.a1, config_path=config_path)
    revised = configs.transient_matched_config()
    revised["training"]["a1"]["max_epochs"] = 10
    changed_cap = configs.resolve_plan(revised)
    second = identity.build_transient_experiment_record(*changed_cap.stages[:2], stage_a1=changed_cap.a1, config_path=config_path)
    assert first["parent_identity_sha256"] == second["parent_identity_sha256"]
    revised["training"]["b"]["train_seconds"] = 0.5
    changed_budget = configs.resolve_plan(revised)
    third = identity.build_transient_experiment_record(*changed_budget.stages[:2], stage_a1=changed_budget.a1, config_path=config_path)
    assert third["parent_identity_sha256"] != first["parent_identity_sha256"]


def test_budget_receipt_is_immutable_and_binds_exact_a0_origin(tmp_path: Path) -> None:
    b_config = configs.resolve_plan(configs.transient_matched_config()).b
    digest_a = "a" * 64
    digest_b = "b" * 64
    summary = {
        "status": "completed",
        "terminal_controller": {
            "budget_complete": True,
            "clock_kind": "train_seconds",
            "train_seconds": 0.31,
            "post_handoff_optimizer_steps": 3,
            "batch_cursor": 1,
        },
        "terminal_epoch": 2,
        "best_epoch": 1,
        "best_checkpoint_sha256": "c" * 64,
        "last_checkpoint_sha256": "d" * 64,
    }
    receipt = budget.build_b_receipt(parent_identity_sha256=digest_a, a0_handoff_sha256=digest_b, b_config=b_config, b_summary=summary)
    assert receipt["protocol"] == "matched_a1_b"
    assert receipt["clock_kind"] == "train_seconds"
    assert receipt["budget_value"] == receipt["train_seconds"] == 0.31
    path = budget.publish_b_receipt(tmp_path, receipt)
    assert budget.publish_b_receipt(tmp_path, receipt) == path
    assert budget.load_b_receipt(path, parent_identity_sha256=digest_a, a0_handoff_sha256=digest_b, b_run_name=b_config["run"]["name"]) == receipt
    with pytest.raises(ValueError, match="selected A0/B"):
        budget.load_b_receipt(path, parent_identity_sha256="e" * 64, a0_handoff_sha256=digest_b, b_run_name=b_config["run"]["name"])
    altered = copy.deepcopy(receipt)
    altered["train_seconds"] = 0.5
    altered["budget_value"] = 0.5
    altered["receipt_sha256"] = budget.common.serialization.canonical_json_sha256(
        {key: value for key, value in altered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(FileExistsError, match="differs"):
        budget.publish_b_receipt(tmp_path, altered)


def test_matched_a1_receipt_rejects_budget_value_that_differs_from_train_seconds() -> None:
    config = configs.resolve_plan(configs.transient_matched_config()).b
    summary = {
        "status": "completed",
        "terminal_controller": {
            "budget_complete": True,
            "clock_kind": "train_seconds",
            "train_seconds": 0.31,
            "post_handoff_optimizer_steps": 3,
            "batch_cursor": 1,
        },
        "terminal_epoch": 2,
        "best_epoch": 1,
        "best_checkpoint_sha256": "c" * 64,
        "last_checkpoint_sha256": "d" * 64,
    }
    receipt = budget.build_b_receipt(
        parent_identity_sha256="a" * 64,
        a0_handoff_sha256="b" * 64,
        b_config=config,
        b_summary=summary,
    )
    receipt["budget_value"] = 0.5
    receipt["receipt_sha256"] = common.serialization.canonical_json_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="must equal train_seconds"):
        budget.validate_b_receipt(receipt)


def test_matched_permutation_resumes_exact_unconsumed_samples_and_next_epoch() -> None:
    dataset = TensorDataset(torch.arange(7))
    generator = torch.Generator().manual_seed(91)
    source = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0)
    matched = loop._matched_train_loader(source)
    sampler = matched.batch_sampler
    assert isinstance(sampler, loop.MatchedPermutationSampler)
    sampler.begin_epoch(0)
    iterator = iter(matched)
    consumed = [int(next(iterator)[0].item()) for _ in range(3)]
    sampler.advance(3)
    saved_sampler = sampler.state_dict()
    assert matched.generator is not None
    saved_generator = matched.generator.get_state().clone()
    remainder = [int(batch[0].item()) for batch in iterator]
    assert sorted(consumed + remainder) == list(range(7))

    restored_generator = torch.Generator().manual_seed(0)
    restored_generator.set_state(saved_generator)
    restored_source = DataLoader(dataset, batch_size=1, shuffle=True, generator=restored_generator, num_workers=0)
    resumed = loop._matched_train_loader(restored_source)
    resumed_sampler = resumed.batch_sampler
    assert isinstance(resumed_sampler, loop.MatchedPermutationSampler)
    resumed_sampler.load_state_dict(saved_sampler)
    resumed_iterator = iter(resumed)
    assert resumed.generator is not None
    resumed.generator.set_state(saved_generator)
    assert [int(batch[0].item()) for batch in resumed_iterator] == remainder
    resumed_sampler.advance(4)
    sampler.advance(4)
    resumed_sampler.begin_epoch(1)
    sampler.begin_epoch(1)
    assert resumed_sampler.permutation == sampler.permutation


def test_fake_clock_cap_stops_after_committed_group_and_resumes_exact_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(loop.time, "perf_counter", clock)
    dataset = TensorDataset(torch.arange(4, dtype=torch.float32).reshape(-1, 1))
    source = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(12), num_workers=0)
    loader = loop._matched_train_loader(source)
    sampler = loader.batch_sampler
    assert isinstance(sampler, loop.MatchedPermutationSampler)
    sampler.begin_epoch(0)
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.001)
    adapter = _Adapter()
    first = loop._train_adapter_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.MSELoss(),
        torch.device("cpu"),
        adapter,
        max_optimizer_steps=1,
    )
    assert first["transient/train/operational_cap_reached"] == 1
    assert sampler.cursor == 1
    committed_seconds = adapter.train_seconds
    saved = sampler.state_dict()
    assert loader.generator is not None
    saved_generator = loader.generator.get_state().clone()
    resumed_source = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator(), num_workers=0)
    assert resumed_source.generator is not None
    resumed_source.generator.set_state(saved_generator)
    resumed = loop._matched_train_loader(resumed_source)
    resumed_sampler = resumed.batch_sampler
    assert isinstance(resumed_sampler, loop.MatchedPermutationSampler)
    resumed_sampler.load_state_dict(saved)
    second = loop._train_adapter_one_epoch(
        model,
        resumed,
        optimizer,
        torch.nn.MSELoss(),
        torch.device("cpu"),
        adapter,
        max_train_seconds=committed_seconds + 0.01,
    )
    assert second["transient/train/operational_cap_reached"] == 1
    assert adapter.train_seconds > committed_seconds
    assert resumed_sampler.cursor == 2
    assert adapter.cursors == [1, 2]
    assert adapter.samples == [sampler.permutation[0], sampler.permutation[1]]


def test_budget_unmet_b_stops_before_a1_and_keeps_parent_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    executed: list[str] = []

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = config["training"]["comparison_arm"]
        executed.append(name)
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True)
        status = "budget_unmet" if name == "b" else "completed"
        return {"run_dir": run_dir, "result": {"status": status}, "device_resolution": kwargs["device_resolution"]}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: None)
    outcome = run_service.run_experiment(config_path, device="cpu", output_root=tmp_path)
    assert executed == ["a0", "b"]
    assert tuple(outcome["stage_outcomes"]) == ("a0", "b")
    assert outcome["training_status"] == "budget_unmet"
    assert outcome["experiment_record"].is_file()
    assert not outcome["stage_runs"]["a1"].exists()


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
        return {"run_dir": run_dir, "result": {"status": "completed"}, "device_resolution": kwargs["device_resolution"]}

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_service, "validate_completed_run", lambda _run_dir: {"config": stages["b"], "summary": {"train_seconds": 1.0}})
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: {"test": "receipt"})
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
    assert cli_train.main([str(config_path), "--artifacts", "none"]) == expected_exit
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
    expected_command = f"python -m src.experiments.cli.cli_train {config_path} --train-stages a1 --artifacts none"
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
    validated_roles: list[str] = []

    def load_valid_checkpoint(path: Path, *, expected_role: str, **_kwargs: Any) -> dict[str, Any]:
        assert path.is_file()
        validated_roles.append(expected_role)
        return {"best_metric": 0.1, "best_epoch": 1, "completed_epoch": 1, "batch_progress": None}

    monkeypatch.setattr(checkpoint_service, "load_checkpoint", load_valid_checkpoint)
    assert run_service._matched_failed_stage_recovery(run_dir, requested, config_path=config_path)[0]
    assert validated_roles == ["last", "best"]
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
        return {"run_dir": run_dir, "result": {"status": "completed"}, "device_resolution": kwargs["device_resolution"]}

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
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: {"test": "receipt"})
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


@pytest.mark.parametrize("cap", [{"max_optimizer_steps": 1}, {"max_train_seconds": 0.5}])
def test_trailing_accumulation_group_checks_cap_and_excludes_stop_iteration_wait(monkeypatch: pytest.MonkeyPatch, cap: dict[str, Any]) -> None:
    monkeypatch.setattr(loop.time, "perf_counter", _Clock())
    dataset = TensorDataset(torch.arange(2, dtype=torch.float32).reshape(-1, 1))
    source = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(15), num_workers=0)
    loader = loop._matched_train_loader(source)
    sampler = loader.batch_sampler
    assert isinstance(sampler, loop.MatchedPermutationSampler)
    sampler.begin_epoch(0)
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.001)
    adapter = _Adapter()
    adapter.gradient_accumulation_steps = 3
    result = loop._train_adapter_one_epoch(model, loader, optimizer, torch.nn.MSELoss(), torch.device("cpu"), adapter, **cap)
    assert result["transient/train/operational_cap_reached"] == 1
    assert result["optimizer_steps"] == 1
    assert sampler.cursor == 2
    assert adapter.train_seconds == pytest.approx(0.75)


def test_train_loop_partial_cap_resumes_same_epoch_then_selects_terminal_partial_a1(tmp_path: Path) -> None:
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    config = {
        "task": "transient_drying",
        "training": {
            "epochs": 2,
            "evaluation_interval": 1,
            "ood_evaluation_interval": 1,
            "comparison_arm": "a1",
            "matched_a1_b": {"role": "a1"},
        },
        "evaluation": {"objective": _loop_identity()["objective"]},
        "tracking": {"wandb": {"monitor": {"enabled": False, "interval": 1, "max_cases": 1}}},
    }
    identity = _loop_identity()
    events: list[int] = []

    def run(*, resume: bool, max_steps: int | None) -> dict[str, Any]:
        model = torch.nn.Linear(1, 1)
        optimizer = SGD(model.parameters(), lr=0.001)
        train_loader = DataLoader(values, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(21), num_workers=0)
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
            max_steps=max_steps,
            epoch_state_callback=lambda epoch, _metrics: events.append(epoch),
            epoch_end_callback=lambda epoch, _metrics: events.append(epoch),
        )

    first = run(resume=False, max_steps=1)
    assert first["status"] == "budget_unmet"
    assert first["objective_history"] == []
    assert events == []
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["batch_progress"] == {"epoch": 1, "cursor": 1, "size": 3, "complete": False}
    # The operational partial epoch has no history record yet, so recovery replays it.
    assert checkpoint_service.recorded_history_epochs(saved) == 0
    second = run(resume=True, max_steps=None)
    assert second["status"] == "completed"
    assert second["best_epoch"] == 1
    assert [entry["epoch"] for entry in second["objective_history"]] == [1]
    assert events == [1, 1]
    terminal = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert terminal["batch_progress"]["cursor"] == 2
    assert terminal["batch_progress"]["complete"] is False
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
    assert {"next_epoch", "batch_progress", "optimizer_state_dict", "train_sampler_state_dict"}.isdisjoint(best)
    assert (best["completed_epoch"], best["best_epoch"], best["global_step"]) == (1, 1, terminal["global_step"])


@pytest.mark.parametrize(("mode", "kind"), [("matched_a1_b", "full"), ("stage_a_only", "evaluation")])
def test_only_the_matched_a0_best_keeps_its_training_state_for_handoff(tmp_path: Path, mode: str, kind: str) -> None:
    """The matched A0 best is the teacher handoff source; a stage-A-only A0 never branches and keeps evaluation state."""
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    config = {
        "task": "transient_drying",
        "training": {
            "epochs": 1,
            "evaluation_interval": 1,
            "ood_evaluation_interval": 1,
            "comparison_arm": "a0",
            "stage_schedule": {"mode": mode},
        },
        "evaluation": {"objective": _loop_identity()["objective"]},
        "tracking": {"wandb": {"monitor": {"enabled": False, "interval": 1, "max_cases": 1}}},
    }
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    result = loop.train_loop(
        config=config,
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(values, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(21), num_workers=0),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        checkpoint_identity=_loop_identity(),
        adapter=_EpochBudgetLoopAdapter(budget_groups=3),
    )

    assert result["status"] == "completed"
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
            scheduler=None,
            loss=torch.nn.MSELoss(),
            train_loader=DataLoader(values, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(5), num_workers=0),
        )
        assert restored["source_global_step"] == result["global_step"] == 3
        assert all(torch.equal(value, best["model_state_dict"][name]) for name, value in target.state_dict().items())
        assert target_optimizer.state_dict()["state"] == best["optimizer_state_dict"]["state"]


def test_budget_point_arm_validates_only_at_epochs_that_crossed_a_new_point(tmp_path: Path) -> None:
    """A measured B/A1 arm skips epoch ends without a new budget point but always validates at completion."""
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    config = {
        "task": "transient_drying",
        "training": {
            "epochs": 6,
            "evaluation_interval": 1,
            "ood_evaluation_interval": 1,
            "comparison_arm": "b",
            "matched_a1_b": {"role": "b"},
        },
        "evaluation": {"objective": _loop_identity()["objective"]},
        "tracking": {"wandb": {"monitor": {"enabled": False, "interval": 1, "max_cases": 1}}},
    }
    adapter = _PointsLoopAdapter(budget_groups=12, due_epochs={2})
    model = torch.nn.Linear(1, 1)
    result = loop.train_loop(
        config=config,
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=DataLoader(values, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(21), num_workers=0),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        checkpoint_identity=_loop_identity(),
        adapter=adapter,
    )

    assert result["status"] == "completed"
    assert [entry["epoch"] for entry in result["objective_history"]] == [2, 4]
    assert adapter.marked == [2, 4]


def _windowed_terminal_loop(
    tmp_path: Path,
    *,
    resume: bool,
    events: list[tuple[str, int]],
    adapter: _LoopAdapter | None = None,
    epochs: int = 2,
) -> tuple[dict[str, Any], torch.nn.Module]:
    """Run the real matched loop; by default its budget crosses after the first window of the first trajectory batch."""
    values = TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    config = {
        "task": "transient_drying",
        "training": {
            "epochs": epochs,
            "evaluation_interval": 1,
            "ood_evaluation_interval": 1,
            "comparison_arm": "b",
            "matched_a1_b": {"role": "b"},
        },
        "evaluation": {"objective": _loop_identity()["objective"]},
        "tracking": {"wandb": {"monitor": {"enabled": False, "interval": 1, "max_cases": 1}}},
    }
    model = torch.nn.Linear(1, 1)
    result = loop.train_loop(
        config=config,
        device=torch.device("cpu"),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.001),
        train_loader=DataLoader(values, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(21), num_workers=0),
        eval_loader=DataLoader(values, batch_size=1, shuffle=False, num_workers=0),
        train_loss=torch.nn.MSELoss(),
        eval_metrics={"mse": _Metric()},
        save_dir=tmp_path,
        resume_from=tmp_path / "last_checkpoint.pt" if resume else None,
        checkpoint_identity=_loop_identity(),
        adapter=_WindowedLoopAdapter(budget_groups=1) if adapter is None else adapter,
        epoch_state_callback=lambda epoch, _metrics: events.append(("history", epoch)),
        epoch_end_callback=lambda epoch, _metrics: events.append(("observers", epoch)),
    )
    return result, model


def test_budget_crossing_in_the_first_trajectory_batch_publishes_a_loadable_zero_cursor(tmp_path: Path) -> None:
    """
    Complete the budget after a window that consumed no loader item yet.

    The terminal checkpoints then record cursor 0; they must stay loadable for
    terminal validation and recovery, while a zero cursor without a completed
    budget would describe no committed loader progress and stays rejected. The
    terminal epoch's history record precedes the checkpoint, so recovery keeps it.
    """
    events: list[tuple[str, int]] = []
    result, _model = _windowed_terminal_loop(tmp_path, resume=False, events=events)

    assert result["status"] == "completed"
    assert result["best_epoch"] == 1
    terminal = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert terminal["batch_progress"] == {"epoch": 1, "cursor": 0, "size": 3, "complete": False}
    assert checkpoint_service.recorded_history_epochs(terminal) == events.count(("history", 1)) == 1
    open_budget = copy.deepcopy(terminal)
    open_budget["adapter_state_dict"]["controller"]["budget_complete"] = False
    torch.save(open_budget, tmp_path / "open_budget.pt")
    with pytest.raises(ValueError, match="valid epoch cursor"):
        checkpoint_service.load_checkpoint(
            tmp_path / "open_budget.pt",
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
    assert events == [("history", 1), ("observers", 1)]
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved["model_state_dict"][name])


def test_terminal_budget_at_the_epoch_cap_still_resumes_to_finalization(tmp_path: Path) -> None:
    """
    Complete the budget with the last optimizer group of the last admitted epoch.

    Its terminal checkpoint describes a complete epoch equal to training.epochs;
    the run service's epoch-cap rule must admit the finalize-only resume of an
    interrupted run, while a published budget_unmet or completed run must extend.
    """
    events: list[tuple[str, int]] = []
    _windowed_terminal_loop(tmp_path, resume=False, events=events, adapter=_LoopAdapter(budget_groups=3), epochs=1)
    saved = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    admits = run_service._epoch_cap_admits_resume

    assert saved["batch_progress"] == {"epoch": 1, "cursor": 3, "size": 3, "complete": True}
    assert admits(saved, 1, status="interrupted")
    assert admits(saved, 1, status="running")
    assert not admits(saved, 1, status="budget_unmet")
    assert not admits(saved, 1, status="completed")
    assert admits(saved, 2, status="budget_unmet")
    assert not admits(saved, 0, status="interrupted")

    resumed, model = _windowed_terminal_loop(tmp_path, resume=True, events=events, adapter=_LoopAdapter(budget_groups=3), epochs=1)

    assert resumed["status"] == "completed"
    assert events == [("history", 1), ("observers", 1)]
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved["model_state_dict"][name])


def test_matched_trajectory_batches_never_mix_lengths_and_resume_on_batch_boundaries() -> None:
    """
    Wrap a grouped trajectory loader whose items stop at two different lengths.

    Every matched batch holds one length, an epoch is an exact permutation, the
    cursor resumes exactly after committed batches, and a cursor inside a batch
    is rejected instead of silently replaying or skipping items.
    """
    lengths = (6, 5, 6, 5, 6)
    generator = torch.Generator().manual_seed(4)
    grouped = datasets.runtime.factory.GroupedBatchSampler(lengths, 2, generator)
    source = DataLoader(TensorDataset(torch.arange(5)), batch_sampler=grouped, generator=generator, num_workers=0)
    loader = loop._matched_train_loader(source)
    sampler = loader.batch_sampler
    assert isinstance(sampler, loop.MatchedPermutationSampler)
    assert len(sampler) == 3
    sampler.begin_epoch(0)
    batches = list(sampler)

    assert sorted(item for batch in batches for item in batch) == list(range(5))
    assert all(len({lengths[item] for item in batch}) == 1 and len(batch) <= 2 for batch in batches)
    assert [[int(value) for value in batch[0]] for batch in loader] == batches
    sampler.advance(len(batches[0]))
    resumed = loop.MatchedPermutationSampler(5, torch.Generator(), batch_size=2, keys=lengths)
    resumed.load_state_dict(sampler.state_dict())
    assert list(resumed) == batches[1:]
    # Three length-6 and two length-5 items always form at least one full batch of two.
    full = next(index for index, batch in enumerate(batches) if len(batch) == 2)
    inside = sampler.state_dict()
    inside["cursor"] = sum(len(batch) for batch in batches[:full]) + 1
    resumed.load_state_dict(inside)
    with pytest.raises(ValueError, match="batch boundary"):
        list(resumed)


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "interrupted"])
@pytest.mark.parametrize("unpublished_prior_attempt", [False, True])
def test_completed_summary_refreshes_prior_budget_unmet_clocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    unpublished_prior_attempt: bool,
) -> None:
    """Preserve capped attempts across completion, failure, interruption, and an attempt that was killed before publishing its duration."""
    config = configs.resolve(configs.direct_config(), output_root=tmp_path)
    run_dir = run_service.prepare_fresh_run(config, run_dir=tmp_path / "run")
    processor = SimpleNamespace(
        in_normalizer=None,
        out_normalizer=SimpleNamespace(std=torch.ones(1)),
        to=lambda _device: None,
    )
    loaders = {
        "train": [0],
        "eval": [0],
        "id_test": [0],
        "material_ood": None,
        "parameter_ood": None,
        "data_processor": processor,
        "scaling_artifact": {},
        "split_indices": {},
    }
    monkeypatch.setattr(run_service, "_steady_grid_spacing", lambda *_args: (0.003, 0.003))
    monkeypatch.setattr(run_service.config_loader, "create_dataloaders_from_config", lambda *_args, **_kwargs: loaders)
    monkeypatch.setattr(run_service.learning.models.factory, "build_model", lambda *_args, **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(run_service.learning.losses.factory, "build_training_loss", lambda *_args, **_kwargs: torch.nn.MSELoss())
    monkeypatch.setattr(run_service.learning.metrics.metrics, "build_evaluation_metrics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_service.learning.training.optim, "build_scheduler", lambda *_args: None)
    checkpoint_identity = {"effective_config_digest": "a" * 64}
    monkeypatch.setattr(run_service.learning.training.checkpoint, "build_checkpoint_identity", lambda *_args, **_kwargs: checkpoint_identity)
    monkeypatch.setattr(run_service.tracking, "build_monitor_membership", lambda *_args: None)
    objective = run_service.config_loader.get_resolved_objective(config)
    result = {
        "status": "budget_unmet",
        "train_seconds": 0.25,
        "target_train_seconds": 1.0,
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
    }
    for filename in ("best_checkpoint.pt", "last_checkpoint.pt", "split_indices.pt"):
        (run_dir / filename).touch()
    clock = [100.0]
    monkeypatch.setattr(run_service.time, "monotonic", lambda: clock[0])

    def measured_loop(**_kwargs: Any) -> dict[str, Any]:
        running = run_service.read_run_summary(run_dir)
        assert running["elapsed_seconds"] is None
        assert running["duration_coverage"] == "attempt_in_progress"
        clock[0] += 4.0 if result["status"] == "budget_unmet" else 6.0
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
    run_service.execute_prepared_run(config, run_dir=run_dir, device_resolution=resolution)
    first = run_service.read_run_summary(run_dir)
    assert first["status"] == "budget_unmet"
    assert first["elapsed_seconds"] == first["attempt_elapsed_seconds"] == 4.0
    assert first["duration_coverage"] == "complete"
    if unpublished_prior_attempt:
        # A hard kill leaves the state of the "running" transition: no published duration for that attempt.
        first.update(duration_coverage="attempt_in_progress", elapsed_seconds=None, attempt_elapsed_seconds=None)
        common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), first)

    result.update(status=terminal_status, train_seconds=1.5, global_step=2)
    error_type = RuntimeError if terminal_status == "failed" else KeyboardInterrupt
    arguments = {
        "run_dir": run_dir,
        "resume_from": run_dir / "last_checkpoint.pt",
        "restored_scaling_artifact": loaders["scaling_artifact"],
        "saved_split_indices": {},
        "device_resolution": resolution,
    }
    if terminal_status == "completed":
        returned = run_service.execute_prepared_run(config, **arguments)
        assert returned["operational_timing"]["selected_checkpoint_evaluation_seconds"] == 2.0
    else:
        with pytest.raises(error_type):
            run_service.execute_prepared_run(config, **arguments)
    summary = run_service.read_run_summary(run_dir)
    assert summary["status"] == terminal_status
    second_seconds = 8.0 if terminal_status == "completed" else 6.0
    assert summary["attempt_elapsed_seconds"] == second_seconds
    assert summary["measured_elapsed_seconds"] == second_seconds + 4.0
    assert summary["elapsed_seconds"] == (None if unpublished_prior_attempt else 4.0 + second_seconds)
    assert summary["duration_coverage"] == ("unknown_prior_attempts" if unpublished_prior_attempt else "complete")
    if terminal_status == "completed":
        assert summary["operational_timing"]["selected_checkpoint_evaluation_seconds"] == 2.0
        assert summary["effective_config_digest"] == checkpoint_identity["effective_config_digest"]
    assert [event["status"] for event in summary["status_history"]][-3:] == ["budget_unmet", "running", terminal_status]


@pytest.mark.parametrize("alias", ["train_seconds", "target_train_seconds"])
def test_completed_summary_rejects_budget_alias_that_contradicts_checkpoint_controller(
    alias: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject stale presentation clocks even when all completion/controller evidence agrees."""
    controller = curriculum.MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="train_seconds",
        config_digest="a" * 64,
        budget_control="train_seconds",
        teacher_handoff=curriculum.TeacherHandoffIdentity(
            source_run_name="a0",
            source_checkpoint_sha256="e" * 64,
            source_scaling_sha256="c" * 64,
            task_contract_sha256="d" * 64,
            tensorizer_sha256="f" * 64,
            model_kind="fno",
            input_profile="canonical_physics_complete",
        ),
        planned_train_seconds=2.0,
        train_seconds=1.89419,
    ).state_dict()
    terminal = {key: controller[key] for key in run_service._TERMINAL_CONTROLLER_KEYS}
    curriculum_state = {"active_stage": 0, "max_horizon": 1, "draw_index": 0}
    checkpoint_identity = {"effective_config_digest": "b" * 64}
    summary = {
        "checkpoint_identity": checkpoint_identity,
        "transient_scaling": {"semantic_digest": "c" * 64, "task_contract_digest": "d" * 64},
        "terminal_controller": terminal,
        "terminal_curriculum": curriculum_state,
        **run_service._terminal_budget_aliases(controller),
    }
    monkeypatch.setattr(run_service.transient_adapter_history, "validate_completed_history", lambda *_args, **_kwargs: {"epochs": []})
    monkeypatch.setattr(run_service.transient_curriculum, "validate_terminal_evidence", lambda *_args, **_kwargs: None)
    arguments = {
        "config": {"task": "transient_drying", "run": {"name": "b"}, "task_contract": {"digest": "d" * 64}, "training": {}},
        "normalizer_state": SimpleNamespace(digest="c" * 64),
        "checkpoint_identity": checkpoint_identity,
        "completed_epoch": 1,
    }
    run_service._validate_transient_summary_evidence(tmp_path, summary=summary, **arguments)
    run_service._validate_transient_terminal_state(summary, {"adapter_state_dict": {"controller": controller, "curriculum": curriculum_state}})
    summary[alias] = 0.43219
    with pytest.raises(run_service.RunLifecycleError, match="contradicts its terminal controller"):
        run_service._validate_transient_summary_evidence(tmp_path, summary=summary, **arguments)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        pytest.param("terminal_controller", "budget_complete", False, id="incomplete-budget"),
        pytest.param("terminal_controller", "post_handoff_optimizer_steps", None, id="null-counter"),
        pytest.param("terminal_curriculum", "draw_index", None, id="null-curriculum"),
        pytest.param("terminal_controller", "wall_seconds", 0.75, id="disagrees-with-history"),
    ],
)
def test_completed_transient_terminal_values_are_admitted_against_their_final_history_record(
    section: str, key: str, value: Any, tmp_path: Path
) -> None:
    """A trial keeps no last checkpoint, so its completed terminal values must be valid and equal the final epoch record."""
    controller = curriculum.MatchedComputeController(
        arm="A0", stage="stage_a_teacher_forcing", clock_kind="optimizer_steps", config_digest="a" * 64, planned_stage_epochs=1
    )
    controller.record_completed_work(
        optimizer_device_seconds=None,
        microbatches=1,
        processed_target_transitions=4,
        forward_transitions=4,
        wall_seconds=0.5,
        epoch_index=0,
        microbatch_index=0,
    )
    controller.record_validation_work(0.25)
    controller.record_within_budget_evaluation(0.5, epoch_index=0)
    rollout = curriculum.RolloutCurriculumState.create(curriculum.RolloutCurriculum(lengths=(1,), milestone_fractions=(0.0,)), seed=0)
    # The adapter's own telemetry names the values that the training loop writes into each epoch record.
    stand_in = SimpleNamespace(spec=SimpleNamespace(controller=controller, stage=controller.stage), curriculum_state=rollout)
    telemetry = transient_adapter.TransientTrainingAdapter.telemetry_state(cast("transient_adapter.TransientTrainingAdapter", stand_in))
    checkpoint_identity = {"effective_config_digest": "b" * 64}
    record_epoch = transient_history.make_epoch_state_callback(
        tmp_path, task="transient_drying", run_name="a0", checkpoint_identity=checkpoint_identity
    )
    record_epoch(1, {f"transient/{name}": float(item) for name, item in telemetry.items() if item is not None})
    state = controller.state_dict()
    summary = {
        "checkpoint_identity": checkpoint_identity,
        "transient_scaling": {"semantic_digest": "c" * 64, "task_contract_digest": "d" * 64},
        "terminal_controller": {name: state[name] for name in run_service._TERMINAL_CONTROLLER_KEYS},
        "terminal_curriculum": {name: getattr(rollout, name) for name in run_service._TERMINAL_CURRICULUM_KEYS},
    }
    arguments = {
        "config": {"task": "transient_drying", "run": {"name": "a0"}, "task_contract": {"digest": "d" * 64}},
        "normalizer_state": SimpleNamespace(digest="c" * 64),
        "checkpoint_identity": checkpoint_identity,
        "completed_epoch": 1,
    }
    run_service._validate_transient_summary_evidence(tmp_path, summary=summary, **arguments)

    summary[section][key] = value
    with pytest.raises(run_service.RunLifecycleError, match="terminal evidence"):
        run_service._validate_transient_summary_evidence(tmp_path, summary=summary, **arguments)


def test_artifact_all_reports_unfinished_stages_without_masking_budget_unmet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)
    a0, b, a1 = (tmp_path / name for name in ("a0", "b", "a1"))
    b.mkdir()
    configs.write_yaml(common.paths.resolve_run_config_path(b), {"training": {"epochs": 2}})
    calls: list[Path] = []
    resolution = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(
        run_service,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": b,
            "result": {"status": "budget_unmet", "train_seconds": 0.25, "target_train_seconds": 1.0, "global_step": 1, "completed_epoch": 1},
            "stage_runs": {"a0": a0, "b": b, "a1": a1},
            "stage_outcomes": {"a0": {"result": {"status": "completed"}}, "b": {"result": {"status": "budget_unmet"}}},
            "device_resolution": resolution,
        },
    )

    def validate(stage_dir: Path) -> dict[str, Any]:
        if stage_dir == a0:
            return {"status": "completed"}
        msg = "not completed"
        raise run_service.RunLifecycleError(msg)

    monkeypatch.setattr(run_service, "validate_completed_run", validate)
    monkeypatch.setattr(run_service, "read_run_summary", lambda _stage_dir: {"status": "budget_unmet"})
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)
    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_or_build_run_artifacts",
        lambda stage_dir, **_kwargs: (
            calls.append(stage_dir) or SimpleNamespace(role_actions={"id": "reused"}, loaded_run=SimpleNamespace(role_artifacts=dict))
        ),
    )
    assert cli_train.main([str(path), "--artifacts", "all"]) == 2
    assert calls == [a0]
    output = capsys.readouterr().out
    assert "Training status: budget_unmet" in output
    assert "Stage b artifacts: unavailable" in output
    assert "Stage a1 artifacts: unavailable" in output
    assert "Post-training artifacts: partial" in output
    assert "RESOURCE_OPTIONS" not in output
    assert "NEW_EPOCH_CAP" not in output


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
