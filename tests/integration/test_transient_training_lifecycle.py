"""Exercise a compact transient package-to-runtime training boundary.

Responsibilities:
  - Exercise compact transient packages, scaling, A0/B/A1 lifecycle, A0 budget identity, and model optimizer paths
  - Exercise best-checkpoint payload kinds and the refusal of non-completed B runs for evaluation
  - Exercise one real A0 Optuna trial whose study signature equals the dry run's and whose trial and history validate

Design principles:
  - Persisted PT data and explicit conditioning survive canonical source removal

This module does NOT:
  - Train production campaigns or replace focused unit contracts
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
import yaml
from support import configs, optuna_trials, run_state
from support.generation_fixtures import successful_source_case
from torch.optim.adamw import AdamW

from src import analysis, common, datasets, domain, experiments, generation, learning
from src.datasets.packages import dataset_packages_selection as selection
from src.experiments import experiments_wandb_schema as wandb_schema
from src.learning.inference import learning_inference_transient as transient_inference
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient import learning_transient_adapter as transient_adapter
from src.learning.transient import learning_transient_handoff as handoff
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum, RolloutCurriculumState

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration
_LR_SCALE = 0.5
_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA integration lifecycle requires an available CUDA device.",
)


@pytest.mark.usefixtures("_dataset_producer_evidence")
def test_steady_selection_reuses_matching_sources_and_rejects_fixed_conditioning_mismatch(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Keep package-fixed steady physics equal across independently authored Sources."""
    storage = tmp_path / "scientific-storage"
    selector_keys = (
        "campaign_name",
        "campaign_digest",
        "material_family",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "case_hdf5_sha256",
    )

    def admitted_selector(*, count: int, changed_fixed_value: bool = False) -> dict[str, str]:
        config_path, _ = generation_config_factory(
            simulation_profile="steady_flow",
            executable=fake_comsol,
            natural_count=count,
        )
        if changed_fixed_value:
            common_path = config_path.parent / "common.yaml"
            common_config = yaml.safe_load(common_path.read_text(encoding="utf-8"))
            common_config["scientific_fixed_values"]["T_flow_ref"]["value"] = 301.65
            common_path.write_text(yaml.safe_dump(common_config, sort_keys=False), encoding="utf-8")
            authored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            authored["revision"] = "warm-flow"
            config_path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
        campaign = generation.cases.config.load_campaign_config(config_path)
        generation.lifecycle.create_campaign(campaign, storage_root=storage)
        batch = campaign.batches[0]
        inputs = generation.cases.input_generation.generate_input_cases(batch, len(batch.case_indices), storage_root=storage)
        successful_source_case(campaign.source_name, batch, batch.case_indices[0], storage, tmp_path / "work")
        run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
        run_dir = generation.publication.campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = generation.campaign._new_campaign_manifest(
            campaign,
            run_id=run_id,
            requested_commit="a" * 40,
            run_directory=run_dir,
            scheduler_log_directory=tmp_path / "logs",
            storage_root=storage,
            input_sources={
                batch.batch_name: {
                    "input_generation_id": inputs.input_generation_id,
                    "source_git_commit": "a" * 40,
                    "case_indices": list(batch.case_indices),
                }
            },
        )
        common.serialization.atomic_write_json(
            generation.publication.campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), run_manifest
        )
        rows = generation.publication.catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)
        successful = next(row for row in rows if row["outcome"] == "successful")
        return {key: successful[key] for key in selector_keys}

    first = admitted_selector(count=1)
    second = admitted_selector(count=2)
    changed = admitted_selector(count=2, changed_fixed_value=True)

    def request(name: str, sources: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "schema_kind": "dataset_selection_request",
            "source_task": "steady_flow",
            "dataset_name": name,
            "dataset_view": "steady_flow",
            "selection": {"kind": "exact"},
            "source_cases": sources,
        }

    matching = request("steady_matching_sources", [first, second])
    published = datasets.packages.build_dataset_selection(matching, storage_root=storage)
    assert published["status"] == "complete"
    reused = datasets.packages.build_dataset_selection(matching, storage_root=storage)
    assert reused["status"] == "reused"
    assert reused["dataset_id"] == published["dataset_id"]
    with pytest.raises(ValueError, match="incompatible package-fixed stationary conditioning"):
        datasets.packages.build_dataset_selection(
            request("steady_mismatched_sources", [first, changed]),
            storage_root=storage,
        )


def _assert_item_equal(left: Any, right: Any) -> None:
    """Require exact physical parity through one nested transient item."""
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_item_equal(left[key], right[key])
    else:
        assert left == right


def _close_loader_dataset(loader: Any) -> None:
    """Close one test-owned transient Dataset behind a DataLoader."""
    loader.dataset.close()


def test_transient_package_pt_runtime_and_scaling_lifecycle(
    transient_package: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep package provenance, PT equivalence, roles, scaling, and curriculum connected."""
    root = transient_package["root"]
    id_dataset_id = transient_package["train_dataset"]
    one_step = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    rollout = datasets.contracts.transient.TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=1, window_offset=0)
    index_path = common.paths.resolve_dataset_path("transient_drying", id_dataset_id, dataset_root=root / "02_datasets")
    canonical = datasets.runtime.transient.TransientPhysicalDataset(index_path, sampling=one_step, source_root=root)
    sharded = datasets.runtime.transient.TransientPTShardDataset(index_path, sampling=one_step, source_root=root)
    sampled_canonical = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=one_step,
        source_root=root,
        spatial_stride=3,
    )
    sampled_sharded = datasets.runtime.transient.TransientPTShardDataset(
        index_path,
        sampling=one_step,
        source_root=root,
        spatial_stride=3,
    )
    window = datasets.runtime.transient.TransientPTShardDataset(index_path, sampling=rollout, source_root=root)
    try:
        _assert_item_equal(canonical[0], sharded[0])
        sampled_hdf5_item = sampled_canonical[0]
        sampled_pt_item = sampled_sharded[0]
        _assert_item_equal(sampled_hdf5_item, sampled_pt_item)
        assert sampled_hdf5_item["state"].shape[-2:] == (6, 6)
        assert sampled_hdf5_item["static"].shape[-2:] == (6, 6)
        assert sampled_hdf5_item["target"].shape[-2:] == (6, 6)
        assert sampled_hdf5_item["metadata"]["spatial_y_indices"] == (0, 3, 6, 9, 12, 15)
        assert sampled_hdf5_item["metadata"]["spatial_x_indices"] == (0, 3, 6, 9, 12, 15)
        assert sampled_hdf5_item["metadata"]["spatial_index_identity_sha256"] == sampled_pt_item["metadata"]["spatial_index_identity_sha256"]
        assert sharded.storage_backend == "pt_shards"
        assert window[0]["target"].shape[0] == 2
    finally:
        canonical.close()
        sharded.close()
        sampled_canonical.close()
        sampled_sharded.close()
        window.close()
    receipt = datasets.packages.transient_shards.load_transient_shard_receipt(
        id_dataset_id,
        storage_root=str(root),
        validation_depth="evidence",
    )
    manifest = datasets.packages.load_package_manifest("transient_drying", id_dataset_id, storage_root=root)
    assert receipt["schema_kind"] == "transient_dataset_pt_shard_receipt"
    assert receipt["publication_identity"] == {
        "dataset_id": id_dataset_id,
        "dataset_digest": manifest["dataset_digest"],
        "index_digest": receipt["index_digest"],
    }
    receipt_path = datasets.packages.transient_shards.transient_shard_directory(id_dataset_id, storage_root=root) / "receipt.json"
    assert manifest["derived_payload_sha256"] == common.serialization.file_sha256(receipt_path)
    assert receipt["transition_count"] >= 2
    tensorizer = TransientTensorizerSpec(input_profile="canonical_physics_complete", temporal_conditioning=TemporalConditioningSpec("none"))
    loaders = datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=id_dataset_id,
        material_ood_dataset_ids=[],
        tensorizer=tensorizer,
        train_sampling=one_step,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0),
        model_sha256="a" * 64,
        storage_root=str(root),
        train_subset=None,
    )
    try:
        assert loaders.data_processor.spatial_shape == (16, 16)
        assert loaders.split["roles"]["train"]["case_ids"] == transient_package["case_partition"]["train"]
        assert loaders.split["roles"]["id_test"]["case_ids"] == transient_package["case_partition"]["id_test"]
        assert loaders.material_ood is None
        assert loaders.split["dataset_identity"]["train"]["dataset_id"] == id_dataset_id
    finally:
        for value in (loaders.train, loaders.evaluation, loaders.material_ood, loaders.parameter_ood, loaders.id_test):
            if value is not None:
                _close_loader_dataset(value)
    curriculum = RolloutCurriculum(lengths=(2, 4), milestone_fractions=(0.0, 0.2))
    curriculum_state = RolloutCurriculumState.create(curriculum, seed=9)
    horizon, origins = curriculum_state.select(progress=0.2, available_length=4, batch_size=2)
    assert horizon in (2, 4)
    assert origins.shape == (2,)
    assert curriculum_state.active_stage == 1
    assert curriculum_state.max_horizon == 4
    assert domain.tasks.registry.get_task("transient_drying").id == "transient_drying"

    campaign = transient_package["source_campaign"]
    selectors = transient_package["source_selectors"]
    target_counts = {"lentil": {"train": 2, "validation": 1, "test": 1}}  # one of the two test cases stays out
    ranked = list(selection.select_ordered_cases("transient_drying", [campaign], target_counts, storage_root=root))
    method = {"kind": "design_order", "source_names": [campaign], "target_counts": target_counts}
    ranked_request = {
        "schema_kind": "dataset_selection_request",
        "source_task": "transient_drying",
        "dataset_name": "transient_ranked_lifecycle",
        "dataset_view": "transient_drying",
        "selection": method,
        "source_cases": ranked,
    }
    published_ranked = datasets.packages.build_dataset_selection(ranked_request, storage_root=root)
    assert published_ranked["status"] == "complete"
    assert datasets.packages.load_package_manifest("transient_drying", published_ranked["dataset_id"], storage_root=root)["selection"] == method
    excluded = next(reference for reference in selectors if reference not in ranked)
    substituted = [*ranked[:-1], excluded]
    for position, change in enumerate(
        (
            {"source_cases": list(reversed(ranked))},
            {"source_cases": substituted},
            {"selection": {**method, "target_counts": {"lentil": {"train": 1, "validation": 1, "test": 1}}}},
        )
    ):
        # A fresh name reaches the ordered-claim check instead of the name-binding guard.
        claimed = {**ranked_request, **change, "dataset_name": f"transient_ranked_false_claim_{position}"}
        with pytest.raises(ValueError, match="Design-order Dataset source cases disagree"):
            datasets.packages.build_dataset_selection(claimed, storage_root=root)
    monkeypatch.setattr(
        selection, "select_ordered_cases", lambda *_args, **_kwargs: pytest.fail("immutable reuse recomputed the changed source pool")
    )
    reused_ranked = datasets.packages.build_dataset_selection(ranked_request, storage_root=root)
    assert reused_ranked["dataset_id"] == published_ranked["dataset_id"]
    assert reused_ranked["status"] == "reused"


def _authored_fno_plan(
    *,
    train_dataset: str,
    destination: Path,
    device: str = "cuda",
    train_seconds: float = 1.0e-6,
    lr_scale: float = 1.0,
    scheduler: bool = False,
) -> Path:
    """Write one compact test-owned authored FNO A0/B/A1 plan."""
    raw = configs.transient_matched_config(model_kind="fno", seed=9)
    raw["run"].update({"device": device, "deterministic": False})
    raw["data"].update(
        {
            "train_dataset": train_dataset,
            "material_ood_datasets": [],
            "batch_size": 2,
        }
    )
    raw["model"]["params"].update({"n_modes": [8, 8], "hidden_channels": 8, "n_layers": 1, "lifting_channel_ratio": 1, "projection_channel_ratio": 1})
    if not scheduler:
        raw["scheduler"] = None
    raw["tracking"]["wandb"]["mode"] = "disabled"
    raw["training"]["a0"]["max_epochs"] = 1
    raw["training"]["b"].update(
        {
            "max_epochs": 2,
            "train_seconds": train_seconds,
            "lr_scale": lr_scale,
            "sampling": {"mode": "rollout_window", "rollout_length": 2, "window_stride": 1, "window_offset": 0},
            "curriculum": {"lengths": [2], "milestone_fractions": [0.0]},
        }
    )
    raw["training"]["a1"].update({"max_epochs": 64, "lr_scale": lr_scale})
    return configs.write_config(destination.parent / destination.stem, raw)  # a canonical tree below the destination stem


def _assert_curated_series_are_produced(run_dir: Path, *, cuda: bool) -> None:
    """Find every curated W&B source key in the stage's local history and every projected ``selected/`` key in its selected metrics."""
    config = experiments.run.validate_completed_run(run_dir)["config"]
    evaluation = config["evaluation"]
    definitions = experiments.tracking.automatic_history_metric_definitions(
        evaluation["metrics"],
        objective_id=str(evaluation["objective"]["id"]),
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        cuda_enabled=cuda,
        task_id="transient_drying",
        state_aux_enabled=float(config["loss"]["data"].get("state_aux_weight", 0.0)) > 0.0,
        ood_roles=datasets.preprocessing.splits.read_saved_split(run_dir).ood_roles,
    )
    records = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))["epochs"]
    produced = set().union(*records)
    selected = experiments.run.read_run_summary(run_dir)["selected_metrics"]
    assert {definition.source_key for definition in definitions} <= produced
    selected_keys = {wandb_schema.selected_summary_key(definition.source_key) for definition in definitions} - {None}
    assert selected_keys <= set(selected)


@_CUDA_REQUIRED
def test_transient_a0_b_a1_lifecycle(transient_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one public CUDA FNO A0/B/A1 lifecycle from persisted source cases."""
    root = transient_package["root"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    plan_path = _authored_fno_plan(
        train_dataset=transient_package["train_dataset"],
        destination=root / "technical_fno.yaml",
    )

    outcome = experiments.run.run_experiment(plan_path, device="cuda", output_root=root / "03_experiments")
    a0_dir = outcome["stage_runs"]["a0"]
    b_dir = outcome["stage_runs"]["b"]
    a1_dir = outcome["stage_runs"]["a1"]
    a0 = experiments.run.validate_completed_run(a0_dir)
    b = experiments.run.validate_completed_run(b_dir)
    a1 = experiments.run.validate_completed_run(a1_dir)
    a_summary = experiments.run.read_run_summary(a0_dir)
    b_summary = experiments.run.read_run_summary(b_dir)
    a1_summary = experiments.run.read_run_summary(a1_dir)
    resolution = outcome["device_resolution"]

    assert outcome["run_dir"] == a1_dir
    # Each stage is its own leaf below the shared branch directory.
    assert (a0_dir.name, b_dir.name, a1_dir.name) == ("a0", "b", "a1")
    assert (a0["best_checkpoint"]["payload_kind"], b["best_checkpoint"]["payload_kind"], a1["best_checkpoint"]["payload_kind"]) == (
        "full",
        "evaluation",
        "evaluation",
    )
    assert (a0_dir / "history.json").is_file()
    manifest = handoff.validate_stage_a_handoff(
        a0_dir / handoff.HANDOFF_DIRECTORY,
        target_config=b["config"],
        device=resolution.as_dict(),
        expected_source_run_name=a0["config"]["run"]["name"],
    )
    assert manifest["model_kind"] == "fno"
    assert b_summary["status"] == "completed"
    assert a_summary["seed_plan"] == b_summary["seed_plan"] == a1_summary["seed_plan"]
    assert a_summary["seed_plan"]["curriculum_a0"] != a_summary["seed_plan"]["curriculum_b"]
    assert b_summary["terminal_controller"]["budget_control"] == "train_seconds"
    assert b_summary["terminal_controller"]["budget_complete"] is True
    assert b_summary["terminal_controller"]["train_seconds"] > 0
    assert b_summary["terminal_controller"]["train_seconds"] >= b_summary["terminal_controller"]["planned_train_seconds"]
    assert b_summary["terminal_controller"]["post_handoff_optimizer_steps"] == 1
    assert b_summary["terminal_controller"]["best_within_budget_metric"] == b_summary["best_metric"]
    assert b_summary["terminal_controller"]["best_within_budget_epoch"] + 1 == b_summary["best_epoch"]
    assert b_summary["global_step"] == b_summary["terminal_controller"]["post_handoff_optimizer_steps"]
    assert b_summary["global_step"] == 1
    assert a_summary["global_step"] >= 1
    assert b_summary["terminal_curriculum"]["draw_index"] > 0
    assert b_summary["terminal_curriculum"]["max_horizon"] == 2
    assert len(json.loads((b_dir / "history.json").read_text(encoding="utf-8"))["epochs"]) == 1
    assert (b_dir / "teacher_handoff_manifest.json").is_file()
    assert (b_dir / "b_train_budget_receipt.json").is_file()
    assert a1_summary["status"] == "completed"
    assert a1_summary["terminal_controller"]["budget_control"] == "train_seconds"
    assert a1_summary["terminal_controller"]["budget_complete"] is True
    assert a1_summary["terminal_controller"]["train_seconds"] >= b_summary["terminal_controller"]["train_seconds"]
    assert a1_summary["terminal_controller"]["matched_receipt_sha256"] is not None
    for stage_dir in (a0_dir, b_dir, a1_dir):
        _assert_curated_series_are_produced(stage_dir, cuda=True)
    assert a1["config"]["training"]["teacher_handoff"]["source_run_name"] == a0["config"]["run"]["name"]
    assert b["config"]["training"]["teacher_handoff"]["source_run_name"] == a0["config"]["run"]["name"]
    _assert_completed_inference_and_tracking_lifecycle(
        transient_package,
        monkeypatch,
        a0_dir=a0_dir,
        config=a0["config"],
    )


def test_a0_optuna_trial_records_the_dry_run_signature_and_validates_its_history(
    transient_package: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-trial A0 study binds the Dataset evidence of its dry run; its trial is admitted with its epoch history."""
    monkeypatch.setenv("STORAGE_ROOT", str(transient_package["root"]))
    request = configs.optuna_transient_a0_config()
    request["study"].update(n_trials=1, pruner={"kind": "none"})
    request["experiment"]["data"]["train_dataset"] = transient_package["train_dataset"]
    request["experiment"]["training"]["epochs"] = 1
    request["experiment"]["training"]["stage_schedule"]["total_epochs"] = 1
    request["search_space"] = {"optimizer.lr": {"name": "learning_rate", "kind": "float", "low": 1.0e-3, "high": 2.0e-3, "log": True}}

    completed = optuna_trials.run_one_trial_study(request, directory=tmp_path)

    (completed["run_dir"] / "history.json").unlink()
    with pytest.raises(experiments.run.RunLifecycleError, match="history"):
        experiments.run.validate_completed_run(completed["run_dir"])


def _saved_learning_rates(run_dir: Path) -> tuple[float, float]:
    """Return the optimizer group rate and the scheduler's last-rate record saved in one leaf's last checkpoint."""
    checkpoint = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    return float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]), float(checkpoint["scheduler_state_dict"]["_last_lr"][0])


def _assert_evaluation_refused(run_dir: Path, status: str) -> None:
    """Refuse a really published non-completed run for execution and saved-artifact evaluation."""
    assert experiments.run.read_run_summary(run_dir)["status"] == status
    resume_message = rf"{status}.*Resume the run to completion"
    with pytest.raises(experiments.run.RunLifecycleError, match=resume_message):
        experiments.run.validate_evaluable_run(run_dir)
    with pytest.raises(experiments.run.RunLifecycleError, match=resume_message):
        analysis.evaluation.artifact_loader.load_run_artifacts(run_dir)


def _interrupt_training(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for the training loop and raise the interruption a user signal produces."""
    message = "synthetic interruption"
    raise KeyboardInterrupt(message)


def test_matched_b_scales_the_restored_learning_rate_once(transient_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A fresh B starts from A0's restored learning rate times lr_scale; resuming that B never scales it again.

    The matched A0 handoff best keeps full training state. B reaches its epoch cap
    without its train-time budget and publishes an evaluation-only best; the
    interrupted and the budget_unmet publication of B are both refused for evaluation.
    """
    root = transient_package["root"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    plan_path = _authored_fno_plan(
        train_dataset=transient_package["train_dataset"],
        destination=root / "scaled_fno.yaml",
        device="cpu",
        train_seconds=3600.0,
        lr_scale=_LR_SCALE,
        scheduler=True,
    )
    output_root = root / "03_experiments"
    a0_dir = experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("a0",))["run_dir"]
    teacher = torch.load(a0_dir / handoff.HANDOFF_DIRECTORY / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    assert teacher["payload_kind"] == "full"
    restored_rate = float(teacher["optimizer_state_dict"]["param_groups"][0]["lr"])
    _assert_curated_series_are_produced(a0_dir, cuda=False)  # the CPU run of a real A0 checks the producers where no GPU is needed

    first = experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("b",), max_steps=1)
    b_dir = first["run_dir"]

    assert first["result"]["status"] == "budget_unmet"
    assert first["result"]["global_step"] == 1
    assert _saved_learning_rates(b_dir) == pytest.approx((restored_rate * _LR_SCALE,) * 2)
    # B counts its own work from zero; A0's committed groups are not carried across the handoff.
    first_work = experiments.run.read_run_summary(b_dir)["training_work"]
    assert first_work["optimizer_steps"] == 1
    assert first_work["gradient_predictions"] > 0
    with monkeypatch.context() as drift:
        # An A0 trained by other training code is no longer a valid teacher for continuing B.
        run_state.drift_training_code(drift, "transient_drying")
        before_drift = {path: path.read_bytes() for path in sorted(b_dir.rglob("*")) if path.is_file()}
        with pytest.raises(ValueError, match=r"A0 .*: training_transient_drying implementation changed"):
            experiments.run.run_experiment(b_dir / "config.yaml", resume=b_dir, max_steps=2)
        assert {path: path.read_bytes() for path in sorted(b_dir.rglob("*")) if path.is_file()} == before_drift
    with monkeypatch.context() as interruption:
        interruption.setattr(learning.training.loop, "train_loop", _interrupt_training)
        with pytest.raises(KeyboardInterrupt):
            experiments.run.run_experiment(b_dir / "config.yaml", resume=b_dir, max_steps=2)
    _assert_evaluation_refused(b_dir, "interrupted")
    resumed = experiments.run.run_experiment(b_dir / "config.yaml", resume=b_dir, max_steps=2)

    assert resumed["result"]["status"] == "budget_unmet"
    assert resumed["result"]["global_step"] == 2
    resumed_work = experiments.run.read_run_summary(b_dir)["training_work"]
    assert resumed_work["optimizer_steps"] == 2
    assert resumed_work["gradient_predictions"] > first_work["gradient_predictions"]
    assert _saved_learning_rates(b_dir) == pytest.approx((restored_rate * _LR_SCALE,) * 2)
    # One batch per epoch: the resumed second step ends at B's epoch cap, which validates and selects a best.
    assert resumed["result"]["best_epoch"] == 2
    best = torch.load(b_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    assert best["payload_kind"] == "evaluation"
    assert "adapter_state_dict" in best
    assert "optimizer_state_dict" not in best
    _assert_evaluation_refused(b_dir, "budget_unmet")


def _resolved_direct_model_config(
    root: Path,
    *,
    train_dataset: str,
    model_kind: str,
) -> dict[str, Any]:
    """Derive and shrink a test-owned architecture-specific single-stage A0 config for one update."""
    raw = configs.transient_stage_a_only_config(model_kind=model_kind, seed=9)
    raw["training"]["epochs"] = raw["training"]["stage_schedule"]["total_epochs"] = 1
    resolved = configs.resolve(raw)
    resolved["run"].update({"device": "cuda", "deterministic": False})
    resolved["data"].update(
        {
            "train_dataset": train_dataset,
            "material_ood_datasets": [],
            "batch_size": 1,
        }
    )
    resolved["scheduler"] = None
    resolved["tracking"]["wandb"]["mode"] = "disabled"
    if model_kind == "uno":
        resolved["model"]["params"].update(
            {
                "modes_x": 8,
                "modes_y": 8,
                "hidden_channels": 4,
                "n_layers": 5,
                "uno_scalings": [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            }
        )
    elif model_kind == "rno":
        resolved["temporal"]["sampling"] = {"mode": "trajectory", "rollout_length": 2}
        resolved["training"].update(
            {
                "curriculum": {"lengths": [2], "milestone_fractions": [0.0]},
            }
        )
        resolved["model"]["params"].update({"n_modes": [8, 8], "hidden_channels": 4, "n_layers": 1})
    else:
        msg = f"Unsupported direct lifecycle model kind {model_kind!r}."
        raise ValueError(msg)
    resolved["paths"].update(
        {
            "storage_root": str(root),
            "dataset_root": str(common.paths.get_datasets_root(storage_root=root)),
            "output_root": str(root / "03_experiments"),
        }
    )
    return resolved


def _run_direct_cuda_update(config: dict[str, Any], *, root: Path) -> torch.nn.Module:
    tensorizer = TransientTensorizerSpec(
        input_profile="canonical_physics_complete",
        temporal_conditioning=TemporalConditioningSpec("none"),
    )
    sampling = datasets.contracts.transient.TransientSamplingSpec.from_mapping(config["temporal"]["sampling"])
    loaders = datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=str(config["data"]["train_dataset"]),
        material_ood_dataset_ids=config["data"]["material_ood_datasets"],
        train_subset=None,
        tensorizer=tensorizer,
        train_sampling=sampling,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0),
        model_sha256="a" * 64,
        storage_root=str(root),
    )
    device = learning.device.resolve_device("cuda").device
    try:
        adapter = transient_adapter.build_transient_training_adapter(
            config,
            scaling=loaders.data_processor,
            device=device,
            curriculum_seed=9,
        )
        raw_batch = next(iter(loaders.train))
        batch = adapter.prepare_batch(raw_batch, device=device, training=True)
        model = learning.models.factory.build_model(config, device=device)
        loss = learning.losses.factory.build_training_loss(config, device=device)
        optimizer = AdamW(model.parameters(), lr=1.0e-3)
        before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
        step = next(adapter.training_steps(model, batch, loss))
        optimizer.zero_grad(set_to_none=True)
        step.loss.backward()
        optimizer.step()
        window = batch.rollout_length if adapter.spec.trajectory_window is None else adapter.spec.trajectory_window
        assert torch.isfinite(step.loss).item()
        assert step.processed_target_transitions == batch.batch_size * window
        assert any(not torch.equal(previous, current.detach()) for previous, current in zip(before, model.parameters(), strict=False))
        return model
    finally:
        for value in (loaders.train, loaders.evaluation, loaders.material_ood, loaders.parameter_ood, loaders.id_test):
            if value is not None:
                _close_loader_dataset(value)


@_CUDA_REQUIRED
def test_transient_uno_and_rno_cuda_optimizer_lifecycle(transient_package: dict[str, Any]) -> None:
    """Update real UNO and official single-step recurrent RNO parameters on CUDA."""
    root = transient_package["root"]
    shared = {"train_dataset": transient_package["train_dataset"]}
    uno = _resolved_direct_model_config(root, **shared, model_kind="uno")
    _run_direct_cuda_update(uno, root=root)
    assert "rno" in learning.models.factory.available_model_kinds()
    rno = _resolved_direct_model_config(root, **shared, model_kind="rno")
    _run_direct_cuda_update(rno, root=root)


def _assert_completed_inference_and_tracking_lifecycle(
    transient_package: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    a0_dir: Path,
    config: dict[str, Any],
) -> None:
    """Check inference and observer mapping from the just-completed A0 run."""
    root = transient_package["root"]
    context = transient_inference.load_transient_inference_context(run_dir=a0_dir, device=learning.device.resolve_device("cuda").device)
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=1, window_offset=0)
    index_path = common.paths.resolve_dataset_path("transient_drying", transient_package["train_dataset"], dataset_root=root / "02_datasets")
    dataset = datasets.runtime.transient.TransientPTShardDataset(index_path, sampling=sampling, source_root=root)
    try:
        item = dataset[0]
    finally:
        dataset.close()
    state = item["state"].unsqueeze(0).to(context.device)
    static = item["static"].unsqueeze(0).to(context.device)
    boundary = item["boundary"].unsqueeze(0).to(context.device)
    scalars = item["scalars"].unsqueeze(0).to(context.device)
    time = item["time"]
    t_n = time["t_n"].unsqueeze(0).to(context.device)
    t_next = time["t_n_plus_1"].unsqueeze(0).to(context.device)
    dt = time["dt"].unsqueeze(0).to(context.device)
    step = transient_inference.predict_transient_step(
        context, state=state, static=static, boundary=boundary, scalars=scalars, t_n=t_n, t_next=t_next, dt=dt
    )
    rollout = transient_inference.rollout_transient_autonomous(
        context, state=state, static=static, boundary=boundary, scalars=scalars, t_n=t_n, t_next=t_next, dt=dt
    )
    assert step.next_state.shape == state.shape
    assert rollout.states.shape == (1, 2, 4, 16, 16)

    monkeypatch.setattr(experiments.tracking.importlib, "import_module", lambda _name: pytest.fail("disabled observer imported SDK"))
    disabled = experiments.tracking.initialize_wandb(config, run_dir=root / "disabled_tracking", ood_roles=())
    disabled.log_epoch(1, {"train/loss_total": 1.0})
    assert not disabled.enabled
    assert not (root / "disabled_tracking").exists()

    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict[str, float | int], int]] = []

        def log(self, values: dict[str, float | int], *, step: int) -> None:
            self.logged.append((dict(values), step))

    definitions = experiments.tracking.automatic_history_metric_definitions(
        config["evaluation"]["metrics"],
        objective_id=str(config["evaluation"]["objective"]["id"]),
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        cuda_enabled=True,
        task_id="transient_drying",
    )
    fake_run: Any = FakeRun()
    observer = experiments.tracking.WandbSession(
        fake_run,
        None,
        "normalized_drying_group_macro_rmse",
        "minimize",
        frozenset(),
        "offline",
        "fake",
        root,
        "fake",
        "transient_drying",
        {},
        history_metric_definitions=definitions,
    )
    observer.log_epoch(1, {"train/loss_total": 1.25, "transient/curriculum_max_horizon": 4.0, "ignored": 9.0})
    payload, epoch = fake_run.logged[0]
    assert epoch == 1
    assert payload["Overview/train-loss_total"] == 1.25
    assert payload["Training/curriculum-horizon"] == 4.0
    assert "ignored" not in payload


def test_zz_transient_pt_survives_canonical_source_removal(transient_package: dict[str, Any]) -> None:
    """Admit evidence-depth PT shards after test-owned canonical HDF5 removal."""
    root = transient_package["root"]
    dataset_id = transient_package["train_dataset"]
    sources = tuple(transient_package["source_root"].rglob("case.h5"))
    assert sources
    for source in sources:
        source.unlink()
    index_path = common.paths.resolve_dataset_path("transient_drying", dataset_id, dataset_root=root / "02_datasets")
    reopened = datasets.runtime.transient.TransientPTShardDataset(
        index_path,
        sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        source_root=root,
    )
    try:
        _assert_item_equal(transient_package["expected_item"], reopened[0])
    finally:
        reopened.close()
    receipt = datasets.packages.transient_shards.load_transient_shard_receipt(
        dataset_id,
        storage_root=str(root),
        validation_depth="evidence",
    )
    assert receipt["transition_count"] >= 2
    shards = datasets.packages.transient_shards
    receipt_path = shards.transient_shard_directory(dataset_id, storage_root=root) / shards.TRANSIENT_PT_RECEIPT_FILENAME
    historical = {**receipt, "schema_kind": "transient_pt_shard_receipt"}
    common.serialization.atomic_write_json(receipt_path, historical)
    with pytest.raises(ValueError, match="identity is invalid"):
        datasets.packages.transient_shards.load_transient_shard_receipt(
            dataset_id,
            storage_root=root,
            validation_depth="evidence",
        )
