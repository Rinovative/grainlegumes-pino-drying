"""Exercise a compact transient package-to-runtime training boundary.

Responsibilities:
  - Exercise compact transient packages, scaling, A0/B/A1 lifecycle, A0 budget identity, and model optimizer paths
  - Exercise best-checkpoint payload kinds and the refusal of non-completed B runs for evaluation
  - Exercise the plan's refusal, without rerun advice or traceback, to continue a B that its persisted state or a pinned
    input refuses, and its rerun advice, also without a traceback, for a B whose job runs on another GPU type or in
    another image
  - Exercise B as its receipt's one owner: its saved-config resume and its plan step publish the same bytes, an A1 without
    the receipt is refused, naming B's step, and A1 writes only its own copy, whose failed write leaves A1 interrupted,
    as does host memory exhausted while A1 decodes its best after its terminal last checkpoint
  - Exercise a pinned manifest that is malformed, has a grid digest its binding refuses or contradicts the saved split,
    which refuses an interrupted A1 alike in its recovery check and its rerun, and a Dataset area behind a symbolic link,
    which both read like the transient loaders
  - Compare an interrupted B, resumed with another memory batch, group by group with an uninterrupted B
  - Exercise one real A0 Optuna trial whose study signature equals the dry run's and whose trial and history validate

Design principles:
  - Persisted PT data and explicit conditioning survive canonical source removal

This module does NOT:
  - Train production campaigns or replace focused unit contracts
"""

from __future__ import annotations

import errno
import json
import math
from typing import TYPE_CHECKING, Any

import pytest
import torch
import yaml
from support import configs, optuna_trials, run_state
from support.generation_fixtures import successful_source_case
from torch.optim.adamw import AdamW

from src import analysis, common, datasets, domain, experiments, generation, learning
from src.datasets.packages import dataset_packages_selection as selection
from src.experiments import experiments_matched_budget as matched_budget
from src.experiments import experiments_wandb_schema as wandb_schema
from src.experiments.cli import cli_train
from src.learning.inference import learning_inference_transient as transient_inference
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient import learning_transient_adapter as transient_adapter
from src.learning.transient import learning_transient_handoff as handoff
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum, RolloutCurriculumState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

pytestmark = pytest.mark.integration
_LR_SCALE = 0.5
# K of the interrupted B: its K x N transitions outlast its first group (at most two windows of two transitions),
# and the resumed B still finishes in seconds.
_INTERRUPTED_K = 3
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
        assert sampled_hdf5_item["next_state"].shape[-2:] == (6, 6)
        assert sampled_hdf5_item["metadata"]["spatial_y_indices"] == (0, 3, 6, 9, 12, 15)
        assert sampled_hdf5_item["metadata"]["spatial_x_indices"] == (0, 3, 6, 9, 12, 15)
        assert sampled_hdf5_item["metadata"]["spatial_index_identity_sha256"] == sampled_pt_item["metadata"]["spatial_index_identity_sha256"]
        assert sharded.storage_backend == "pt_shards"
        assert window[0]["next_state"].shape[0] == 2
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
        units_per_update=2,
        memory_batch=1,
        model_sha256="a" * 64,
        storage_root=str(root),
        train_subset=None,
    )
    try:
        assert loaders.scaling.spatial_shape == (16, 16)
        assert loaders.split["roles"]["train"]["case_ids"] == transient_package["case_partition"]["train"]
        assert loaders.split["roles"]["id_test"]["case_ids"] == transient_package["case_partition"]["id_test"]
        assert loaders.split["dataset_identity"]["train"]["dataset_id"] == id_dataset_id
    finally:
        for value in (loaders.train, loaders.evaluation):
            _close_loader_dataset(value)
    curriculum = RolloutCurriculum(lengths=(2, 4), milestone_fractions=(0.0, 0.2))
    curriculum_state = RolloutCurriculumState.create(curriculum, seed=9)
    draws = curriculum_state.draw(progress=0.2, available_lengths=(4, 4))
    assert len(draws) == 2
    assert all(horizon in (2, 4) and 0 <= origin <= 4 - horizon for horizon, origin in draws)
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
    budget_epochs: int = 1,
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
        }
    )
    raw["training"]["units_per_update"] = 2
    raw["model"]["params"].update({"n_modes": [8, 8], "hidden_channels": 8, "n_layers": 1, "lifting_channel_ratio": 1, "projection_channel_ratio": 1})
    if not scheduler:
        raw["scheduler"] = None
    raw["tracking"]["wandb"]["mode"] = "disabled"
    raw["training"]["a0"]["budget_epochs"] = budget_epochs  # the one K: A0's passes and B's K x N transitions
    raw["training"]["b"].update(
        {
            "lr_scale": lr_scale,
            "sampling": {"mode": "rollout_window", "rollout_length": 2, "window_stride": 1, "window_offset": 0},
            "curriculum": {"lengths": [2], "milestone_fractions": [0.0]},
        }
    )
    raw["training"]["a1"].update({"lr_scale": lr_scale})
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
        rollout_curriculum="curriculum" in config["training"],  # only the self-fed Stage B fills the curriculum series
    )
    persisted = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    produced = set().union(*persisted["epochs"], *persisted["validations"])
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
        expected_source_run_name=a0["config"]["run"]["name"],
    )
    assert manifest["model_kind"] == "fno"
    assert b_summary["status"] == "completed"
    # Only B draws rollout horizons, so only B derives a curriculum seed and records a terminal curriculum.
    assert a_summary["seed_plan"] == a1_summary["seed_plan"] == {key: value for key, value in b_summary["seed_plan"].items() if key != "curriculum_b"}
    assert "curriculum_b" in b_summary["seed_plan"]
    assert "terminal_curriculum" not in a_summary
    assert "terminal_curriculum" not in a1_summary
    # A0 trains K passes in units, B K x N supervised transitions (K = 1 here), A1 exactly B's receipt; each ends at point 20.
    a0_budget, b_budget, a1_budget = (summary["terminal_budget"] for summary in (a_summary, b_summary, a1_summary))
    assert a0_budget["unit"] == "units"
    assert a0_budget["units"] >= a0_budget["total"]
    assert b_budget["unit"] == a1_budget["unit"] == "supervised_transitions"
    assert b_budget["total"] == b_summary["training_units"]["full_set_transitions"] > 0
    assert b_budget["supervised_transitions"] >= b_budget["total"]
    assert a1_budget["total"] == b_budget["supervised_transitions"]
    assert a1_budget["supervised_transitions"] >= a1_budget["total"]
    assert a0_budget["validated_point"] == b_budget["validated_point"] == a1_budget["validated_point"] == 20
    # The pair's one GPU model is B's, recorded in its receipt, and A1 trained on it.
    receipt = json.loads((b_dir / "b_train_budget_receipt.json").read_text(encoding="utf-8"))
    assert receipt["device"] == handoff.device_evidence(resolution.as_dict())
    assert b_summary["terminal_controller"]["best_within_budget_metric"] == b_summary["best_metric"]
    assert b_summary["terminal_controller"]["best_within_budget_epoch"] + 1 == b_summary["best_epoch"]
    assert b_summary["global_step"] == b_summary["terminal_controller"]["post_handoff_optimizer_steps"] >= 1
    assert a_summary["global_step"] >= 1
    assert b_summary["terminal_curriculum"]["draw_index"] > 0
    assert b_summary["terminal_curriculum"]["max_horizon"] == 2
    assert len(json.loads((b_dir / "history.json").read_text(encoding="utf-8"))["epochs"]) == 1
    assert (b_dir / "teacher_handoff_manifest.json").is_file()
    assert (b_dir / "b_train_budget_receipt.json").is_file()
    assert a1_summary["status"] == "completed"
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
    request["experiment"]["training"]["a0"]["budget_epochs"] = 1
    request["search_space"] = {"optimizer.lr": {"name": "learning_rate", "kind": "float", "low": 1.0e-3, "high": 2.0e-3, "log": True}}

    completed = optuna_trials.run_one_trial_study(request, directory=tmp_path)

    (completed["run_dir"] / "history.json").unlink()
    with pytest.raises(experiments.run.RunLifecycleError, match="history"):
        experiments.run.validate_completed_run(completed["run_dir"])


def _saved_learning_rates(run_dir: Path) -> tuple[float, float]:
    """Return the optimizer group rate and the schedule's start rate saved in one leaf's last checkpoint."""
    checkpoint = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_state_dict"]["kind"] == "cosine"
    return float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]), float(checkpoint["scheduler_state_dict"]["start_rate"])


def _adam_steps(optimizer_state: Mapping[str, Any]) -> set[float]:
    """Return the distinct AdamW step counters of one optimizer state."""
    return {float(parameter_state["step"]) for parameter_state in optimizer_state["state"].values()}


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


def _exhausted_memory(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for the selected-checkpoint evaluation whose decode of the best checkpoint exhausts host memory."""
    message = "DefaultCPUAllocator: can't allocate memory"
    raise MemoryError(message)


def _contradicted_split(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for the split resolution of a Dataset that changed since a run saved its split."""
    message = "Saved split does not match the current Dataset manifests, partition, evaluation membership or item expansion."
    raise ValueError(message)


def _malformed(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a Dataset manifest whose first case lost its Source entry, which the manifest schema reads."""
    del manifest["source_case_identities"][0]["source"]
    return manifest


def _regridded(axes_sha256: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a change of a Dataset manifest to the grid digest ``axes_sha256``, with the Dataset digest of that content."""

    def regrid(manifest: dict[str, Any]) -> dict[str, Any]:
        manifest["grid"]["axes_sha256"] = axes_sha256
        provenance = {key: manifest[key] for key in datasets.packages.manifest.PACKAGE_PROVENANCE_KEYS}
        manifest["dataset_digest"] = datasets.contracts.identity.package_identity_from_provenance(provenance)[1]
        return manifest

    return regrid


def _failed_continuation(run_dir: Path, capsys: pytest.CaptureFixture[str], command: Sequence[str]) -> str:
    """Run ``command``, which must fail to continue ``run_dir`` and leave it unchanged, and return what it printed to stderr."""
    before_attempt = {path: path.read_bytes() for path in sorted(run_dir.rglob("*")) if path.is_file()}
    capsys.readouterr()
    assert cli_train.main(list(command)) == 1
    assert {path: path.read_bytes() for path in sorted(run_dir.rglob("*")) if path.is_file()} == before_attempt
    return capsys.readouterr().err


def _assert_refused_without_rerun(stderr: str, reason: str, *, refusal: str = "not eligible for checkpoint resume") -> None:
    """Require a refusal that names ``reason`` and ``refusal``, advises no rerun or retry, which is refused again, and prints no traceback."""
    assert reason in stderr
    assert refusal in stderr
    assert "Rerun" not in stderr
    assert "Retry" not in stderr
    assert "Traceback" not in stderr


def _assert_rerun_offered(stderr: str, reason: str, *, rerun: str = "Rerun the same command inside a matching Slurm allocation") -> None:
    """Require an end by the job's environment that names ``reason`` and offers the ``rerun`` in a matching allocation, without a traceback."""
    assert reason in stderr
    assert rerun in stderr
    assert "not eligible" not in stderr
    assert "Traceback" not in stderr


def _interrupting_the_second_group() -> Callable[..., Any]:
    """Return the training loop's group training, interrupted at its second call, after the first committed group."""
    real_groups = learning.training.loop.train_update_groups
    calls = [0]

    def interrupt_the_second_group(*args: Any, **kwargs: Any) -> Any:
        # Each call trains up to the next stage event, and the first event publishes the last checkpoint.
        calls[0] += 1
        if calls[0] == 2:
            message = "synthetic interruption after the first committed group"
            raise KeyboardInterrupt(message)
        return real_groups(*args, **kwargs)

    return interrupt_the_second_group


def _a1_trains_on_the_receipt_b_published(
    plan_path: Path, output_root: Path, b_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Train A1 on the receipt that only B writes, which B's saved config published when it continued B to its end.

    B's plan step republishes the same bytes. Without the receipt, A1's plan and A1's ``--resume`` are refused, naming B's
    step, and change neither leaf. A1's started attempt writes only its own copy: a damaged copy refuses A1 unchanged, and
    a write of it that the job's environment fails leaves A1 interrupted, which its ``--resume`` finishes. A pinned
    manifest that A1's loaders refuse refuses the interrupted A1 in its recovery check as in its rerun.
    """
    b_receipt = b_dir / matched_budget.RECEIPT_NAME
    published = b_receipt.read_bytes()
    plan_a1 = (str(plan_path), "--train-stages", "a1")

    def republished_by_bs_step() -> None:
        experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("b",))
        assert b_receipt.read_bytes() == published

    def refused_without_bs_receipt(command: Sequence[str]) -> None:
        b_receipt.unlink()
        stderr = _failed_continuation(b_dir.parent, capsys, command)  # B's leaf and A1's
        _assert_refused_without_rerun(stderr, "--train-stages b", refusal="refused before its attempt started")
        republished_by_bs_step()

    republished_by_bs_step()
    refused_without_bs_receipt(plan_a1)
    b_leaf = {path: path.read_bytes() for path in sorted(b_dir.rglob("*")) if path.is_file()}
    with monkeypatch.context() as interruption:
        interruption.setattr(learning.training.loop, "train_update_groups", _interrupting_the_second_group())
        with pytest.raises(experiments.run.MatchedStageInterrupted) as interrupted:
            experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("a1",))
    a1_dir = interrupted.value.run_dir
    a1_copy = a1_dir / matched_budget.RECEIPT_NAME
    assert interrupted.value.recovery == "resumable"
    assert a1_copy.read_bytes() == published
    refused_without_bs_receipt(("--resume", str(a1_dir)))
    a1_copy.write_bytes(published[: len(published) // 2])
    _assert_refused_without_rerun(_failed_continuation(a1_dir, capsys, plan_a1), "Immutable B budget receipt differs")
    a1_copy.unlink()
    publish, writes = matched_budget.publish_b_receipt, []

    def unwritable_a1(run_dir: Path, receipt: Mapping[str, Any]) -> Path:
        writes.append(run_dir)
        if a1_dir.samefile(run_dir):
            raise OSError(errno.ENOSPC, "No space left on device", str(a1_copy))
        return publish(run_dir, receipt)

    with monkeypatch.context() as unwritable:
        unwritable.setattr(matched_budget, "publish_b_receipt", unwritable_a1)
        with pytest.raises(experiments.run.MatchedStageExecutionError) as ended:
            experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("a1",))
    assert [a1_dir.samefile(path) for path in writes] == [True]
    assert (ended.value.recovery, experiments.run.read_run_summary(a1_dir)["status"]) == ("resumable", "interrupted")
    # A pinned manifest A1's loaders refuse is A1's state: an interrupt in A1's admission gets the recovery check, which
    # refuses A1 by that manifest as the rerun's admission does, and advises no rerun; the restored manifest admits A1
    # again. That is a malformed manifest, a grid digest its binding refuses, and another grid, which contradicts A1's split.
    saved_a1 = experiments.config.loader.load_yaml(a1_dir / "config.yaml")
    manifest_path = common.paths.resolve_dataset_manifest_path(
        "transient_drying", saved_a1["data"]["train_dataset"], dataset_root=saved_a1["paths"]["dataset_root"]
    )
    pinned = manifest_path.read_bytes()

    def recovery_of_an_interrupted_admission() -> tuple[str, str]:
        """Interrupt A1's admission once and return the recovery check's verdict and reason."""
        require_attempt_history, admissions = experiments.run._require_attempt_history, []

        def interrupted_once(*args: Any, **kwargs: Any) -> None:
            admissions.append(args)
            if len(admissions) == 1:
                message = "synthetic interruption in A1's admission"
                raise KeyboardInterrupt(message)
            require_attempt_history(*args, **kwargs)

        with monkeypatch.context() as interruption:
            interruption.setattr(experiments.run, "_require_attempt_history", interrupted_once)
            with pytest.raises(experiments.run.MatchedStageInterrupted) as interrupted_admission:
                experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("a1",))
        return interrupted_admission.value.recovery, interrupted_admission.value.recovery_reason

    for damage, reason in (
        (_malformed, f"Dataset manifest is malformed: {manifest_path}"),
        (_regridded("not-a-digest"), "Dataset grid axes_sha256 must be"),
        (_regridded("0" * 64), "Saved split does not match the current Dataset"),
    ):
        common.serialization.atomic_write_json(manifest_path, damage(json.loads(pinned)))
        recovery, recovery_reason = recovery_of_an_interrupted_admission()
        assert (recovery, reason in recovery_reason) == ("refused", True), recovery_reason
        _assert_refused_without_rerun(_failed_continuation(a1_dir, capsys, plan_a1), reason)
        common.serialization.atomic_write_bytes(manifest_path, pinned)
    # Its loaders read the Datasets below the storage root, also through a Dataset area that is a symbolic link, and so
    # does its admission: the recovery check finds A1 resumable.
    datasets_root = common.paths.get_datasets_root(storage_root=saved_a1["paths"]["storage_root"])
    moved = datasets_root.with_name("datasets_behind_a_link")
    datasets_root.rename(moved)
    datasets_root.symlink_to(moved, target_is_directory=True)
    assert recovery_of_an_interrupted_admission()[0] == "resumable"
    datasets_root.unlink()
    moved.rename(datasets_root)
    # Host memory exhausted while A1 decodes its best after its terminal last is the job's environment too: A1 ends
    # interrupted, reported so without a traceback and advised the rerun that continues it, which then only finalizes it.
    with monkeypatch.context() as exhausted:
        exhausted.setattr(learning.training.loop, "evaluate_selected_checkpoint", _exhausted_memory)
        capsys.readouterr()
        assert cli_train.main(list(plan_a1)) == 1
    stderr = capsys.readouterr().err
    assert "Matched stage a1 interrupted" in stderr
    _assert_rerun_offered(stderr, "MemoryError", rerun="Rerun the same command inside a matching Slurm allocation; it continues this stage")
    assert experiments.run.read_run_summary(a1_dir)["status"] == "interrupted"
    assert experiments.run.run_experiment(a1_dir / "config.yaml", resume=a1_dir)["result"]["status"] == "completed"
    assert a1_copy.read_bytes() == published
    assert {path: path.read_bytes() for path in sorted(b_dir.rglob("*")) if path.is_file()} == b_leaf


def test_matched_b_restarts_its_schedule_at_the_scaled_a0_start_rate(
    transient_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A fresh B starts its own cosine at lr_scale times A0's start rate and keeps A0's Adam step counters; a resume continues it.

    The matched A0 handoff best keeps full training state without schedule state. B is
    interrupted after its first committed group, refused for evaluation while interrupted,
    resumed on its full identity to the end of its budget, and then never extended. While its
    A0 was trained by other training code, a Dataset contradicts its saved split, its last
    checkpoint does not restore, or its admission before the writer lease refuses its summary,
    the plan, and for its A0 also its saved config, refuses to continue B, leaves its leaf
    unchanged and advises no rerun, which would be refused again, and prints no traceback.
    A job on another GPU type than B's first attempt, in another image than its checkpoints', or
    with other packages than its A0 fails by its environment: B stays unchanged and is offered
    the rerun in a matching allocation, also without a traceback, unless the read-only recovery
    check finds B's own state refusing, such as a truncated last checkpoint, whose reason is then
    the only advice. B continued to its end by its
    saved config publishes its receipt, an interrupted continuation none; A1 trains on that
    receipt and writes only its own copy.
    """
    root = transient_package["root"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    plan_path = _authored_fno_plan(
        train_dataset=transient_package["train_dataset"],
        destination=root / "scaled_fno.yaml",
        device="cpu",
        budget_epochs=_INTERRUPTED_K,
        lr_scale=_LR_SCALE,
        scheduler=True,
    )
    output_root = root / "03_experiments"
    a0_dir = experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("a0",))["run_dir"]
    teacher = torch.load(a0_dir / handoff.HANDOFF_DIRECTORY / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    assert teacher["payload_kind"] == "full"
    assert teacher["scheduler_state_dict"] is None
    start_rate = float(experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(a0_dir))["optimizer"]["lr"])
    _assert_curated_series_are_produced(a0_dir, cuda=False)  # the CPU run of a real A0 checks the producers where no GPU is needed

    with monkeypatch.context() as interruption:
        interruption.setattr(learning.training.loop, "train_update_groups", _interrupting_the_second_group())
        with pytest.raises(experiments.run.MatchedStageInterrupted) as interrupted:
            experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("b",))
    b_dir = interrupted.value.run_dir
    assert interrupted.value.recovery == "resumable"
    _assert_evaluation_refused(b_dir, "interrupted")
    # The first group ran at the scaled stage start rate (p = 0), whatever rate A0's cosine had reached at its best; the
    # checkpoint holds that start rate and the cosine rate at the fraction of K x N transitions the group committed.
    b_last = torch.load(b_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    progress = b_last["budget_state_dict"]["supervised_transitions"] / b_last["budget_state_dict"]["total"]
    assert 0.0 < progress < 1.0
    scaled = start_rate * _LR_SCALE
    assert _saved_learning_rates(b_dir) == pytest.approx((scaled * (1.0 + math.cos(math.pi * progress)) / 2.0, scaled), rel=1e-3)
    assert b_last["global_step"] == 1
    assert _adam_steps(b_last["optimizer_state_dict"]) == {step + 1 for step in _adam_steps(teacher["optimizer_state_dict"])}
    # B counts its own work from zero; A0's committed groups are not carried across the handoff.
    first_work = experiments.run.read_run_summary(b_dir)["training_work"]
    assert first_work["optimizer_steps"] == 1
    assert first_work["gradient_predictions"] > 0

    plan_b = (str(plan_path), "--train-stages", "b")

    def continue_b(*command: str) -> str:
        """Continue B by ``command``, its plan by default, which must fail and leave B unchanged, and return its stderr."""
        return _failed_continuation(b_dir, capsys, command or plan_b)

    with monkeypatch.context() as drift:
        # An A0 trained by other training code is no longer a valid teacher for continuing B: its persisted state refuses B
        # before its attempt, whether its plan or its saved config continues it.
        run_state.drift_training_code(drift, "transient_drying")
        for command in (plan_b, ("--resume", str(b_dir))):
            stderr = continue_b(*command)
            _assert_refused_without_rerun(stderr, "training_transient_drying implementation changed", refusal="Run continuation refused")

    # Another GPU type than B's first attempt, or another image than the one its checkpoints come from, is the job's
    # environment, not B's state: the plan offers the rerun in a matching allocation, which continues B, without a traceback.
    summary_path = common.paths.resolve_run_summary_path(b_dir)
    recorded = summary_path.read_bytes()
    summary = json.loads(recorded)
    summary["runtime_sessions"][0].update(device_type="cuda", cuda_device_name="NVIDIA RTX 6000 Ada Generation")
    common.serialization.atomic_write_json(summary_path, summary)
    _assert_rerun_offered(continue_b(), "Run it on that GPU type")
    # With a truncated last checkpoint as well, the recovery check finds B's own state refusing, which that rerun would
    # meet again: its reason only, no rerun.
    last_path = b_dir / "last_checkpoint.pt"
    restorable = last_path.read_bytes()
    last_path.write_bytes(restorable[: len(restorable) // 2])
    _assert_refused_without_rerun(continue_b(), "Checkpoint is unreadable")
    last_path.write_bytes(restorable)
    summary_path.write_bytes(recorded)
    environment = learning.training.checkpoint.numerical_environment_identity()
    with monkeypatch.context() as image:
        image.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: {**environment, "sif_sha256": "0" * 64})
        _assert_rerun_offered(continue_b(), "differs in sif_sha256")
    # So is another package version, which already B's teacher, A0, cannot be read in: the plan offers the rerun before B.
    with monkeypatch.context() as packages:
        packages.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: {**environment, "numpy": "0.0.0"})
        _assert_rerun_offered(continue_b(), "differs in numpy", rerun="next: rerun the same command in a matching allocation, the run is unchanged")
    # A Dataset that changed since B started contradicts B's saved split: B's pinned input refuses the continuation.
    with monkeypatch.context() as changed:
        changed.setattr(datasets.preprocessing.splits, "resolve_split", _contradicted_split)
        _assert_refused_without_rerun(continue_b(), "Saved split does not match the current Dataset")
    # A loss state with an unexpected key passes the read-only checks and fails only the attempt's full restore.
    common.serialization.atomic_torch_save({**b_last, "loss_state_dict": {**b_last["loss_state_dict"], "unexpected": torch.zeros(1)}}, last_path)
    _assert_refused_without_rerun(continue_b(), "Unexpected key")
    last_path.write_bytes(restorable)
    # An attempt history of an earlier summary schema is refused by the read-only admission before the writer lease.
    common.serialization.atomic_write_json(summary_path, {key: value for key, value in json.loads(recorded).items() if key != "duration_contract"})
    _assert_refused_without_rerun(continue_b(), "earlier summary schema")
    summary_path.write_bytes(recorded)
    with monkeypatch.context() as interruption:
        interruption.setattr(learning.training.loop, "train_loop", _interrupt_training)
        with pytest.raises(KeyboardInterrupt):
            experiments.run.run_experiment(b_dir / "config.yaml", resume=b_dir)
    _assert_evaluation_refused(b_dir, "interrupted")
    assert not (b_dir / matched_budget.RECEIPT_NAME).exists()  # an unfinished B has no receipt
    resumed = experiments.run.run_experiment(b_dir / "config.yaml", resume=b_dir)

    assert resumed["result"]["status"] == "completed"
    summary = experiments.run.read_run_summary(b_dir)
    assert summary["training_outcome"]["status"] == "completed"
    budget = summary["terminal_budget"]
    assert budget["supervised_transitions"] >= budget["total"] == _INTERRUPTED_K * summary["training_units"]["full_set_transitions"]
    assert summary["training_work"]["optimizer_steps"] > first_work["optimizer_steps"]
    # The resume continues B's own cosine from its saved start rate and never rescales it.
    assert _saved_learning_rates(b_dir)[1] == pytest.approx(start_rate * _LR_SCALE)
    experiments.run.validate_completed_run(b_dir)
    # The finished B has spent its budget: it is never extended.
    with pytest.raises(experiments.run.ContinuationRefusedError, match="never extended"):
        experiments.run.run_experiment(b_dir / "config.yaml", resume=b_dir)
    # Continued to its end by its saved config, B published its receipt, the one A1 trains on.
    _a1_trains_on_the_receipt_b_published(plan_path, output_root, b_dir, monkeypatch, capsys)
    # Its training curve continues from the A0 handoff; without its history the admission itself refuses the stage.
    selected = experiments.run.validate_evaluable_run(b_dir)["selected_checkpoint_sha256"]
    history = analysis.evaluation.run_discovery.stage_history(b_dir, expected_checkpoint_sha256=selected)
    assert isinstance(history, analysis.evaluation.run_discovery.StageHistory)
    a0_name = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(a0_dir))["run"]["name"]
    assert (history.stage, history.pipeline, history.start_epoch) == ("B", a0_name, teacher["best_epoch"])
    learning.training.history.history_path(b_dir).unlink()
    with pytest.raises(experiments.run.RunLifecycleError, match="history"):
        analysis.evaluation.run_discovery.stage_history(b_dir, expected_checkpoint_sha256=selected)


def test_an_interrupted_b_resumed_with_another_memory_batch_trains_like_an_uninterrupted_b(
    transient_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    B interrupted after its first committed group and resumed with another memory batch trains like an uninterrupted B.

    One plan under two output roots gives two B leaves of one identity. Group by group both draw the same units at the
    same committed progress and the same curriculum horizons and origins, train at the same learning rates, and commit
    the same supervised transitions; the memory batch only splits each group.
    """
    root = transient_package["root"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    plan_path = _authored_fno_plan(
        train_dataset=transient_package["train_dataset"],
        destination=root / "parity_fno.yaml",
        device="cpu",
        budget_epochs=_INTERRUPTED_K,
        scheduler=True,
    )
    output_roots = {"whole": root / "parity_whole", "resumed": root / "parity_resumed"}
    for output_root in output_roots.values():
        experiments.run.run_experiment(plan_path, device="cpu", output_root=output_root, train_stages=("a0",))
    real_groups = learning.training.loop.train_update_groups
    real_begin = transient_adapter.TransientTrainingAdapter.begin_group
    draws: dict[str, list[tuple[Any, ...]]] = {"whole": [], "resumed": []}
    rates: dict[str, list[float]] = {"whole": [], "resumed": []}
    state: dict[str, Any] = {"run": "whole", "calls": 0, "interrupt_at": None, "optimizer": None}

    def recording_groups(*args: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        if state["calls"] == state["interrupt_at"]:
            message = "synthetic interruption after the first committed group"
            raise KeyboardInterrupt(message)
        state["optimizer"] = args[2]
        return real_groups(*args, **kwargs)

    def recording_begin(adapter: Any, group: Any, *, progress: float) -> None:
        real_begin(adapter, group, progress=progress)
        draws[state["run"]].append((group.items, progress, adapter._group_draws))
        rates[state["run"]].extend(param_group["lr"] for param_group in state["optimizer"].param_groups)

    with monkeypatch.context() as recording:
        recording.setattr(learning.training.loop, "train_update_groups", recording_groups)
        recording.setattr(transient_adapter.TransientTrainingAdapter, "begin_group", recording_begin)
        whole = experiments.run.run_experiment(plan_path, device="cpu", output_root=output_roots["whole"], train_stages=("b",))["run_dir"]
        state.update(run="resumed", calls=0, interrupt_at=2)
        with pytest.raises(experiments.run.MatchedStageInterrupted) as interrupted:
            experiments.run.run_experiment(plan_path, device="cpu", output_root=output_roots["resumed"], train_stages=("b",))
        resumed = interrupted.value.run_dir
        assert 0 < len(draws["resumed"]) < len(draws["whole"])
        # The same authored plan with another B memory batch, an execution setting outside identity, continues the leaf.
        base_path = plan_path.parents[2] / "base.yaml"  # the task base the plan file extends holds its training section
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        assert base["training"]["b"]["memory_batch"] == 1
        base["training"]["b"]["memory_batch"] = 2
        base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        state["interrupt_at"] = None
        experiments.run.run_experiment(plan_path, device="cpu", output_root=output_roots["resumed"], train_stages=("b",))

    assert draws["resumed"] == draws["whole"]
    assert rates["resumed"] == pytest.approx(rates["whole"], rel=1e-12)
    b_dirs = {"whole": whole, "resumed": resumed}
    summaries = {name: experiments.run.read_run_summary(run_dir) for name, run_dir in b_dirs.items()}
    assert [session["memory_batch"] for session in summaries["resumed"]["runtime_sessions"]] == [1, 2]
    budgets = {name: {key: value for key, value in summaries[name]["terminal_budget"].items() if key != "train_seconds"} for name in b_dirs}
    assert budgets["resumed"] == budgets["whole"]
    assert summaries["resumed"]["training_work"] == summaries["whole"]["training_work"]
    assert summaries["resumed"]["terminal_curriculum"] == summaries["whole"]["terminal_curriculum"]
    last = {name: torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False) for name, run_dir in b_dirs.items()}
    assert last["resumed"]["update_work"] == last["whole"]["update_work"]


def _resolved_direct_model_config(
    root: Path,
    *,
    train_dataset: str,
    model_kind: str,
) -> dict[str, Any]:
    """Derive and shrink a test-owned architecture-specific single-stage A0 config for one update."""
    raw = configs.transient_stage_a_only_config(model_kind=model_kind, seed=9)
    raw["training"]["budget_epochs"] = raw["training"]["stage_schedule"]["budget_epochs"] = 1
    resolved = configs.resolve(raw)
    resolved["run"].update({"device": "cuda", "deterministic": False})
    resolved["data"].update(
        {
            "train_dataset": train_dataset,
            "material_ood_datasets": [],
            "evaluation_batch_size": 1,
        }
    )
    resolved["training"].update({"units_per_update": 2, "memory_batch": 1})
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
        units_per_update=config["training"]["units_per_update"],
        memory_batch=config["training"]["memory_batch"],
        model_sha256="a" * 64,
        storage_root=str(root),
    )
    device = learning.device.resolve_device("cuda").device
    try:
        adapter = transient_adapter.build_transient_training_adapter(
            config,
            scaling=loaders.scaling,
            device=device,
            curriculum_seed=None,  # a teacher-forced A0 draws no rollout horizon
        )
        model = learning.models.factory.build_model(config, device=device)
        loss = learning.losses.factory.build_training_loss(config, device=device)
        optimizer = AdamW(model.parameters(), lr=1.0e-3)
        before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
        # The same epoch setup and update path the training loop runs; a 20-unit budget puts a validation point after every unit,
        # so the stretch returns after its first update group.
        learning.training.loop.train_unit_sampler(loaders.train).begin_epoch(0)
        metrics = learning.training.loop.train_update_groups(
            model,
            loaders.train,
            optimizer,
            loss,
            device,
            adapter,
            budget=learning.training.budget.StageBudget(unit="units", total=20),
            adapter=adapter,
        )
        assert metrics["optimizer_steps"] == 1
        # One group of at most k=2 units, split into memory microbatches of one.
        assert 1 <= metrics["units"] <= 2
        assert metrics["train/microbatches"] == metrics["units"]
        assert metrics["gradient_predictions"] >= metrics["units"]
        assert torch.isfinite(torch.tensor(metrics["train/loss_total"])).item()
        assert any(not torch.equal(previous, current.detach()) for previous, current in zip(before, model.parameters(), strict=False))
        return model
    finally:
        for value in (loaders.train, loaders.evaluation):
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
    disabled = experiments.tracking.initialize_wandb(config, run_dir=root / "disabled_tracking")
    disabled.log_event(1, {"train/loss_total": 1.0, "budget/supervised_transitions": 1.0})
    assert not disabled.enabled
    assert not (root / "disabled_tracking").exists()

    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict[str, float | int], int]] = []

        def log(self, values: dict[str, float | int]) -> None:
            # History is logged without W&B's own step; the payload carries both axes.
            self.logged.append((dict(values), int(values["global_step"])))

    definitions = experiments.tracking.automatic_history_metric_definitions(
        config["evaluation"]["metrics"],
        objective_id=str(config["evaluation"]["objective"]["id"]),
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        cuda_enabled=True,
        task_id="transient_drying",
        rollout_curriculum=True,  # map a self-fed run's curriculum series through the observer
    )
    fake_run: Any = FakeRun()
    observer = experiments.tracking.WandbSession(
        fake_run,
        "normalized_drying_group_macro_rmse",
        "minimize",
        frozenset(),
        "offline",
        "fake",
        root,
        "fake",
        history_metric_definitions=definitions,
    )
    observer.log_event(
        1,
        {
            "train/loss_total": 1.25,
            "budget/supervised_transitions": 6.0,
            "budget/train_seconds": 2.0,
            "transient/curriculum_max_horizon": 4.0,
            "ignored": 9.0,
        },
    )
    payload, step = fake_run.logged[0]
    assert step == 1
    assert payload["global_step"] == 1
    assert payload["supervised_transitions"] == 6
    assert payload["Training/optimisation_seconds"] == 2.0
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
