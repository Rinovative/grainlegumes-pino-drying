# ruff: noqa: S101, PLR2004, SLF001, TRY003, EM102
"""Exercise a compact transient package-to-runtime training boundary."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support import configs
from torch.optim.adamw import AdamW

from src import common, datasets, domain, experiments, generation, learning
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.packages import dataset_packages_manifest as package_manifest
from src.datasets.packages import dataset_packages_trajectory as trajectory
from src.experiments.config import experiments_config_transient_plan as transient_plan
from src.learning.inference import learning_inference_transient as transient_inference
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient import learning_transient_adapter as transient_adapter
from src.learning.transient import learning_transient_handoff as handoff
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum, RolloutCurriculumState
from tests.generation.test_generation_transient import _small_scientific_contract, _source, _write_transient_case
from tests.generation.test_generation_transient_shards import _assert_item_equal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration
_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA integration smoke requires an available CUDA device.",
)


def _publication_identity() -> dict[str, str]:
    return {
        "campaign_run_id": "technical_runtime_smoke",
        "campaign_id": "a" * 64,
        "git_commit": "b" * 40,
        "campaign_terminal_sha256": "c" * 64,
        "transfer_inventory_sha256": "d" * 64,
    }


def _provenance(index: dict[str, Any], *, regime: str) -> dict[str, Any]:
    view = datasets.contracts.views.get_view("transient_drying")
    memberships = {str(case["package_case_id"]): str(case["membership"]) for case in index["cases"]}
    source_cases = [
        {
            "package_case_id": case["package_case_id"],
            "batch_id": case["source_batch_id"],
            "source_case_id": case["package_case_id"],
            "source_relative_path": case["source_relative_path"],
            "case_input_id": case["case_input_id"],
            "simulation_case_id": case["simulation_case_id"],
            "case_hdf5_sha256": case["source_hdf5_sha256"],
            "material_family": case["material_family"],
            "material_role": "seen",
            "evaluation_regime": regime,
            "natural_support_state": "natural",
            "simulation_profile": "transient_drying",
            "membership": case["membership"],
            "ood_group": case["ood_group"],
            "ood_parameters": list(case["ood_parameters"]),
            "task_relevant_ood_parameters": list(case["ood_parameters"]),
            "ood_evidence": case["ood_evidence"],
        }
        for case in index["cases"]
    ]
    source_ids = [case["package_case_id"] for case in index["cases"]]
    return {
        "schema_kind": package_manifest.DATASET_PACKAGE_SCHEMA_KIND,
        "schema_version": package_manifest.DATASET_PACKAGE_SCHEMA_VERSION,
        "dataset_name": f"transient_smoke_{regime}",
        "dataset_view": "transient_drying",
        "registered_task_id": view.registered_task_id,
        "evaluation_regime": regime,
        "materials": ["lentil"],
        "channel_contract": datasets.packages.builder._channel_contract("transient_drying"),
        "channel_contract_digest": view.contract_digest,
        "source_simulation_profiles": ["transient_drying"],
        "source_batches": [{"batch_id": "technical_runtime_smoke"}],
        "source_batch_ids": ["technical_runtime_smoke"],
        "source_template_digests": ["e" * 64],
        "source_git_commits": ["b" * 40],
        "source_case_identities": source_cases,
        "included_source_cases": source_ids,
        "excluded_source_cases": [],
        "source_selection_decisions": [],
        "matched_case_input_ids": sorted(case["case_input_id"] for case in index["cases"]),
        "airflow_provenance": [],
        "steady_flow_conditioning": {},
        "material_file_identities": {"lentil": "f" * 64},
        "operation_config_digests": ["d" * 64],
        "campaign_name": "technical_runtime_smoke",
        "campaign_id": "a" * 64,
        "campaign_digest": "a" * 64,
        "campaign_purpose": "technical_runtime_smoke",
        "material_roles": {"lentil": ["seen"]},
        "evaluation_regimes": [regime],
        "material_memberships": {"lentil": ["train", "validation", "id_test"] if regime == "id" else ["parameter_ood"]},
        "source_role": "seen",
        "training_eligible": True,
        "duplicate_case_input_policy": "forbid",
        "case_membership": memberships,
        "split_membership": memberships,
        "membership_counts": {name: list(memberships.values()).count(name) for name in sorted(set(memberships.values()))},
        "available_ood_groups": ["bed"] if regime == "parameter_ood" else [],
        "ood_group_indexes": {"bed": source_ids} if regime == "parameter_ood" else {},
        "ood_parameter_indexes": {"kappa_mean": source_ids} if regime == "parameter_ood" else {},
        "task_relevant_ood_parameters": ["kappa_mean"] if regime == "parameter_ood" else [],
        "material_counts": {"lentil": len(source_ids)},
        "source_profile_counts": {"transient_drying": len(source_ids)},
        "candidate_source_case_count": len(source_ids),
        "builder_identity": identity.dataset_conversion_contract_identity("transient_drying"),
        "schema_identity": {
            "package": package_manifest.DATASET_PACKAGE_SCHEMA_VERSION,
            "case_hdf5": generation.publication.storage.HDF5_SCHEMA_VERSION,
            "generation_case": generation.cases.case.CASE_CONTRACT_DIGEST,
            "transient_index": trajectory.TRANSIENT_INDEX_SCHEMA_VERSION,
        },
    }


def _publish_package(root: Path, *, regime: str, sources: list[Any]) -> tuple[str, Path, dict[str, Any]]:
    provisional = f"transient_smoke_{regime}"
    index_preview = trajectory.build_transient_index(
        sources, None, dataset_name=provisional, dataset_id=provisional, evaluation_regime=regime, source_root=root
    )
    provenance = _provenance(index_preview, regime=regime)
    dataset_id, dataset_digest = identity.package_identity_from_provenance(provenance)
    payload = root / "02_datasets" / "packages" / dataset_id / f"{dataset_id}.json"
    payload.parent.mkdir(parents=True)
    index = trajectory.build_transient_index(
        sources, payload, dataset_name=provenance["dataset_name"], dataset_id=dataset_id, evaluation_regime=regime, source_root=root
    )
    manifest = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": payload.name,
        "sample_count": index["sample_count"],
        "source_case_count": index["source_case_count"],
        "transition_count": index["transition_count"],
        "payload_sha256": common.serialization.file_sha256(payload),
    }
    metadata = common.paths.get_dataset_metadata_root(storage_root=root) / dataset_id
    metadata.mkdir(parents=True)
    common.serialization.atomic_write_json(metadata / "dataset_manifest.json", manifest)
    assert package_manifest.load_package_manifest(dataset_id, storage_root=root) == manifest
    return dataset_id, payload, index


@pytest.fixture(scope="module")
def transient_smoke_package(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Publish test-owned ID/OOD canonical packages and their derived PT shards."""
    tmp_path = tmp_path_factory.mktemp("transient_training_smoke")
    scientific = _small_scientific_contract()
    scientific["grid"].update({"nx": 16, "ny": 16, "Lx": 1.5, "Ly": 1.5, "dx": 0.1, "dy": 0.1})
    scientific["time"].update({"stop": 3.0, "interval": 1.0, "regular_times": [0.0, 1.0, 2.0, 3.0]})
    source_root = tmp_path / "01_generation" / "processed"
    id_sources: list[Any] = []
    for number, membership in enumerate(("train", "validation", "id_test"), start=1):
        path = source_root / f"id_{number}" / "case.h5"
        path.parent.mkdir(parents=True)
        _write_transient_case(
            path,
            exact_stop_time=None,
            scientific_contract=scientific,
            regular_state_count=4,
            case_input_id=f"{number:x}" * 64,
            simulation_case_id=f"{number + 3:x}" * 64,
        )
        id_sources.append(
            _source(
                path,
                membership=membership,
                package_case_id=f"smoke_id_{number}",
                case_input_id=f"{number:x}" * 64,
                simulation_case_id=f"{number + 3:x}" * 64,
            )
        )
    ood_path = source_root / "ood" / "case.h5"
    ood_path.parent.mkdir(parents=True)
    _write_transient_case(
        ood_path, exact_stop_time=None, scientific_contract=scientific, regular_state_count=4, case_input_id="a" * 64, simulation_case_id="b" * 64
    )
    ood_source = _source(
        ood_path, regime="parameter_ood", membership="parameter_ood", package_case_id="smoke_ood", case_input_id="a" * 64, simulation_case_id="b" * 64
    )
    id_dataset_id, _, _ = _publish_package(tmp_path, regime="id", sources=id_sources)
    ood_dataset_id, _, _ = _publish_package(tmp_path, regime="parameter_ood", sources=[ood_source])
    for dataset_id in (id_dataset_id, ood_dataset_id):
        datasets.packages.transient_shards.build_transient_shards(
            dataset_id, storage_root=tmp_path, publication_identity=_publication_identity(), target_shard_bytes=1_000_000
        )
    index_path = tmp_path / "02_datasets" / "packages" / id_dataset_id / f"{id_dataset_id}.json"
    canonical = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        source_root=tmp_path,
    )
    try:
        expected_item = _clone_item(canonical[0])
    finally:
        canonical.close()
    return {
        "root": tmp_path,
        "train_dataset": id_dataset_id,
        "ood_dataset": ood_dataset_id,
        "expected_item": expected_item,
    }


def _clone_item(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_item(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_item(item) for item in value]
    return value


def _close_loader_dataset(loader: Any) -> None:
    """Close one test-owned transient Dataset behind a DataLoader."""
    loader.dataset.close()


def test_transient_package_pt_runtime_and_scaling_smoke(transient_smoke_package: dict[str, Any]) -> None:
    """Keep package provenance, PT equivalence, roles, scaling, and curriculum connected."""
    root = transient_smoke_package["root"]
    id_dataset_id = transient_smoke_package["train_dataset"]
    ood_dataset_id = transient_smoke_package["ood_dataset"]
    one_step = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    rollout = datasets.contracts.transient.TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=1, window_offset=0)
    index_path = root / "02_datasets" / "packages" / id_dataset_id / f"{id_dataset_id}.json"
    canonical = datasets.runtime.transient.TransientPhysicalDataset(index_path, sampling=one_step, source_root=root)
    sharded = datasets.runtime.transient.TransientPTShardDataset(index_path, sampling=one_step, source_root=root)
    window = datasets.runtime.transient.TransientPTShardDataset(index_path, sampling=rollout, source_root=root)
    try:
        _assert_item_equal(canonical[0], sharded[0])
        assert sharded.storage_backend == "pt_shards"
        assert window[0]["target"].shape[0] == 2
    finally:
        canonical.close()
        sharded.close()
        window.close()
    receipt = datasets.packages.transient_shards.load_transient_shard_receipt(
        id_dataset_id,
        storage_root=str(root),
        publication_identity=_publication_identity(),
        validation_depth="evidence",
    )
    assert receipt["transition_count"] >= 2
    tensorizer = TransientTensorizerSpec(input_profile="canonical_physics_complete_v1", temporal_conditioning=TemporalConditioningSpec("none"))
    loaders = datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=id_dataset_id,
        ood_dataset_ids=ood_dataset_id,
        tensorizer=tensorizer,
        train_sampling=one_step,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0, hdf5_cache_size=0),
        storage_root=str(root),
        transient_backend_preference="pt_shards",
        transient_backend_required=True,
        allow_technical_smoke=True,
    )
    try:
        assert loaders.runtime_provenance["train"] == "pt_shards"
        assert loaders.scaling_artifact.spatial_shape == (16, 16)
        assert loaders.split["roles"]["train"]["case_ids"] == ["smoke_id_1"]
        assert loaders.dataset_identity["train"]["dataset_id"] == id_dataset_id
    finally:
        for value in (loaders.train, loaders.evaluation, loaders.ood, loaders.id_test):
            _close_loader_dataset(value)
    curriculum = RolloutCurriculum(lengths=(2, 4), milestone_fractions=(0.0, 0.2))
    curriculum_state = RolloutCurriculumState.create(curriculum, seed=9)
    horizon, origins = curriculum_state.select(progress=0.2, available_length=4, batch_size=2)
    assert horizon in (2, 4)
    assert origins.shape == (2,)
    assert curriculum_state.active_stage == 1
    assert curriculum_state.max_horizon == 4
    assert domain.tasks.registry.get_task("transient_drying").id == "transient_drying"


def _authored_fno_plan(
    *,
    train_dataset: str,
    ood_dataset: str,
    destination: Path,
) -> Path:
    """Write one compact test-owned authored FNO A0-to-B plan."""
    raw = configs.transient_two_stage_config(model_kind="fno", seed=9)
    raw["run"].update({"device": "cuda", "deterministic": False, "suffix": "technical"})
    raw["data"].update(
        {
            "train_dataset": train_dataset,
            "ood_datasets": [ood_dataset],
            "batch_size": 2,
            "transient_backend_required": True,
            "allow_technical_smoke": True,
        }
    )
    raw["model"]["params"].update({"n_modes": [8, 8], "hidden_channels": 8, "n_layers": 1, "lifting_channel_ratio": 1, "projection_channel_ratio": 1})
    raw["scheduler"] = None
    raw["tracking"]["wandb"]["mode"] = "disabled"
    raw["training"]["mixed_precision"] = False
    raw["training"]["stage_schedule"] = {"mode": "joint_ab", "budget_unit": "epochs", "total_epochs": 2, "stage_a_fraction": 0.5}
    raw["training"]["stage_a"].update(
        {
            "fixed_evaluation_horizon": 1,
            "curriculum": {"lengths": [1], "milestone_fractions": [0.0], "seed": 9},
        }
    )
    raw["training"]["stage_b"].update(
        {
            "sampling": {"mode": "rollout_window", "rollout_length": 2, "window_stride": 1, "window_offset": 0},
            "fixed_evaluation_horizon": 2,
            "curriculum": {"lengths": [2], "milestone_fractions": [0.0], "seed": 9},
            "matched_compute": {
                "planned_seconds": None,
                "planned_steps": None,
                "rollout_reference_seconds": None,
                "rollout_reference_steps": None,
            },
        }
    )
    experiments.config.loader.save_yaml(raw, destination)
    return destination


def _resolved_fno_config(root: Path, *, train_dataset: str, ood_dataset: str) -> dict[str, Any]:
    """Derive the exact test-owned FNO A0 child used by public plan execution."""
    plan_path = _authored_fno_plan(train_dataset=train_dataset, ood_dataset=ood_dataset, destination=root / "technical_fno.yaml")
    return copy.deepcopy(dict(transient_plan.load_and_resolve_transient_training_plan(plan_path).stage_a))


@_CUDA_REQUIRED
def test_transient_a0_and_b_lifecycle_smoke(transient_smoke_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the public CUDA FNO authored A0-to-B lifecycle with fresh B state."""
    root = transient_smoke_package["root"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    plan_path = _authored_fno_plan(
        train_dataset=transient_smoke_package["train_dataset"],
        ood_dataset=transient_smoke_package["ood_dataset"],
        destination=root / "technical_fno.yaml",
    )

    outcome = experiments.run.run_experiment(plan_path, device="cuda", output_root=root / "03_experiments")
    a0_dir = outcome["stage_runs"]["a"]
    b_dir = outcome["stage_runs"]["b"]
    a0 = experiments.run.validate_completed_run(a0_dir)
    b = experiments.run.validate_completed_run(b_dir)
    a_summary = experiments.run.read_run_summary(a0_dir)
    b_summary = experiments.run.read_run_summary(b_dir)
    resolution = outcome["device_resolution"]

    assert outcome["run_dir"] == b_dir
    assert a0_dir.name.endswith("_a0")
    assert b_dir.name.endswith("_b")
    assert a0["best_checkpoint"]["schema_version"] == 1
    assert (a0_dir / "history.json").is_file()
    manifest = handoff.validate_stage_a_handoff(
        a0_dir / "stage_a_handoff",
        target_config=b["config"],
        device=resolution.as_dict(),
        expected_source_run_name=a0["config"]["run"]["name"],
    )
    assert manifest["model_kind"] == "fno"
    assert b_summary["status"] == "completed"
    assert b_summary["terminal_controller"]["budget_control"] == "stage_epochs"
    assert b_summary["terminal_controller"]["planned_stage_epochs"] == 1
    assert b_summary["terminal_controller"]["completed_stage_epochs"] == 1
    assert b_summary["terminal_controller"]["budget_complete"] is True
    assert b_summary["terminal_controller"]["post_handoff_optimizer_steps"] == 1
    assert b_summary["terminal_controller"]["best_within_budget_metric"] == b_summary["best_metric"]
    assert b_summary["terminal_controller"]["best_within_budget_epoch"] + 1 == b_summary["best_epoch"]
    assert b_summary["global_step"] == b_summary["terminal_controller"]["successful_optimizer_steps"]
    assert b_summary["global_step"] == 1
    assert a_summary["global_step"] > b_summary["global_step"]
    assert b_summary["terminal_curriculum"]["draw_index"] > 0
    assert b_summary["terminal_curriculum"]["max_horizon"] == 2
    assert len(json.loads((b_dir / "history.json").read_text(encoding="utf-8"))["epochs"]) == 1
    assert (b_dir / "teacher_handoff_manifest.json").is_file()


def _resolved_direct_model_config(
    root: Path,
    *,
    train_dataset: str,
    ood_dataset: str,
    model_kind: str,
) -> dict[str, Any]:
    """Derive and shrink a test-owned architecture-specific A0 child for one update."""
    plan = transient_plan.resolve_transient_training_plan(
        configs.transient_two_stage_config(model_kind=model_kind, seed=9),
    )
    resolved = copy.deepcopy(dict(plan.stage_a))
    resolved["run"].update({"device": "cuda", "deterministic": False, "suffix": "technical"})
    resolved["data"].update(
        {
            "train_dataset": train_dataset,
            "ood_datasets": [ood_dataset],
            "batch_size": 1,
            "transient_backend_required": True,
            "allow_technical_smoke": True,
        }
    )
    resolved["scheduler"] = None
    resolved["tracking"]["wandb"]["mode"] = "disabled"
    resolved["training"]["epochs"] = 1
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
        resolved["temporal"]["sampling"] = {"mode": "rollout_window", "rollout_length": 2, "window_stride": 1, "window_offset": 0}
        resolved["training"].update(
            {
                "fixed_evaluation_horizon": 2,
                "curriculum": {"lengths": [2], "milestone_fractions": [0.0], "seed": 9},
            }
        )
        resolved["model"]["params"].update({"n_modes": [8, 8], "hidden_channels": 4, "n_layers": 1})
    else:
        raise ValueError(f"Unsupported direct smoke model kind {model_kind!r}.")
    resolved["paths"].update(
        {
            "storage_root": str(root),
            "dataset_root": str(common.paths.get_dataset_packages_root(storage_root=root)),
            "dataset_metadata_root": str(common.paths.get_dataset_metadata_root(storage_root=root)),
            "output_root": str(root / "03_experiments"),
        }
    )
    return resolved


def _run_direct_cuda_update(config: dict[str, Any], *, root: Path) -> torch.nn.Module:
    tensorizer = TransientTensorizerSpec(
        input_profile="canonical_physics_complete_v1",
        temporal_conditioning=TemporalConditioningSpec("none"),
    )
    sampling = datasets.contracts.transient.TransientSamplingSpec.from_mapping(config["temporal"]["sampling"])
    loaders = datasets.runtime.transient_training.create_transient_training_loaders(
        train_dataset_id=str(config["data"]["train_dataset"]),
        ood_dataset_ids=config["data"]["ood_datasets"],
        tensorizer=tensorizer,
        train_sampling=sampling,
        loader_settings=datasets.runtime.factory.LoaderSettings(batch_size=1, num_workers=0, hdf5_cache_size=0),
        storage_root=str(root),
        transient_backend_preference="pt_shards",
        transient_backend_required=True,
        allow_technical_smoke=True,
    )
    device = learning.device.resolve_device("cuda").device
    try:
        adapter = transient_adapter.build_transient_training_adapter(
            config,
            scaling=loaders.scaling_artifact,
            device=device,
        )
        raw_batch = next(iter(loaders.train))
        batch = adapter.prepare_batch(raw_batch, device=device, training=True)
        model = learning.models.factory.build_model(config, device=device)
        loss = learning.losses.factory.build_training_loss(config, device=device)
        optimizer = AdamW(model.parameters(), lr=1.0e-3)
        before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
        step = adapter.training_step(model, batch, loss)
        optimizer.zero_grad(set_to_none=True)
        step.loss.backward()
        optimizer.step()
        assert torch.isfinite(step.loss).item()
        assert step.processed_target_transitions == batch.batch_size * batch.rollout_length
        assert any(not torch.equal(previous, current.detach()) for previous, current in zip(before, model.parameters(), strict=False))
        return model
    finally:
        for value in (loaders.train, loaders.evaluation, loaders.ood, loaders.id_test):
            _close_loader_dataset(value)


@_CUDA_REQUIRED
def test_transient_uno_and_rno_cuda_optimizer_smoke(transient_smoke_package: dict[str, Any]) -> None:
    """Update real UNO and official single-step recurrent RNO parameters on CUDA."""
    root = transient_smoke_package["root"]
    train_dataset = transient_smoke_package["train_dataset"]
    ood_dataset = transient_smoke_package["ood_dataset"]
    uno = _resolved_direct_model_config(root, train_dataset=train_dataset, ood_dataset=ood_dataset, model_kind="uno")
    _run_direct_cuda_update(uno, root=root)
    assert "rno" in learning.models.factory.available_model_kinds()
    rno = _resolved_direct_model_config(root, train_dataset=train_dataset, ood_dataset=ood_dataset, model_kind="rno")
    _run_direct_cuda_update(rno, root=root)


@_CUDA_REQUIRED
def test_transient_completed_inference_and_tracking_smoke(transient_smoke_package: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Load completed A0 inference and observe transient W&B mapping without network access."""
    root = transient_smoke_package["root"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    completed_config = _resolved_fno_config(
        root,
        train_dataset=transient_smoke_package["train_dataset"],
        ood_dataset=transient_smoke_package["ood_dataset"],
    )
    a0_dir = root / "03_experiments" / "transient_drying" / "runs" / completed_config["run"]["name"]
    context = transient_inference.load_transient_inference_context(run_dir=a0_dir, device=learning.device.resolve_device("cuda").device)
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="rollout_window", rollout_length=2, window_stride=1, window_offset=0)
    index_path = root / "02_datasets" / "packages" / transient_smoke_package["train_dataset"] / f"{transient_smoke_package['train_dataset']}.json"
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

    config = _resolved_fno_config(
        root,
        train_dataset=transient_smoke_package["train_dataset"],
        ood_dataset=transient_smoke_package["ood_dataset"],
    )
    monkeypatch.setattr(experiments.tracking.importlib, "import_module", lambda _name: pytest.fail("disabled observer imported SDK"))
    disabled = experiments.tracking.initialize_wandb(config, run_dir=root / "disabled_tracking")
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
        continuity="none",
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
    assert payload["Overview/train_loss"] == 1.25
    assert payload["Curriculum/horizon"] == 4.0
    assert "ignored" not in payload


def test_zz_transient_pt_survives_canonical_source_removal(transient_smoke_package: dict[str, Any]) -> None:
    """Admit evidence-depth PT shards after test-owned canonical HDF5 removal."""
    root = transient_smoke_package["root"]
    dataset_id = transient_smoke_package["train_dataset"]
    sources = tuple((root / "01_generation" / "processed").rglob("case.h5"))
    assert sources
    for source in sources:
        source.unlink()
    index_path = root / "02_datasets" / "packages" / dataset_id / f"{dataset_id}.json"
    reopened = datasets.runtime.transient.TransientPTShardDataset(
        index_path,
        sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        source_root=root,
    )
    try:
        _assert_item_equal(transient_smoke_package["expected_item"], reopened[0])
    finally:
        reopened.close()
    receipt = datasets.packages.transient_shards.load_transient_shard_receipt(
        dataset_id,
        storage_root=str(root),
        publication_identity=_publication_identity(),
        validation_depth="evidence",
    )
    assert receipt["transition_count"] >= 2
