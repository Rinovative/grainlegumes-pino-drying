"""Exercise the complete steady-flow lifecycle on compact synthetic CPU data.

Responsibilities:
  - Exercise compact CPU training, completed bundles, relocation, resume, and ID/additional artifacts
  - Exercise one real Optuna trial whose study signature equals the dry run's and whose trial validates unpatched

Design principles:
  - Synthetic lifecycle evidence binds exact persisted identities before model forward

This module does NOT:
  - Benchmark production models or exhaustively test formulas and races
"""

from __future__ import annotations

import copy
import errno
import functools
import json
import pickle
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import torch
from neuralop.models import FNO
from support import configs, optuna_trials, run_state
from support.synthetic_task import build_synthetic_selection_provenance, save_steady_package

from src import analysis, common, datasets, domain, experiments, learning
from src.experiments import experiments_wandb_schema as wandb_schema
from src.experiments.cli import cli_train

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from collections.abc import Callable

_SAMPLE_IDS = tuple(f"lentil__case_{index + 1:04d}__{index + 1:012x}" for index in range(4))
_OOD_SAMPLE_IDS = tuple(f"chickpea__case_{index + 1:04d}__{index + 1:012x}" for index in range(4))
_ID_PROVENANCE = build_synthetic_selection_provenance(dataset_name="tiny_steady_id", sample_ids=_SAMPLE_IDS, material_role="id")
_OOD_PROVENANCE = build_synthetic_selection_provenance(
    dataset_name="tiny_steady_ood_named",
    sample_ids=_OOD_SAMPLE_IDS,
    material_family="chickpea",
    material_role="ood",
)
_SECOND_OOD_PROVENANCE = build_synthetic_selection_provenance(
    dataset_name="tiny_steady_ood_second",
    sample_ids=tuple(f"field_pea__case_{index + 1:04d}__{index + 11:012x}" for index in range(2)),
    material_family="field_pea",
    material_role="ood",
)
_ID_DATASET, _ID_DIGEST = datasets.contracts.identity.package_identity_from_provenance(_ID_PROVENANCE)
_OOD_DATASET, _ = datasets.contracts.identity.package_identity_from_provenance(_OOD_PROVENANCE)
_SECOND_OOD_DATASET, _ = datasets.contracts.identity.package_identity_from_provenance(_SECOND_OOD_PROVENANCE)
_PROVENANCES = {_ID_DATASET: _ID_PROVENANCE, _OOD_DATASET: _OOD_PROVENANCE, _SECOND_OOD_DATASET: _SECOND_OOD_PROVENANCE}
# The one material-OOD role covers both OOD Datasets in configured order.
_MATERIAL_OOD_DATASET = datasets.contracts.identity.combined_dataset_id((_OOD_DATASET, _SECOND_OOD_DATASET))
# Both axes hold at least the 10 points spectral Fourier continuation needs.
_SHAPE = (12, 12)
_ARTIFACT_CROP = 2
_SOLVER_TIMING = SimpleNamespace(
    stationary_airflow_solver_seconds=2.0, comsol_process_seconds=3.0, solver_hostname="solver-node", solver_requested_cores=8
)


@pytest.fixture(autouse=True)
def _synthetic_generation_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the Generation campaign the synthetic Datasets lack: every member case has one recorded COMSOL clock."""
    source = SimpleNamespace(load_timing=lambda: _SOLVER_TIMING)
    monkeypatch.setattr(analysis.artifacts.plan, "generation_case_sources", lambda role, **_kwargs: dict.fromkeys(role.case_ids, source))


@dataclass(frozen=True)
class CompletedLifecycle:
    """
    Retain the immutable outputs of the module-scoped one-epoch lifecycle.

    Attributes
    ----------
    config : dict[str, Any]
        Resolved CPU experiment contract used for the run.
    dataset_root : pathlib.Path
        Temporary raw root owning the ID and named OOD final datasets.
    run_dir : pathlib.Path
        Completed saved-run leaf whose artifacts may be mutated only in copied tests.
    id_payload, ood_payload : dict[str, Any]
        Original strict final payloads used for split and normalizer assertions.
    completed : dict[str, Any]
        Result of strict completed-run validation after best/last roles diverge.
    epoch_metrics : list[dict[str, float]]
        The completed-epoch payloads the training loop handed to its epoch-end consumers.

    Notes
    -----
    The dataclass is frozen, but contained dictionaries are test-owned mutable objects.

    """

    config: dict[str, Any]
    dataset_root: Path
    metadata_root: Path
    run_dir: Path
    id_payload: dict[str, Any]
    ood_payload: dict[str, Any]
    completed: dict[str, Any]
    epoch_metrics: list[dict[str, float]]


def _case_components(
    task: domain.tasks.spec.TaskSpec,
    *,
    case_id: str,
    offset: float,
    source: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Build deterministic in-memory case tensors and their scientific identity; ``source`` is the case's provenance source."""
    y_axis = torch.linspace(0.0, 1.0, _SHAPE[0], dtype=torch.float32)
    x_axis = torch.linspace(0.0, 1.0, _SHAPE[1], dtype=torch.float32)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    input_fields = {
        "x": x,
        "y": y,
        "Kxx": -12.0 + 0.02 * offset + 0.01 * x,
        "Kxy": 0.01 * offset + 0.02 * x * y,
        "Kyy": -11.8 + 0.015 * offset + 0.01 * y,
        "eps_bed": 0.25 + 0.005 * offset + 0.02 * x + 0.01 * y,
        "p_in_bc": (1.0 - x) * (1.0 + 0.02 * offset) + 0.01 * y,
    }
    output_fields = {
        "p": (1.0 - x) * (1.0 + 0.01 * offset) + 0.02 * y,
        "u": 1.0e-4 * (1.0 + 0.03 * offset + x + 0.2 * y),
        "v": 1.0e-4 * (-0.5 + 0.02 * offset + 0.1 * x - y),
    }
    inputs = torch.stack([input_fields[name] for name in task.input_names])
    outputs = torch.stack([output_fields[name] for name in task.output_names])
    source_identity = {"generator": "synthetic-lifecycle", "case": case_id}
    # Case-equal validation reads the simulation case and material of every item, as real packages carry them.
    source_metadata = {
        "offset": offset,
        "case_id": case_id,
        "simulation_case_id": source["simulation_case_id"],
        "material_family": source["material_family"],
        "parameters": {"synthetic_parameter": 1.0},
    }
    fingerprint = datasets.contracts.identity.compute_case_fingerprint(
        task=task,
        case_id=case_id,
        source_identity=source_identity,
        source_metadata=source_metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return inputs, outputs, source_identity, source_metadata, fingerprint


def _training_dataset_payload(
    task: domain.tasks.spec.TaskSpec,
    *,
    dataset_id: str,
    offsets: tuple[float, ...],
) -> dict[str, Any]:
    """Build one final synthetic dataset entirely in memory."""
    provenance = _PROVENANCES[dataset_id]
    entries = provenance["source_case_identities"]
    sample_ids = [entry["package_case_id"] for entry in entries]
    cases = [
        _case_components(task, case_id=entry["package_case_id"], offset=offset, source=entry["source"])
        for entry, offset in zip(entries, offsets, strict=True)
    ]
    return datasets.contracts.identity.build_training_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=sample_ids,
        source_selection_identity={
            "schema_kind": "dataset_source_selection",
            "source_cases": provenance["source_case_identities"],
        },
        source_identities=[case[2] for case in cases],
        source_metadata=[case[3] for case in cases],
        source_provenance=provenance,
        case_fingerprints=[case[4] for case in cases],
        inputs=torch.stack([case[0] for case in cases]),
        outputs=torch.stack([case[1] for case in cases]),
    )


def _save_dataset(root: Path, metadata_root: Path, payload: dict[str, Any]) -> Path:
    """Publish one bounded synthetic steady payload with its current manifest."""
    assert metadata_root == root
    return save_steady_package(root, payload)


def test_steady_manifest_rejects_unsafe_or_changed_payload(tmp_path: Path) -> None:
    """Bind the one manifest reader to the folder's own immutable payload bytes."""
    task = domain.tasks.registry.get_task("steady_flow")
    dataset_root = tmp_path / "02_datasets"
    payload = _training_dataset_payload(task, dataset_id=_ID_DATASET, offsets=(0.0, 1.0, 4.0, 10.0))
    path = _save_dataset(dataset_root, dataset_root, payload)
    manifest = datasets.packages.load_package_manifest("steady_flow", _ID_DATASET, storage_root=tmp_path)
    assert manifest["dataset_digest"] == _ID_DIGEST
    assert manifest["sample_count"] == len(payload["sample_ids"])
    assert manifest["derived_payload_sha256"] is None
    external_payload = tmp_path / "external_payload.pt"
    path.rename(external_payload)
    path.symlink_to(external_payload)
    with pytest.raises(FileNotFoundError, match="payload is missing or unsafe"):
        datasets.packages.load_package_manifest("steady_flow", _ID_DATASET, storage_root=tmp_path)
    path.unlink()
    external_payload.rename(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="payload bytes disagree"):
        datasets.packages.load_package_manifest("steady_flow", _ID_DATASET, storage_root=tmp_path)


def _tiny_config(*, dataset_root: Path, output_root: Path) -> dict[str, Any]:
    """
    Resolve the smallest public one-epoch CPU FNO experiment used end to end.

    The recipe retains production semantic validation, splitting, normalization,
    metrics, checkpoints, and paths while reducing only model/data size and duration.
    """
    config = configs.resolve(_tiny_request())
    config["paths"]["dataset_root"] = str(dataset_root)
    config["paths"]["output_root"] = str(output_root)
    return config


def _tiny_request(*, workflow: str = "train") -> dict[str, Any]:
    """Return the request of the smallest one-epoch CPU FNO experiment on the ID and named OOD Datasets."""
    raw = configs.direct_config(model_kind="fno", physics_enabled=False, workflow=workflow)
    raw["run"].update(
        {
            "seed": 23,
            "device": "cpu",
        }
    )
    raw["data"].update(
        {
            "train_dataset": _ID_DATASET,
            "material_ood_datasets": [_OOD_DATASET, _SECOND_OOD_DATASET],
            "evaluation_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )
    raw["model"] = {
        "kind": "fno",
        "params": {
            "n_modes": [2, 2],
            "hidden_channels": 4,
            "n_layers": 1,
            "positional_embedding": None,
            "domain_padding": 0.1,
        },
    }
    raw["loss"]["data"].update(
        {
            "kind": "mse",
            "weight": 1.0,
        }
    )
    raw["loss"]["physics"] = {"enabled": False}
    raw["optimizer"].update(
        {
            "kind": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
        }
    )
    raw["scheduler"] = None
    raw["training"]["budget_epochs"] = 1
    raw["tracking"]["wandb"]["mode"] = "disabled"
    return raw


def _refresh_summary_digest(
    run_dir: Path,
    *,
    summary_key: str,
    artifact_path: Path,
) -> None:
    """
    Republish one run-summary file digest after an intentional payload mutation.

    This keeps the outer file-integrity layer valid so a test can isolate the deeper
    task, config, split, or checkpoint identity boundary it intends to corrupt.
    """
    summary = experiments.run.read_run_summary(run_dir)
    summary[summary_key] = common.serialization.file_sha256(artifact_path)
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        summary,
    )


def _make_last_checkpoint_distinct(run_dir: Path) -> None:
    """
    Mutate one numeric ``last`` weight and republish its authoritative digest.

    The checkpoint schema and run identity remain valid. Only model state changes so
    inference can prove it loads selection-only ``best`` rather than continuation ``last``.
    """
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    payload = copy.deepcopy(torch.load(last_path, map_location="cpu", weights_only=False))
    state = payload["model_state_dict"]
    changed = False
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            replacement = value.detach().clone()
            replacement.reshape(-1)[0] += 1.0
            state[name] = replacement
            changed = True
            break
    if not changed:
        message = "The FNO checkpoint contained no mutable numeric parameter."
        raise AssertionError(message)
    common.serialization.atomic_torch_save(payload, last_path)
    _refresh_summary_digest(
        run_dir,
        summary_key="last_checkpoint_sha256",
        artifact_path=last_path,
    )


def _nested_state_equal(left: Any, right: Any) -> bool:
    """
    Return exact equality for nested tensor, mapping, and sequence checkpoint state.

    Tensors compare on CPU without tolerance. The helper intentionally supports only
    structures used by model state dictionaries in this lifecycle fixture.
    """
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_nested_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_state_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True))
    return bool(left == right)


def _state_dict_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Return exact equality for two model state mappings."""
    return _nested_state_equal(left, right)


@pytest.fixture(scope="module")
def completed_lifecycle(tmp_path_factory: pytest.TempPathFactory) -> CompletedLifecycle:
    """
    Build ID/OOD datasets, train one tiny CPU epoch, and validate one completed run.

    The module-scoped fixture pays the bounded training cost once, verifies runtime
    session facts, then makes ``last`` observably distinct while preserving validity.
    It must not be treated as a performance or scientific-accuracy benchmark.
    """
    root = tmp_path_factory.mktemp("steady_flow_lifecycle")
    dataset_root = root / "02_datasets"
    metadata_root = dataset_root
    output_root = root / "processed"
    task = domain.tasks.registry.get_task("steady_flow")
    id_payload = _training_dataset_payload(
        task,
        dataset_id=_ID_DATASET,
        offsets=(0.0, 1.0, 4.0, 10.0),
    )
    ood_payload = _training_dataset_payload(
        task,
        dataset_id=_OOD_DATASET,
        offsets=(20.0, 21.0, 24.0, 30.0),
    )
    _save_dataset(dataset_root, metadata_root, id_payload)
    _save_dataset(dataset_root, metadata_root, ood_payload)
    _save_dataset(dataset_root, metadata_root, _training_dataset_payload(task, dataset_id=_SECOND_OOD_DATASET, offsets=(40.0, 41.0)))

    config = _tiny_config(dataset_root=dataset_root, output_root=output_root)
    assert config["tracking"]["wandb"]["mode"] == "disabled"
    run_dir = experiments.run.prepare_fresh_run(
        config,
        run_dir=common.paths.resolve_config_run_dir(config, output_root=output_root),
    )
    environment = pytest.MonkeyPatch()
    environment.delenv("WANDB_API_KEY", raising=False)
    epoch_metrics: list[dict[str, float]] = []
    try:
        experiments.run.execute_prepared_run(
            config,
            run_dir=run_dir,
            event_callback=lambda _step, metrics: epoch_metrics.append(dict(metrics)),
            device_resolution=learning.device.resolve_device("cpu"),
        )
    finally:
        environment.undo()
    assert not any(candidate.is_dir() for candidate in root.rglob("wandb"))
    experiments.run.validate_completed_run(run_dir)

    _make_last_checkpoint_distinct(run_dir)
    completed = experiments.run.validate_completed_run(run_dir)
    summary = completed["summary"]
    # One epoch of one two-sample optimizer group: a budget of two units, so the group crosses all 20 points and validates once.
    assert summary["training_work"] == {"optimizer_steps": 1, "gradient_predictions": 2, "units": 2, "update_distinct_cases": 2}
    assert summary["terminal_budget"]["unit"] == "units"
    assert summary["terminal_budget"]["total"] == summary["terminal_budget"]["units"] == 2
    assert summary["terminal_budget"]["validated_point"] == 20
    assert summary["validation_plan_deviation"] == {"skipped_points": list(range(1, 20))}
    assert [metrics["validation/point"] for metrics in epoch_metrics if "validation/point" in metrics] == [20.0]
    assert summary["runtime_device"]["requested_policy"] == "cpu"
    assert summary["runtime_device"]["resolved_device"] == "cpu"
    assert len(summary["runtime_sessions"]) == 1
    assert summary["runtime_sessions"][0]["requested_policy"] == "cpu"
    assert summary["runtime_sessions"][0]["resolved_device"] == "cpu"
    # The real run records its precision after the model build and ends its one attempt with the committed work.
    attempt = summary["runtime_sessions"][0]["attempt"]
    assert summary["runtime_sessions"][0]["precision"]["parameter_dtypes"]
    assert attempt["work_at_end"] == {**summary["training_work"], "train_seconds": summary["terminal_budget"]["train_seconds"]}
    assert (summary["duration_coverage"], summary["attempted_train_seconds"]) == ("complete", attempt["train_seconds"])
    assert summary["training_units"]["seen_cases"] == summary["training_units"]["train_cases"] == 2
    assert not _state_dict_equal(
        completed["best_checkpoint"]["model_state_dict"],
        completed["last_checkpoint"]["model_state_dict"],
    )
    return CompletedLifecycle(
        config=config,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        run_dir=run_dir,
        id_payload=id_payload,
        ood_payload=ood_payload,
        completed=completed,
        epoch_metrics=epoch_metrics,
    )


def _artifact_inventory(targets: tuple[Path, ...]) -> dict[Path, tuple[str, int]]:
    """
    Return SHA-256 and nanosecond write-time identity for every file below targets.

    Tests use both values to prove cache reuse is mutation-free and rebuild changes
    only selected artifact roots.
    """
    return {
        path: (common.serialization.file_sha256(path), path.stat().st_mtime_ns)
        for target in targets
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def test_real_steady_flow_lifecycle_and_artifacts(  # noqa: PLR0915
    completed_lifecycle: CompletedLifecycle,
) -> None:
    """
    Execute the bounded synthetic lifecycle from training through artifact rebuild.

    Saved splits and the Train-only scaling must reconstruct exactly, inference must
    load ``best`` rather than the deliberately distinct ``last``, and ID/OOD artifacts
    must expose current metrics, physics arrays, provenance, and normalized evidence.
    Valid caches remain byte/time-identical. Corrupt provenance, Parquet, and NPZ
    content fail read-only. Explicit rebuild replaces only selected targets. This
    protects the integration seams without substituting for unit formula/race coverage.
    """
    lifecycle = completed_lifecycle
    split = datasets.preprocessing.splits.admit_split(lifecycle.completed["split_indices"])
    seed_plan = experiments.run.build_seed_plan(lifecycle.config["run"]["seed"], stage=None)
    rebuilt = experiments.config.loader.create_dataloaders_from_config(
        lifecycle.config,
        seed_plan=seed_plan,
    )
    assert datasets.preprocessing.splits.admit_split(rebuilt["split_indices"]) == split
    assert split.train_subset is None
    assert dict(split.case_partition) == _ID_PROVENANCE["case_partition"]
    assert [part.dataset_id for part in split.material_ood] == [_OOD_DATASET, _SECOND_OOD_DATASET]

    # The curated W&B series have real producers; physics training and CUDA are off in this run, so those series are not projected.
    # Training evaluates no held-out role, so the run projects no OOD series.
    config = lifecycle.config
    definitions = experiments.tracking.automatic_history_metric_definitions(
        config["evaluation"]["metrics"],
        objective_id=str(config["evaluation"]["objective"]["id"]),
        physics_training_enabled=bool(config["loss"]["physics"]["enabled"]),
        physics_monitor_enabled=bool(config["tracking"]["wandb"]["monitor"]["enabled"]),
        cuda_enabled=False,
        task_id="steady_flow",
    )
    # The material part is declared, yet training produced no series of it or of any other held-out role.
    assert split.ood_roles == ("material_ood",)
    held_out = ("id_test", "material_ood", "parameter_ood")
    assert not any(role in key for epoch in lifecycle.epoch_metrics for key in epoch for role in held_out)
    assert not any(role in key for key in lifecycle.completed["summary"]["selected_metrics"] for role in held_out)
    # Undefined on this tiny data only: every case's p, u and v reference RMS lies below its TaskSpec floor
    # (domain_task_steady_flow.py: 1 Pa, 1e-4 m/s), so physical_relative_l2 has no finite value and is absent from
    # the event. Its producer is covered by test_metric_spaces.py::test_validation_produces_the_curated_relative_l2_above_the_reference_floors.
    undefined = {"id/physical_relative_l2"}
    assert {definition.source_key for definition in definitions} - undefined <= set().union(*lifecycle.epoch_metrics)
    projected_selected = {wandb_schema.selected_summary_key(definition.source_key) for definition in definitions} - {None}
    projected_selected -= {wandb_schema.selected_summary_key(key) for key in undefined}
    assert projected_selected <= set(lifecycle.completed["summary"]["selected_metrics"])

    train_indices = split.train.indices
    normalizer_artifact = torch.load(
        common.paths.resolve_normalizer_path(lifecycle.run_dir),
        map_location="cpu",
        weights_only=False,
    )
    scaling_identity = normalizer_artifact["identity"]
    assert scaling_identity["dataset"]["dataset_id"] == split.train.dataset_id
    assert scaling_identity["dataset"]["fingerprint"] == split.train.dataset["fingerprint"]
    assert scaling_identity["train_membership_digest"] == split.train.membership_digest
    scaling = datasets.preprocessing.normalization.TrainScaling.from_state_dict(normalizer_artifact["state"])
    task = domain.tasks.registry.get_task(str(lifecycle.config["task"]))
    output_units = tuple(field.unit for field in task.outputs)
    velocity_group = next(group for group in task.output_groups if group.id == "velocity")
    velocity_units = {task.field(field).unit for field in velocity_group.fields}
    assert len(velocity_units) == 1
    speed_unit = next(iter(velocity_units))
    objective_id = str(lifecycle.config["evaluation"]["objective"]["id"])
    train_scales = analysis.artifacts.contracts.output_scales(scaling, output_fields=task.output_names)
    # Every steady case is one equal-size sample, so the case-equal Train means are the pooled Train means.
    input_mean, _ = scaling.input_affine()
    fitted_inputs = [index for index, name in enumerate(task.input_names) if name not in datasets.preprocessing.normalization.STEADY_BOUNDARY_STATES]
    expected_input_mean = lifecycle.id_payload["inputs"][train_indices].mean(dim=(0, 2, 3))
    expected_output_mean = lifecycle.id_payload["outputs"][train_indices].mean(dim=(0, 2, 3))
    assert torch.allclose(input_mean[fitted_inputs], expected_input_mean[fitted_inputs])
    assert torch.allclose(scaling.physical_state.mean, expected_output_mean)
    full_input_mean = lifecycle.id_payload["inputs"].mean(dim=(0, 2, 3))
    eps_index = task.input_names.index("eps_bed")
    assert not torch.isclose(input_mean[eps_index], full_input_mean[eps_index])

    model, loader, restored_scaling, device = learning.inference.context.load_inference_context(
        run_dir=lifecycle.run_dir,
        dataset_root=lifecycle.dataset_root,
        split="id_test",
        batch_size=1,
        device_policy="cpu",
    )
    assert device.type == "cpu"
    selected_dataset = loader.dataset
    assert isinstance(selected_dataset, learning.inference.context.IndexedSubset)
    assert torch.equal(selected_dataset.source_indices, split.id_test.indices)
    assert restored_scaling.digest == scaling.digest
    loaded_state = model.state_dict()
    best_state = lifecycle.completed["best_checkpoint"]["model_state_dict"]
    last_state = lifecycle.completed["last_checkpoint"]["model_state_dict"]
    assert _state_dict_equal(loaded_state, best_state)
    assert not _state_dict_equal(loaded_state, last_state)

    generated = analysis.artifacts.service.build_artifacts(
        runs_root=lifecycle.run_dir,
        dataset_root=lifecycle.dataset_root,
        device_policy="cpu",
        roles=("id", "material_ood"),
    )
    frames = generated[str(lifecycle.run_dir.resolve())]
    assert set(frames) == {"eval", "material_ood"}
    role_evidence: dict[analysis.artifacts.plan.ArtifactSplit, Any] = {"eval": (split.id_test,), "material_ood": split.material_ood}
    plan = analysis.artifacts.plan.artifact_plan_from_completed(lifecycle.completed, run_dir=lifecycle.run_dir)
    for role, evidence in role_evidence.items():
        frame = frames[role]
        assert not frame.empty
        assert frame.columns.is_unique
        assert frame["source_index"].tolist() == list(plan.role(role).source_indices)
        assert frame["source_dataset_id"].tolist() == [part.dataset_id for part in evidence for _ in part.case_ids]
        assert all(Path(path).is_file() for path in frame["npz_path"])
        case_statistics = {column for field in task.output_names for column in analysis.artifacts.contracts.case_statistic_columns(field)}
        assert {
            *analysis.artifacts.contracts.case_metric_columns(task.output_names),
            "physical_rmse_speed_magnitude",
            "momentum_residual_mse",
            "flux_balance_mse",
            "pressure_boundary_mse",
            "pressure_inlet_mse",
            "pressure_outlet_mean_square",
            *case_statistics,
        }.issubset(frame.columns)
        with np.load(Path(frame.iloc[0]["npz_path"]), allow_pickle=False) as payload:
            assert payload["output_fields"].tolist() == list(task.output_names)
            assert payload["artifact_fields"].tolist() == [*task.output_names, "U"]
            assert payload["artifact_units"].tolist() == [*output_units, speed_unit]
            assert {"Rx", "Ry", "flux_balance", "coordinates"}.issubset(payload.files)
            assert frame.iloc[0]["flux_balance_mse"] == pytest.approx(float(np.mean(payload["flux_balance"] ** 2)))
        assert frame.attrs["output_units"] == output_units
        artifact_provenance = frame.attrs["artifact_provenance"]
        artifact_summary = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
            frame,
            output_groups=task.output_groups,
            train_scales=train_scales,
            reference_floors={field: task.reference_floor(field) for field in task.output_names},
        )
        assert artifact_provenance["normalizer"]["output_scales"] == train_scales
        assert frame.attrs[objective_id] == artifact_summary

    id_target = common.paths.resolve_id_analysis_dir(lifecycle.run_dir)
    ood_target = common.paths.resolve_ood_analysis_dir(lifecycle.run_dir, "material_ood", _MATERIAL_OOD_DATASET)
    targets = (id_target, ood_target)
    assert (id_target / "artifacts.parquet").is_file()
    assert (ood_target / "artifacts.parquet").is_file()
    loaded_id = analysis.evaluation.artifact_loader._read_artifact_root(id_target)
    assert loaded_id.attrs["provenance_complete"] is True
    assert loaded_id.attrs["artifact_root"] == str(id_target.resolve())
    assert loaded_id["source_index"].tolist() == list(split.id_test.index_values)
    for role, target in zip(("eval", "material_ood"), targets, strict=True):
        stored_provenance = json.loads((target / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
        assert stored_provenance["outputs"] == analysis.artifacts.contracts.artifact_output_manifest(target)
        assert stored_provenance["run"]["normalizer_sha256"] == lifecycle.completed["summary"]["normalizer_sha256"]
        assert stored_provenance["evaluator"]["objective"] == lifecycle.completed["config"]["evaluation"]["objective"]
        assert stored_provenance["physics"] == analysis.artifacts.contracts.steady_physics_provenance(task)
        assert stored_provenance["runtime"]["requested_policy"] == "cpu"
        assert stored_provenance["runtime"]["resolved_device"] == "cpu"
        assert stored_provenance["run"]["best_checkpoint_sha256"] == lifecycle.completed["summary"]["best_checkpoint_sha256"]
        assert stored_provenance["dataset"]["membership_digests"] == [part.membership_digest for part in role_evidence[role]]
        assert stored_provenance["dataset"]["source_identities"] == [dict(part.dataset) for part in role_evidence[role]]
        assert stored_provenance["normalizer"]["sha256"] == lifecycle.completed["summary"]["normalizer_sha256"]
        assert stored_provenance["evaluator"]["metric_evidence"] == analysis.artifacts.contracts.metric_evidence_provenance(task)
        physics = stored_provenance["physics"]
        assert physics["task_contract_digest"] == domain.tasks.registry.get_task("steady_flow").contract_digest
        assert physics["derivatives"] == {
            "kind": "physical",
            "operator_axes": [2, 3],
            "grid_axes": ["y", "x"],
        }
        assert physics["interior_crop"] == _ARTIFACT_CROP
        assert physics["constants"]["dynamic_viscosity_pa_s"] == domain.physics.contracts.AIR_DYNAMIC_VISCOSITY
        assert physics["continuity_kind"] == "cross_section_flux_balance"
        assert physics["scalar_definitions"]["flux_balance_mse"] == {"formula": "mean(flux_balance**2)", "unit": "1"}
        assert physics["residual_evaluation_region"]["residual_arrays"] == "full grid"
        timing_block = analysis.artifacts.timing.timing_block(stored_provenance, records=frozenset({"report"}))
        runtime_comparison = timing_block["report"]
        assert runtime_comparison["split_role"] == role
        assert runtime_comparison["measurement"] == {
            "clock": "time.perf_counter_ns",
            "batch_size": lifecycle.config["data"]["evaluation_batch_size"],
            "case_duration_attribution": analysis.artifacts.timing.CASE_DURATION_ATTRIBUTION,
            "warmup_passes": 1,
            "cuda_synchronized": False,
        }
        assert len(runtime_comparison["cases"]) == len(frames[role])
        assert runtime_comparison["speedups"]["solver_speedup"]["paired_case_count"] == len(frames[role])
        assert runtime_comparison["solver_hardware"] == [{"hostname": "solver-node", "requested_cores": 8, "case_count": len(frames[role])}]

    id_provenance = json.loads((id_target / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert id_provenance["selection"]["index_key"] == "id_test_indices"
    assert id_provenance["selection"]["effective_case_count"] == split.id_test.count

    for target in targets:
        (target / "cache_marker.txt").write_text("preserve", encoding="utf-8")
    before_cache = _artifact_inventory(targets)
    cached = analysis.artifacts.service.build_artifacts(
        runs_root=lifecycle.run_dir,
        dataset_root=lifecycle.dataset_root,
        device_policy="cpu",
        roles=("id", "material_ood"),
    )
    pd.testing.assert_frame_equal(cached[str(lifecycle.run_dir.resolve())]["eval"], frames["eval"])
    pd.testing.assert_frame_equal(cached[str(lifecycle.run_dir.resolve())]["material_ood"], frames["material_ood"])
    assert _artifact_inventory(targets) == before_cache

    provenance_path = id_target / analysis.artifacts.contracts.ARTIFACT_PROVENANCE_FILENAME
    valid_provenance = provenance_path.read_text(encoding="utf-8")
    for remove_timing in (True, False):
        altered_provenance = json.loads(valid_provenance)
        if remove_timing:
            del altered_provenance[analysis.artifacts.timing.TIMING_KEY]
        else:
            altered_provenance[analysis.artifacts.timing.TIMING_KEY]["report"]["split_role"] = "altered"
        common.serialization.atomic_write_json(provenance_path, altered_provenance)
        altered_timing_cache = _artifact_inventory((id_target,))
        with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match="--rebuild"):
            analysis.artifacts.service.run_or_load_artifacts(
                run_dir=lifecycle.run_dir,
                dataset_name=_ID_DATASET,
                split="eval",
                device_resolution=learning.device.resolve_device("cpu"),
                dataset_root=lifecycle.dataset_root,
            )
        assert _artifact_inventory((id_target,)) == altered_timing_cache
        provenance_path.write_text(valid_provenance, encoding="utf-8")

    provenance_path.write_text("{}\n", encoding="utf-8")
    incompatible_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifacts.service.ArtifactCacheError,
        match="provenance",
    ):
        analysis.artifacts.service.run_or_load_artifacts(
            run_dir=lifecycle.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=lifecycle.dataset_root,
        )
    assert _artifact_inventory((id_target,)) == incompatible_cache
    provenance_path.write_text(valid_provenance, encoding="utf-8")

    parquet_path = id_target / "artifacts.parquet"
    valid_parquet = parquet_path.read_bytes()
    parquet_path.write_bytes(valid_parquet + b"corrupt")
    corrupted_parquet_cache = _artifact_inventory((id_target,))
    with pytest.raises(
        analysis.artifacts.service.ArtifactCacheError,
        match="output manifest does not match",
    ):
        analysis.artifacts.service.run_or_load_artifacts(
            run_dir=lifecycle.run_dir,
            dataset_name=_ID_DATASET,
            split="eval",
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=lifecycle.dataset_root,
        )
    assert _artifact_inventory((id_target,)) == corrupted_parquet_cache
    parquet_path.write_bytes(valid_parquet)

    # The load admits the payload structure only; a changed case payload is refused by its digest when the case opens.
    corrupted_npz = Path(frames["eval"].iloc[0]["npz_path"])
    corrupted_npz.write_bytes(corrupted_npz.read_bytes() + b"corrupt")
    corrupted_cache = _artifact_inventory((id_target,))
    loaded = analysis.artifacts.service.run_or_load_artifacts(
        run_dir=lifecycle.run_dir,
        dataset_name=_ID_DATASET,
        split="eval",
        device_resolution=learning.device.resolve_device("cpu"),
        dataset_root=lifecycle.dataset_root,
    )
    with pytest.raises(ValueError, match="Steady case payload digest changed"):
        analysis.evaluation.case.load_case(loaded, 0)
    assert _artifact_inventory((id_target,)) == corrupted_cache

    sibling_marker = common.paths.resolve_ood_analysis_dir(lifecycle.run_dir, "material_ood", "unselected_ood") / "keep.txt"
    sibling_marker.parent.mkdir(parents=True)
    sibling_marker.write_text("keep", encoding="utf-8")
    rebuilt_artifacts = analysis.artifacts.service.build_artifacts(
        runs_root=lifecycle.run_dir,
        dataset_root=lifecycle.dataset_root,
        device_policy="cpu",
        roles=("id", "material_ood"),
        rebuild=True,
    )
    for target in targets:
        assert not (target / "cache_marker.txt").exists()
    assert sibling_marker.read_text(encoding="utf-8") == "keep"
    pd.testing.assert_frame_equal(
        rebuilt_artifacts[str(lifecycle.run_dir.resolve())]["eval"],
        frames["eval"],
    )
    pd.testing.assert_frame_equal(
        rebuilt_artifacts[str(lifecycle.run_dir.resolve())]["material_ood"],
        frames["material_ood"],
    )


def _copy_without_analysis(source: Path, destination: Path) -> Path:
    """Copy one temporary run bundle without retaining prior artifact state."""
    shutil.copytree(source, destination)
    analysis_root = common.paths.resolve_analysis_root(destination)
    if analysis_root.exists():
        shutil.rmtree(analysis_root)
    return destination


def test_additional_part_whose_package_changed_since_training_is_refused(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
) -> None:
    """Refuse the material-OOD role when one part's current package differs from its saved split identity."""
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "run")
    dataset_root = tmp_path / "02_datasets"
    shutil.copytree(completed_lifecycle.dataset_root, dataset_root)
    task = domain.tasks.registry.get_task("steady_flow")
    _save_dataset(dataset_root, dataset_root, _training_dataset_payload(task, dataset_id=_SECOND_OOD_DATASET, offsets=(40.0, 42.0)))
    with pytest.raises(RuntimeError, match=rf"{_SECOND_OOD_DATASET!r} contradicts its saved split Dataset identity"):
        analysis.artifacts.service.run_or_load_artifacts(
            run_dir=run_dir,
            dataset_name=_MATERIAL_OOD_DATASET,
            split="material_ood",
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=dataset_root,
        )
    assert not common.paths.resolve_ood_analysis_dir(run_dir, "material_ood", _MATERIAL_OOD_DATASET).exists()


def _rewrite_parquet_npz_paths(artifact_root: Path, values: list[str]) -> None:
    """Rewrite test-owned path rows and refresh their exact payload manifest."""
    provenance_path = analysis.artifacts.contracts.artifact_provenance_path(artifact_root)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    parquet_name = provenance["outputs"]["parquet"]["path"]
    parquet_path = artifact_root / parquet_name
    frame = pd.read_parquet(parquet_path)
    assert len(frame) == len(values)
    frame.loc[:, "npz_path"] = values
    frame.to_parquet(parquet_path, index=False)
    provenance["outputs"] = analysis.artifacts.contracts.artifact_output_manifest(artifact_root)
    common.serialization.atomic_write_json(provenance_path, provenance)


def test_completed_bundle_is_portable_with_exact_relative_payload_paths(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rename, generate, move, and reuse one completed bundle without identity drift."""
    original = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "original")
    renamed = tmp_path / "paper_model_a"
    original.rename(renamed)
    contexts = (
        analysis.evaluation.workflow.EvaluationContextSpec(key="id", label="ID", artifact_role="id"),
        analysis.evaluation.workflow.EvaluationContextSpec(key="material_ood", label="Material OOD", artifact_role="material_ood"),
    )
    selection = (analysis.evaluation.workflow.EvaluationRunSelection(run_dir=renamed, label="Paper model A"),)
    # The workflow only reads: a held-out role is refused with its explicit artifact command until that call releases it.
    with pytest.raises(analysis.evaluation.artifact_loader.MissingEvaluationArtifactsError, match="--split id"):
        analysis.evaluation.workflow.prepare_evaluation_workflow(selection, contexts)
    analysis.artifacts.service.build_artifacts(
        runs_root=renamed, dataset_root=completed_lifecycle.dataset_root, device_policy="cpu", roles=("id", "material_ood")
    )
    evaluation = analysis.evaluation.workflow.prepare_evaluation_workflow(selection, contexts)
    prepared = evaluation.prepared_runs[0]
    first = prepared.loaded_run
    assert prepared.role_actions == {"id": "reused", "material_ood": "reused"}
    assert first.storage_alias == "paper_model_a"
    assert first.scientific_run_name == completed_lifecycle.config["run"]["name"]
    assert first.id_artifact is not None
    assert first.material_ood_artifact is not None
    assert evaluation.report[0]["artifact_actions"] == {"id": "reused", "material_ood": "reused"}
    checkpoint_digest = first.selected_checkpoint_sha256
    evaluation.close()

    id_root = first.id_artifact.root
    raw_id = pd.read_parquet(id_root / "artifacts.parquet")
    relative_paths = tuple(Path(value).parts for value in raw_id["npz_path"])
    expected_paths = tuple(("npz", f"{_SAMPLE_IDS[int(source_index)]}.npz") for source_index in raw_id["source_index"])
    assert relative_paths == expected_paths
    moved = tmp_path / "archive" / "promoted_model"
    moved.parent.mkdir()
    renamed.rename(moved)

    def unexpected_inference(**_kwargs: Any) -> Any:
        msg = "valid relocated artifacts must not reconstruct a model"
        raise AssertionError(msg)

    original_inference = learning.inference.context.load_inference_context_with_resolution
    monkeypatch.setattr(learning.inference.context, "load_inference_context_with_resolution", unexpected_inference)
    reused = analysis.artifacts.service.load_prepared_run_artifacts(moved)
    loaded = reused.loaded_run
    assert reused.role_actions == {"id": "reused", "material_ood": "reused"}
    assert loaded.storage_alias == "promoted_model"
    assert loaded.scientific_run_name == first.scientific_run_name
    assert loaded.id_artifact is not None
    assert loaded.material_ood_artifact is not None
    assert loaded.selected_checkpoint_sha256 == checkpoint_digest
    assert all(Path(value).is_relative_to(loaded.id_artifact.root) for value in loaded.id_artifact.frame["npz_path"])

    with monkeypatch.context() as drift:
        drift.setattr(learning.inference.context, "load_inference_context_with_resolution", original_inference)
        # Training code changed after the run: reading it stays admitted on its recorded evidence; continuing the
        # completed run is refused as terminal before any code check.
        run_state.drift_training_code(drift, "steady_flow")
        before_drift = _artifact_inventory((moved,))
        experiments.run.validate_completed_run(moved)
        experiments.run.validate_evaluable_run(moved)
        learning.inference.context.load_inference_context_with_resolution(
            run_dir=moved,
            dataset_root=completed_lifecycle.dataset_root,
            split="validation",
            device_resolution=learning.device.resolve_device("cpu"),
        )
        assert analysis.evaluation.artifact_loader.load_run_artifacts(moved).selected_checkpoint_sha256 == checkpoint_digest
        # Exercise the same continuation service used by ordinary resolved launches.
        with pytest.raises(
            experiments.run.ContinuationRefusedError, match="A completed run, also one that diverged before its budget end, is never extended"
        ):
            experiments.run._run_resolved_experiment(
                copy.deepcopy(completed_lifecycle.config),
                config_path=common.paths.resolve_run_config_path(moved),
                resume=moved,
                device_resolution=learning.device.resolve_device("cpu"),
            )
        assert _artifact_inventory((moved,)) == before_drift

    with monkeypatch.context() as drift:
        environment = learning.training.checkpoint.numerical_environment_identity()
        drift.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: {**environment, "torch": "0.0.0"})
        # Another numerical environment is the reader's environment, not the run's state.
        with pytest.raises(learning.training.checkpoint.ResumeEnvironmentError, match="differs in torch"):
            experiments.run.validate_evaluable_run(moved)
        before_read = _artifact_inventory((moved,))
        reread = analysis.evaluation.artifact_loader.load_run_artifacts(moved)
        assert reread.id_artifact is not None
        assert reread.material_ood_artifact is not None
        pd.testing.assert_frame_equal(reread.id_artifact.frame, loaded.id_artifact.frame)
        pd.testing.assert_frame_equal(reread.material_ood_artifact.frame, loaded.material_ood_artifact.frame)
        assert reread.selected_checkpoint_sha256 == checkpoint_digest
        assert _artifact_inventory((moved,)) == before_read
        prepared_read = analysis.artifacts.service.load_prepared_run_artifacts(moved)
        assert prepared_read.role_actions == {"id": "reused", "material_ood": "reused"}
        assert prepared_read.loaded_run.selected_checkpoint_sha256 == checkpoint_digest

        # Reader contracts still bind the published evaluator under numerical-environment drift.
        provenance_path = analysis.artifacts.contracts.artifact_provenance_path(loaded.id_artifact.root)
        original_provenance = provenance_path.read_bytes()
        provenance = json.loads(original_provenance)
        provenance["evaluator"]["input_units"] = {}
        common.serialization.atomic_write_json(provenance_path, provenance)
        with pytest.raises(analysis.evaluation.artifact_loader.IncompatibleEvaluationArtifactsError):
            analysis.evaluation.artifact_loader.load_run_artifacts(moved)
        provenance_path.write_bytes(original_provenance)
        parquet_path = loaded.id_artifact.root / "artifacts.parquet"
        original_parquet = parquet_path.read_bytes()
        parquet_path.write_bytes(original_parquet + b"changed")
        with pytest.raises(analysis.evaluation.artifact_loader.IncompatibleEvaluationArtifactsError):
            analysis.evaluation.artifact_loader.load_run_artifacts(moved)
        parquet_path.write_bytes(original_parquet)
        absent_ood = loaded.material_ood_artifact.root.with_name("retained-ood")
        loaded.material_ood_artifact.root.rename(absent_ood)
        with pytest.raises(analysis.evaluation.artifact_loader.MissingEvaluationArtifactsError):
            analysis.artifacts.service.load_prepared_run_artifacts(moved, artifact_roles=("material_ood",))
        absent_ood.rename(loaded.material_ood_artifact.root)

    raw_moved = pd.read_parquet(loaded.id_artifact.root / "artifacts.parquet")
    escaping = raw_moved["npz_path"].tolist()
    escaping[0] = f"../npz/{_SAMPLE_IDS[int(raw_moved.iloc[0]['source_index'])]}.npz"
    _rewrite_parquet_npz_paths(loaded.id_artifact.root, escaping)
    with pytest.raises(analysis.evaluation.artifact_loader.IncompatibleEvaluationArtifactsError, match=r"npz_path|relative NPZ"):
        analysis.evaluation.artifact_loader.load_run_artifacts(moved)
    shutil.rmtree(loaded.material_ood_artifact.root)
    with pytest.raises(analysis.evaluation.artifact_loader.MissingEvaluationArtifactsError):
        analysis.evaluation.artifact_loader.load_run_artifacts(moved, artifact_roles=("material_ood",))


def _rewrite_as_optuna_trial(run_dir: Path) -> dict[str, Any]:
    """Rewrite a test-owned completed copy with coherent Optuna-trial identity."""
    config_path = common.paths.resolve_run_config_path(run_dir)
    config = experiments.config.loader.load_yaml(config_path)
    config["tracking"]["wandb"]["workflow"] = "optuna_trial"
    config["tracking"]["wandb"].update(experiments.config.loader.derive_wandb_organization(config))
    config["run"]["layout"].update({"family": "optuna", "config_name": "portable_fixture_study"})
    trial = experiments.config.naming.trial_leaf(7, config, {})
    config = experiments.config.loader.finalize_run_layout(config, trial=trial)
    config = experiments.config.loader.validate_resolved_config(config)
    experiments.config.loader.save_yaml(config, config_path)
    split = torch.load(common.paths.resolve_split_indices_path(run_dir), map_location="cpu", weights_only=False)
    identity = learning.training.checkpoint.build_checkpoint_identity(
        config,
        split,
        normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
        persisted_config=config,
    )
    for checkpoint_path in (
        common.paths.resolve_best_checkpoint_file(run_dir),
        common.paths.resolve_last_checkpoint_file(run_dir),
    ):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["identity"] = copy.deepcopy(identity)
        common.serialization.atomic_torch_save(payload, checkpoint_path)
    # The history is bound to the run name and checkpoint identity like the checkpoints.
    history_path = learning.training.history.history_path(run_dir)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history.update(run_name=config["run"]["name"], checkpoint_identity_digest=common.serialization.canonical_json_sha256(identity))
    common.serialization.atomic_write_json(history_path, history)
    summary = experiments.run.read_run_summary(run_dir)
    summary.update(
        {
            "run_name": config["run"]["name"],
            "effective_config_digest": identity["effective_config_digest"],
            "checkpoint_identity": copy.deepcopy(identity),
            "config_sha256": common.serialization.file_sha256(config_path),
            "best_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_best_checkpoint_file(run_dir)),
            "last_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_last_checkpoint_file(run_dir)),
            "study_name": "portable_fixture_study",
            "trial_number": 7,
            "sampled_parameters": {"hidden_channels": 4},
            "overrides": {"optimizer.lr": 1.0e-3},
            "study_role": "optimization",
        }
    )
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    return config


def test_completed_optuna_trial_moved_outside_its_study_stays_a_trial_without_artifacts(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
) -> None:
    """A moved, renamed trial keeps its origin metadata and is refused by artifact planning like a transient trial."""
    trial_dir = _copy_without_analysis(
        completed_lifecycle.run_dir,
        tmp_path / "study" / "trials" / "trial_000007",
    )
    _rewrite_as_optuna_trial(trial_dir)
    original_summary = experiments.run.read_run_summary(trial_dir)
    moved = tmp_path / "promoted" / "paper_trial_b"
    moved.parent.mkdir()
    trial_dir.rename(moved)

    completed = experiments.run.validate_completed_run(moved)
    for key in ("study_name", "trial_number", "sampled_parameters", "overrides", "study_role"):
        assert completed["summary"][key] == original_summary[key]
    with pytest.raises(ValueError, match="is an Optuna trial"):
        analysis.artifacts.service.build_artifacts(runs_root=moved, dataset_root=completed_lifecycle.dataset_root, device_policy="cpu", roles=("id",))
    assert not common.paths.resolve_analysis_root(moved).exists()


def test_a_finished_trial_without_checkpoints_is_admitted_in_another_numerical_environment(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial's history stays bound to the producer identity its summary records, not to the reading container's."""
    trial_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "study" / "trials" / "trial_000007")
    _rewrite_as_optuna_trial(trial_dir)
    # A finished trial writes no last checkpoint and deletes its best; its summary keeps the best digest.
    common.paths.resolve_last_checkpoint_file(trial_dir).unlink()
    common.paths.resolve_best_checkpoint_file(trial_dir).unlink()
    environment = learning.training.checkpoint.numerical_environment_identity()
    monkeypatch.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: {**environment, "sif_sha256": "0" * 64})

    completed = experiments.run.validate_completed_run(trial_dir)

    recorded = completed["summary"]["checkpoint_identity"]
    assert recorded["numerical_environment"] == environment
    assert completed["history"]["checkpoint_identity_digest"] == common.serialization.canonical_json_sha256(recorded)


def test_optuna_trial_records_the_dry_run_signature_and_validates_without_patches(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-trial steady study binds the Dataset evidence of its dry run, and its trial is admitted like any completed run."""
    monkeypatch.setenv("STORAGE_ROOT", str(completed_lifecycle.dataset_root.parent))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    request = configs.optuna_config()
    request["study"].update(n_trials=1, pruner={"kind": "none"})
    request["experiment"] = _tiny_request(workflow="optuna_trial")
    request["search_space"] = {"optimizer.lr": {"name": "learning_rate", "kind": "float", "low": 1.0e-3, "high": 2.0e-3, "log": True}}

    completed = optuna_trials.run_one_trial_study(request, directory=tmp_path)

    # Completed-run validation admitted the saved normalizer as the steady run artifact.
    assert completed["config"]["data"]["train_dataset"] == _ID_DATASET


def test_bundle_without_best_checkpoint_is_not_evaluable(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
) -> None:
    """Reject a bundle whose selected best checkpoint is absent."""
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "missing_best")
    common.paths.resolve_best_checkpoint_file(run_dir).unlink()
    with pytest.raises(experiments.run.RunLifecycleError, match="no valid best checkpoint"):
        experiments.run.validate_evaluable_run(run_dir)


def _interrupted_copy(completed: CompletedLifecycle, destination: Path) -> Path:
    """Copy the completed bundle to ``destination`` and mark the copy interrupted after its first epoch."""
    run_dir = _copy_without_analysis(completed.run_dir, destination)
    summary = experiments.run.read_run_summary(run_dir)
    summary["status"] = "interrupted"
    summary["status_history"][-1]["status"] = "interrupted"
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    return run_dir


def _authored_request(run_dir: Path, tmp_path: Path) -> Path:
    """Write the authored request whose resolution names ``run_dir``: its saved config without the resolved fields."""
    saved = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(run_dir))
    raw = copy.deepcopy(saved)
    for key in ("task_contract", "paths", "producer"):
        raw.pop(key)
    for key in ("name", "layout"):
        raw["run"].pop(key)
    for key in ("project", "entity", "tags"):
        raw["tracking"]["wandb"].pop(key)
    raw["model"]["params"].pop("in_channels")
    raw["model"]["params"].pop("out_channels")
    raw["evaluation"] = {"objective": {"id": saved["evaluation"]["objective"]["id"]}}
    return configs.write_config(tmp_path, raw, canonical_root=tmp_path)


@pytest.mark.parametrize("status", ["interrupted", "completed", "completed_without_history"])
def test_rerunning_an_authored_request_finishes_the_run_in_place(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    status: str,
) -> None:
    """The same command finalizes an interrupted run whose budget is spent, or returns a completed one, in its own directory.

    A completed run without its history fails the completed-run admission: it is neither reported complete nor bound.
    """
    output_root = tmp_path / "processed"
    destination = common.paths.resolve_config_run_dir(completed_lifecycle.config, output_root=output_root)
    if status == "interrupted":
        _interrupted_copy(completed_lifecycle, destination)
    else:
        _copy_without_analysis(completed_lifecycle.run_dir, destination)
    saved = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(destination))
    request_path = _authored_request(destination, tmp_path)
    if status == "completed_without_history":
        learning.training.history.history_path(destination).unlink()
        before = _artifact_inventory((tmp_path,))
        with pytest.raises(experiments.run.RunLifecycleError, match="history"):
            experiments.run.run_experiment(request_path, device="cpu", output_root=output_root)
        # Neither the identity binding below the output root nor a writer lock below the storage root was written.
        assert _artifact_inventory((tmp_path,)) == before
        return

    continued = experiments.run.run_experiment(request_path, device="cpu", output_root=output_root)

    assert continued["run_dir"] == destination.resolve()
    assert continued.get("already_complete", False) is (status == "completed")
    assert [child.name for child in destination.parent.iterdir()] == [destination.name]
    completed = experiments.run.validate_completed_run(destination)
    assert completed["config"]["run"]["name"] == saved["run"]["name"]
    # The spent budget admits no further optimizer group.
    assert completed["last_checkpoint"]["completed_epoch"] == 1
    assert completed["summary"]["training_work"]["optimizer_steps"] == 1


def test_saved_config_resume_continues_a_renamed_interrupted_bundle(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
) -> None:
    """Resume one interrupted bundle by its exact renamed path from its own saved config; another budget is refused."""
    renamed = _interrupted_copy(completed_lifecycle, tmp_path / "renamed_resume_bundle")
    saved_name = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(renamed))["run"]["name"]
    with pytest.raises(experiments.run.ContinuationRefusedError, match="cannot override scientific"):
        experiments.run.run_experiment(common.paths.resolve_run_config_path(renamed), resume=renamed, device="cpu", epochs=2)

    resumed = experiments.run.run_experiment(common.paths.resolve_run_config_path(renamed), resume=renamed, device="cpu")

    assert resumed["run_dir"] == renamed.resolve()
    completed = experiments.run.validate_completed_run(renamed)
    assert completed["config"]["run"]["name"] == saved_name
    assert completed["last_checkpoint"]["completed_epoch"] == 1


def test_steady_run_persists_its_completed_epoch_history(completed_lifecycle: CompletedLifecycle) -> None:
    """A completed steady run binds one record per completed epoch and per validation point to its identity."""
    completed = completed_lifecycle.completed
    last = completed["last_checkpoint"]
    history = learning.training.history.validate_completed_history(
        completed_lifecycle.run_dir,
        task="steady_flow",
        run_name=str(completed["config"]["run"]["name"]),
        checkpoint_identity=completed["best_checkpoint"]["identity"],
        completed_epoch=int(last["completed_epoch"]),
        validations=last["objective_history"],
    )
    (epoch_record,), (validation,) = history["epochs"], history["validations"]
    # The one epoch's terminal event validated point 20: its validation record is that event's evidence before the last publication.
    assert validation == {key: value for key, value in completed_lifecycle.epoch_metrics[-1].items() if key != "checkpoint/last_published"}
    assert validation["validation/point"] == 20.0
    assert "train/loss_total" in epoch_record
    assert "validation/point" not in epoch_record


@pytest.mark.parametrize("change", ["ahead", "behind", "missing"])
def test_steady_resume_reconciles_its_history_with_the_restored_last_checkpoint(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    change: str,
) -> None:
    """Resume truncates records ahead of the restored last checkpoint and refuses a history behind it or none at all."""
    run_dir = _interrupted_copy(completed_lifecycle, tmp_path / "history_resume_bundle")
    path = learning.training.history.history_path(run_dir)
    exact = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        path.unlink()
    else:
        ahead = {**exact["epochs"][-1], "epoch": 2.0, "global_step": exact["epochs"][-1]["global_step"] + 1.0}
        records = [*exact["epochs"], ahead] if change == "ahead" else []
        path.write_text(json.dumps({**exact, "epochs": records}), encoding="utf-8")
    config_path = common.paths.resolve_run_config_path(run_dir)

    if change == "ahead":
        experiments.run.run_experiment(config_path, resume=run_dir, device="cpu")
        assert json.loads(path.read_text(encoding="utf-8")) == exact
        return
    with pytest.raises(experiments.run.ContinuationRefusedError, match="history") as refused:
        experiments.run.run_experiment(config_path, resume=run_dir, device="cpu")
    assert isinstance(refused.value.__cause__, FileNotFoundError if change == "missing" else ValueError)


def test_a_resume_whose_last_checkpoint_does_not_restore_is_refused_before_its_first_write(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A last checkpoint whose loss state holds an unexpected key fails the full restore, which runs before the attempt writes.

    Its ``--resume`` and a start that names the run are refused and leave everything below the output root byte-identical:
    the run still interrupted, its history record ahead of the checkpoint kept (an admitted resume truncates it), no
    identity bound, and no job-log owner named, so no job log is linked into it. The same start from the run's own
    checkpoint continues the run in place, binds its identity and names the run as job-log owner.
    """
    output_root = tmp_path / "processed"
    run_dir = _interrupted_copy(completed_lifecycle, common.paths.resolve_config_run_dir(completed_lifecycle.config, output_root=output_root))
    request_path = _authored_request(run_dir, tmp_path)
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    original = last_path.read_bytes()
    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    payload["loss_state_dict"] = {**payload["loss_state_dict"], "unexpected": torch.zeros(1)}
    common.serialization.atomic_torch_save(payload, last_path)
    _refresh_summary_digest(run_dir, summary_key="last_checkpoint_sha256", artifact_path=last_path)
    path = learning.training.history.history_path(run_dir)
    exact = json.loads(path.read_text(encoding="utf-8"))
    ahead = {**exact["epochs"][-1], "epoch": 2.0, "global_step": exact["epochs"][-1]["global_step"] + 1.0}
    path.write_text(json.dumps({**exact, "epochs": [*exact["epochs"], ahead]}), encoding="utf-8")
    owners = tmp_path / "job_log_owners.txt"
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(owners))
    before = _artifact_inventory((output_root,))
    starts = (
        lambda: experiments.run.run_experiment(common.paths.resolve_run_config_path(run_dir), resume=run_dir, device="cpu"),
        lambda: experiments.run.run_experiment(request_path, device="cpu", output_root=output_root),
    )

    for start in starts:
        with pytest.raises(RuntimeError, match="Unexpected key"):
            start()
        assert _artifact_inventory((output_root,)) == before
    assert experiments.run.read_run_summary(run_dir)["status"] == "interrupted"
    assert not owners.exists()

    last_path.write_bytes(original)
    _refresh_summary_digest(run_dir, summary_key="last_checkpoint_sha256", artifact_path=last_path)
    starts[1]()
    assert owners.read_text(encoding="utf-8").splitlines() == [str(run_dir.resolve())]
    assert json.loads(path.read_text(encoding="utf-8")) == exact
    assert len(list((output_root / ".state" / "identities").glob("*.json"))) == 1


@pytest.mark.parametrize("payload", [b"", b"not a checkpoint"], ids=["empty", "garbage"])
def test_a_resume_whose_last_checkpoint_is_unreadable_is_refused_by_its_state(
    completed_lifecycle: CompletedLifecycle, tmp_path: Path, payload: bytes
) -> None:
    """An empty or partly copied last checkpoint is the run's persisted state: its resume is refused, names the file and changes nothing."""
    run_dir = _interrupted_copy(completed_lifecycle, tmp_path / "unreadable_resume_bundle")
    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    last_path.write_bytes(payload)
    before = _artifact_inventory((run_dir,))

    with pytest.raises(experiments.run.ContinuationRefusedError, match="Checkpoint is unreadable") as refused:
        experiments.run.run_experiment(common.paths.resolve_run_config_path(run_dir), resume=run_dir, device="cpu")
    assert str(last_path) in str(refused.value)
    unreadable = refused.value.__cause__
    assert unreadable is not None
    # torch.load raises these for an empty file and for bytes that are no checkpoint.
    assert isinstance(unreadable.__cause__, EOFError if not payload else pickle.UnpicklingError)
    assert _artifact_inventory((run_dir,)) == before


def test_a_resume_in_another_numerical_environment_is_offered_the_rerun_and_another_dataset_is_refused(
    completed_lifecycle: CompletedLifecycle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A resume in another numerical environment ends by the job's environment; another Dataset fingerprint is the run's state.

    A last checkpoint that differs from the run's identity only in its numerical environment is offered the rerun, which
    continues it in the producing environment; another Dataset fingerprint refuses it, in that environment as well.
    Neither changes the run or prints a traceback.
    """
    run_dir = _interrupted_copy(completed_lifecycle, tmp_path / "environment_resume_bundle")
    environment = learning.training.checkpoint.numerical_environment_identity()
    monkeypatch.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: {**environment, "numpy": "0.0.0"})
    before = _artifact_inventory((run_dir,))
    assert cli_train.main(["--resume", str(run_dir), "--device", "cpu"]) == 1
    reason, step = capsys.readouterr().err.splitlines()
    assert reason.startswith("Run not started: ResumeEnvironmentError: Exact resume requires the producing numerical environment")
    assert "differs in numpy" in reason
    assert step == "  next: rerun the same command in a matching allocation, the run is unchanged"
    assert _artifact_inventory((run_dir,)) == before

    last_path = common.paths.resolve_last_checkpoint_file(run_dir)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last["identity"]["dataset_fingerprints"]["train"] = "0" * 64
    common.serialization.atomic_torch_save(last, last_path)
    before = _artifact_inventory((run_dir,))
    assert cli_train.main(["--resume", str(run_dir), "--device", "cpu"]) == 1
    refusal = capsys.readouterr().err
    assert refusal.startswith("Run continuation refused: ")
    assert "Checkpoint run identity is incompatible" in refusal
    assert "Traceback" not in refusal
    assert _artifact_inventory((run_dir,)) == before


def _unmounted_dataset(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for a Dataset whose storage mount the job cannot reach."""
    message = "Dataset manifest is missing or unsafe: /unmounted/tiny_steady_id/manifest.json."
    raise FileNotFoundError(message)


def _unreadable_dataset(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for a Dataset manifest whose read fails with an I/O error, which its reader reports as unreadable."""
    message = "Dataset manifest is unreadable: /unmounted/tiny_steady_id/manifest.json."
    cause = OSError(errno.EIO, "Input/output error")
    raise ValueError(message) from cause


def _contradicted_split(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for the split resolution of a Dataset that changed since the run saved its split."""
    message = "Saved split does not match the current Dataset manifests, partition, evaluation membership or item expansion."
    raise ValueError(message)


@pytest.mark.parametrize("dataset", ["unmounted", "unreadable", "contradicted_split"])
def test_a_resume_whose_datasets_the_job_cannot_read_is_offered_the_rerun_and_a_contradicted_split_is_refused(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset: str,
) -> None:
    """
    Datasets the job cannot read end a resume by the job's environment; Datasets that contradict its saved split refuse it.

    A Dataset whose mount the job cannot reach, or whose manifest read fails with an I/O error, keeps its error and is
    offered the rerun in a matching allocation. One that contradicts the run's saved split is refused by the resume
    admission before the attempt: the command reports the refused continuation with that reason and no rerun, which would
    be refused again. Neither changes the run.
    """
    run_dir = _interrupted_copy(completed_lifecycle, tmp_path / "dataset_resume_bundle")
    before = _artifact_inventory((run_dir,))
    capsys.readouterr()

    if dataset == "contradicted_split":
        monkeypatch.setattr(datasets.preprocessing.splits, "resolve_split", _contradicted_split)
        assert cli_train.main(["--resume", str(run_dir), "--device", "cpu"]) == 1
        refusal = capsys.readouterr().err
        assert refusal.startswith("Run continuation refused: ")
        assert "Saved split does not match the current Dataset manifests" in refusal
        for advice in ("rerun", "Rerun", "Traceback"):
            assert advice not in refusal
    else:
        monkeypatch.setattr(datasets.runtime.factory, "create_dataset", _unmounted_dataset if dataset == "unmounted" else _unreadable_dataset)
        with pytest.raises(FileNotFoundError if dataset == "unmounted" else ValueError):
            experiments.run.run_experiment(common.paths.resolve_run_config_path(run_dir), resume=run_dir, device="cpu")
        assert "next: rerun the same command in a matching allocation, the run is unchanged" in capsys.readouterr().err
    assert _artifact_inventory((run_dir,)) == before


@pytest.mark.parametrize("environment", ["numerical_environment", "device"])
def test_a_completed_run_is_rerun_in_a_matching_allocation_and_its_resume_refused_whatever_the_environment(
    completed_lifecycle: CompletedLifecycle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], environment: str
) -> None:
    """
    A start that names a completed run ends by the job's environment, its ``--resume`` by the run's completed state.

    The start reads the run's checkpoints, which needs their numerical environment, and admits the job's device: either
    failing prints the rerun in a matching allocation without a traceback. Its ``--resume`` is refused before any check of
    that environment, because a completed run is never resumed. Neither writes anything, an identity binding included.
    """
    output_root = tmp_path / "processed"
    destination = common.paths.resolve_config_run_dir(completed_lifecycle.config, output_root=output_root)
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, destination)
    request_path = _authored_request(run_dir, tmp_path)

    def unusable(*_args: Any, **_kwargs: Any) -> None:
        message = "CUDA is not usable"
        raise learning.device.DeviceResolutionError(message)

    if environment == "numerical_environment":
        identity = learning.training.checkpoint.numerical_environment_identity()
        monkeypatch.setattr(learning.training.checkpoint, "numerical_environment_identity", lambda: {**identity, "numpy": "0.0.0"})
    else:
        monkeypatch.setattr(learning.device, "resolve_device", unusable)
    before = _artifact_inventory((tmp_path,))
    capsys.readouterr()

    with monkeypatch.context() as start:
        # The command has no output-root option; the start names the copy below this test's output root.
        start.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
        assert cli_train.main([str(request_path), "--device", "cpu"]) == 1
    reason, step = capsys.readouterr().err.splitlines()
    ended = "ResumeEnvironmentError: Reading requires" if environment == "numerical_environment" else "DeviceResolutionError: CUDA"
    assert reason.startswith(f"Run not started: {ended}")
    assert step == "  next: rerun the same command in a matching allocation, the run is unchanged"

    assert cli_train.main(["--resume", str(run_dir)]) == 1
    refusal = capsys.readouterr().err
    assert refusal.startswith("Run continuation refused: ")
    assert "never extended" in refusal
    assert "Traceback" not in refusal
    assert _artifact_inventory((tmp_path,)) == before


def test_a_start_that_names_an_interrupted_run_binds_its_identity_only_after_its_start_entry_is_admitted(
    completed_lifecycle: CompletedLifecycle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A continuation in place whose attempt's start entry fails, after the full restore admitted it, binds no identity.

    The start entry fails by the job's environment, its CUDA driver query; its Slurm provenance cannot fail it, because the
    command reads that before any check of the run (the allocation test). The binding is the last step before the attempt's
    first write, so everything below the output root stays byte-identical.
    """
    output_root = tmp_path / "processed"
    run_dir = _interrupted_copy(completed_lifecycle, common.paths.resolve_config_run_dir(completed_lifecycle.config, output_root=output_root))
    request_path = _authored_request(run_dir, tmp_path)

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        message = "CUDA driver query failed"
        raise RuntimeError(message)

    monkeypatch.setattr(learning.device, "cuda_driver_version", unavailable)
    before = _artifact_inventory((output_root,))

    with pytest.raises(RuntimeError, match="CUDA driver query failed"):
        experiments.run.run_experiment(request_path, device="cpu", output_root=output_root)
    assert _artifact_inventory((output_root,)) == before
    assert not list((output_root / ".state" / "identities").glob("*.json"))


@pytest.mark.parametrize("change", ["missing", "behind", "beyond_its_step"])
def test_a_completed_steady_run_is_admitted_only_with_its_valid_history(completed_lifecycle: CompletedLifecycle, tmp_path: Path, change: str) -> None:
    """A completed steady run, like a transient one, is admitted only with exactly the history its last checkpoint counts.

    It is refused when the history is missing, counts other records, or ends after the checkpoint's global step.
    """
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "run_history")
    selected = experiments.run.validate_evaluable_run(run_dir)["selected_checkpoint_sha256"]
    history = analysis.evaluation.run_discovery.stage_history(run_dir, expected_checkpoint_sha256=selected)
    assert isinstance(history, analysis.evaluation.run_discovery.StageHistory)
    assert (history.stage, history.start_epoch, history.selected_point) == ("steady", 0, 20)
    path = learning.training.history.history_path(run_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        path.unlink()
    elif change == "behind":
        path.write_text(json.dumps({**payload, "epochs": []}), encoding="utf-8")
    else:
        # The records still count the checkpoint's epoch and validation, but its epoch ends one optimizer step later.
        payload["epochs"][-1]["global_step"] += 1.0
        path.write_text(json.dumps(payload), encoding="utf-8")

    for admit in (experiments.run.validate_completed_run, experiments.run.validate_evaluable_run):
        with pytest.raises(experiments.run.RunLifecycleError, match="history"):
            admit(run_dir)


def _mutate_task_identity(run_dir: Path, dataset_root: Path) -> None:
    """Break only the saved split task digest."""
    del dataset_root
    path = common.paths.resolve_split_indices_path(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["task_contract_digest"] = "0" * 64
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="split_indices_sha256",
        artifact_path=path,
    )


def _mutate_config_identity(run_dir: Path, dataset_root: Path) -> None:
    """Break persisted config identity while retaining valid YAML."""
    del dataset_root
    path = common.paths.resolve_run_config_path(run_dir)
    config = experiments.config.loader.load_yaml(path)
    config["run"]["name"] = "different_run_identity"
    experiments.config.loader.save_yaml(config, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="config_sha256",
        artifact_path=path,
    )


def _mutate_dataset_identity(run_dir: Path, dataset_root: Path) -> None:
    """Change tensor content without changing its stored fingerprint."""
    del run_dir
    path = common.paths.resolve_dataset_path(
        "steady_flow",
        _ID_DATASET,
        dataset_root=dataset_root,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["inputs"][0, 0, 0, 0] += 0.5
    common.serialization.atomic_torch_save(payload, path)


def _mutate_split_identity(run_dir: Path, dataset_root: Path) -> None:
    """Change ordered membership without changing its digest."""
    del dataset_root
    path = common.paths.resolve_split_indices_path(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["roles"]["train"]["case_ids"].reverse()
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="split_indices_sha256",
        artifact_path=path,
    )


def _mutate_split_datasets(run_dir: Path, dataset_root: Path) -> None:
    """Drop the last material-OOD part, leaving an admissible split that no longer matches config.yaml."""
    del dataset_root
    path = common.paths.resolve_split_indices_path(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    aligned = (payload["dataset_identity"]["material_ood"], payload["roles"]["material_ood"]["parts"])
    for parts in (*aligned, payload["evaluation_assignments"]["material_ood"]):
        parts.pop()
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="split_indices_sha256",
        artifact_path=path,
    )


def _mutate_checkpoint_identity(run_dir: Path, dataset_root: Path) -> None:
    """Break only the best checkpoint run identity."""
    del dataset_root
    path = common.paths.resolve_best_checkpoint_file(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["identity"]["task"] = "different_task"
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="best_checkpoint_sha256",
        artifact_path=path,
    )


def _mutate_checkpoint_schema(run_dir: Path, dataset_root: Path) -> None:
    """Replace the valid best-checkpoint schema kind."""
    del dataset_root
    path = common.paths.resolve_best_checkpoint_file(run_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_kind"] = "other_payload"
    common.serialization.atomic_torch_save(payload, path)
    _refresh_summary_digest(
        run_dir,
        summary_key="best_checkpoint_sha256",
        artifact_path=path,
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_mutate_task_identity, "split_indices.pt task_contract_digest"),
        (_mutate_config_identity, "canonical generated leaf"),
        (_mutate_dataset_identity, "Dataset payload bytes disagree with manifest"),
        (_mutate_split_identity, "train membership digest mismatch"),
        (_mutate_split_datasets, "Saved split Datasets do not match the configured"),
        (_mutate_checkpoint_identity, "Checkpoint run identity"),
        (_mutate_checkpoint_schema, "Malformed or unsupported checkpoint schema_kind"),
    ],
    ids=(
        "task",
        "config",
        "dataset",
        "split",
        "split-datasets",
        "checkpoint",
        "checkpoint-schema",
    ),
)
def test_saved_run_mismatches_are_rejected_before_forward(
    tmp_path: Path,
    completed_lifecycle: CompletedLifecycle,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Path, Path], None],
    match: str,
) -> None:
    """
    Corrupt one saved task/config/dataset/split/split-Dataset/checkpoint-schema layer.

    Every parametrized mutation keeps unrelated outer evidence valid where needed
    but must fail its owning admission check before FNO forward, proving the complete
    saved-run identity chain is fail-closed.
    """
    run_dir = tmp_path / "run"
    dataset_root = tmp_path / "02_datasets"
    shutil.copytree(completed_lifecycle.run_dir, run_dir)
    shutil.copytree(completed_lifecycle.dataset_root, dataset_root)
    mutate(run_dir, dataset_root)

    def fail_forward(self: FNO, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        del self
        message = "Saved-run rejection reached model forward."
        raise AssertionError(message)

    monkeypatch.setattr(FNO, "forward", fail_forward)
    if mutate is not _mutate_dataset_identity:
        with pytest.raises((ValueError, RuntimeError)):
            analysis.evaluation.artifact_loader.load_run_artifacts(run_dir)
    with pytest.raises((ValueError, RuntimeError), match=match):
        learning.inference.context.load_inference_context(
            run_dir=run_dir,
            dataset_root=dataset_root,
            split="id_test",
            device_policy="cpu",
        )


@pytest.mark.parametrize("role", ["best", "last"])
def test_changed_checkpoint_is_rejected_by_digest_before_it_is_deserialized(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    role: str,
) -> None:
    """Refuse a checkpoint whose bytes differ from the summary before pickle loading could run its content."""
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "run")
    path = common.paths.resolve_best_checkpoint_file(run_dir) if role == "best" else common.paths.resolve_last_checkpoint_file(run_dir)
    path.write_bytes(b"not a pickle")  # a load before the digest check would raise an unpickling error instead
    with pytest.raises(experiments.run.RunLifecycleError, match=f"{role} checkpoint digest mismatch"):
        experiments.run.validate_completed_run(run_dir)
    with pytest.raises(experiments.run.RunLifecycleError, match=f"{role} checkpoint digest mismatch"):
        analysis.evaluation.artifact_loader.load_run_artifacts(run_dir)


@pytest.mark.parametrize("filename", common.paths.EVALUABLE_RUN_REQUIRED_FILES)
def test_saved_artifact_reader_rejects_changed_bundle_bytes(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
    filename: str,
) -> None:
    """Reject changed saved config, split, normalizer, checkpoint, or summary bytes."""
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "run")
    target = run_dir / filename
    target.write_bytes(target.read_bytes() + (b"\n" if target.suffix in {".yaml", ".json"} else b"changed"))
    if filename == common.paths.RUN_SUMMARY_FILENAME:
        # Valid JSON whitespace is harmless; the declared lifecycle still must agree.
        summary = experiments.run.read_run_summary(run_dir)
        summary["status_history"][-1]["status"] = "running"
        common.serialization.atomic_write_json(target, summary)
    with pytest.raises((ValueError, RuntimeError)):
        analysis.evaluation.artifact_loader.load_run_artifacts(run_dir)


def test_saved_artifact_reader_rejects_normalizer_and_checkpoint_path_tampering(
    completed_lifecycle: CompletedLifecycle,
    tmp_path: Path,
) -> None:
    """Retain fitted-normalizer identity and regular bundle-local evidence paths."""
    run_dir = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "run")
    normalizer_path = common.paths.resolve_normalizer_path(run_dir)
    normalizer = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    normalizer["state"]["train_membership_digest"] = "0" * 64
    common.serialization.atomic_torch_save(normalizer, normalizer_path)
    _refresh_summary_digest(run_dir, summary_key="normalizer_sha256", artifact_path=normalizer_path)
    with pytest.raises((ValueError, RuntimeError)):
        analysis.evaluation.artifact_loader.load_run_artifacts(run_dir)
    linked = _copy_without_analysis(completed_lifecycle.run_dir, tmp_path / "linked")
    checkpoint = common.paths.resolve_best_checkpoint_file(linked)
    checkpoint.unlink()
    checkpoint.symlink_to(common.paths.resolve_best_checkpoint_file(completed_lifecycle.run_dir))
    with pytest.raises(experiments.run.RunLifecycleError, match="regular bundle-local files"):
        analysis.evaluation.artifact_loader.load_run_artifacts(linked)
