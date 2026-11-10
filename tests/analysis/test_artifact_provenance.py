"""Verify steady-flow artifact generation against current provenance.

Responsibilities:
  - Check field and unit propagation, normalized statistics, dimensionless physics, and
    permeability reconstruction

Design principles:
  - Task-owned semantics remain consistent across batch partitions

This module does NOT:
  - Test cache locking races or visualization behavior
"""

from __future__ import annotations

import itertools
import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import torch
from support.synthetic_task import synthetic_split_payload
from torch import nn
from torch.utils.data import DataLoader

from src import analysis, common, datasets, domain, learning
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract

if TYPE_CHECKING:
    from pathlib import Path

_SYNTHETIC_INPUT_VALUE = 4.0
_SYNTHETIC_METADATA_VALUE = 3.25
# Campaign-local case number 3 and global Dataset position 5 differ on purpose.
_CASE_ID = "lentil__case_0003__fe83e052fbf2"
_OTHER_CASE_ID = "field_pea__case_0001__0123456789ab"
_THIRD_CASE_ID = "chickpea__case_0002__abcdef012345"
_STEADY_SHAPE = (9, 11)
_STEADY_INLET_PRESSURE = 2.0
_NONFINITE_INDEX = (3, 4)


class _IdentityNormalizer:
    """Leave tensors unchanged while exposing unit fitted scales."""

    def __init__(self, channels: int) -> None:
        """Store unit channel scales in the saved-normalizer layout."""
        self.std = torch.ones(1, channels, 1, 1)

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        """Return normalized-space input unchanged."""
        return value

    def inverse_transform(self, value: torch.Tensor) -> torch.Tensor:
        """Return physical-space output unchanged."""
        return value


class _IdentityProcessor:
    """Model the fitted processor lifecycle used by online evaluation."""

    def __init__(self, task: domain.tasks.spec.TaskSpec) -> None:
        """Create task-shaped identity normalizers and call traces."""
        self.in_normalizer = _IdentityNormalizer(task.in_channels)
        self.out_normalizer = _IdentityNormalizer(task.out_channels)
        self.training = True
        self.preprocessed_batch_sizes: list[int] = []

    def eval(self) -> None:
        """Switch preprocessing to evaluation semantics."""
        self.training = False

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize complete-batch inputs while preserving physical targets."""
        if self.training:
            msg = "artifact preprocessing did not enter evaluation mode"
            raise AssertionError(msg)
        inputs = batch["x"]
        if not isinstance(inputs, torch.Tensor):
            msg = "fixture inputs must be tensors"
            raise TypeError(msg)
        self.preprocessed_batch_sizes.append(int(inputs.shape[0]))
        return {**batch, "x": self.in_normalizer.transform(inputs)}


class _SteadyProjection(nn.Module):
    """Return a deterministic task-ordered manufactured steady-flow state."""

    def __init__(self, task: domain.tasks.spec.TaskSpec) -> None:
        super().__init__()
        self.task = task

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Construct pressure and velocity values by TaskSpec field names."""
        x_values = value[:, self.task.input_names.index("x")]
        y_values = value[:, self.task.input_names.index("y")]
        zeros = torch.zeros_like(x_values)
        pressure = next(group for group in self.task.output_groups if group.id == "pressure")
        velocity = next(group for group in self.task.output_groups if group.id == "velocity")
        output_by_field = dict.fromkeys(self.task.output_names, zeros)
        output_by_field[pressure.fields[0]] = x_values
        output_by_field[velocity.fields[0]] = x_values
        # A height-dependent v makes the cross-section flux balance observable.
        output_by_field[velocity.fields[1]] = 1e-3 * y_values
        return torch.stack([output_by_field[field] for field in self.task.output_names], dim=1)


class _NonfiniteSteadyProjection(_SteadyProjection):
    """Return the manufactured state with one NaN ``u`` value in the last case of each batch."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Replace exactly one velocity value by NaN without touching other cases."""
        output = super().forward(value)
        output[-1, self.task.output_names.index("u"), *_NONFINITE_INDEX] = float("nan")
        return output


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """
    Save one strict payload at its canonical logical-dataset path.

    Returning the concrete ``.pt`` path lets the contract test exercise both
    environment-based discovery and direct task-dataset loading.
    """
    dataset_id = payload["dataset_id"]
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.pt"
    torch.save(payload, path)
    return path


def _attach_objective_contract(
    provenance: dict[str, Any],
    *,
    task: domain.tasks.spec.TaskSpec,
) -> None:
    """Attach task-owned groups and unit saved-normalizer evidence in place."""
    selection = provenance.get("selection")
    if not isinstance(selection, dict):
        msg = "fixture selection must be a dictionary"
        raise TypeError(msg)
    count = selection.get("effective_case_count")
    digest = selection.get("effective_ordered_source_indices_sha256")
    selection.update(
        {
            "full_selected_case_count": count,
            "generation_limit": None,
            "full_ordered_source_indices_sha256": digest,
        }
    )
    evaluator = provenance.setdefault("evaluator", {})
    if not isinstance(evaluator, dict):
        msg = "fixture evaluator must be a dictionary"
        raise TypeError(msg)
    evaluator.update(
        {
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
            "objective": next(metric for metric in task.default_metrics if metric.kind == "group_macro_rmse").as_dict(all_fields=task.output_names),
            "output_groups": analysis.artifacts.contracts.output_group_payload(task.output_groups),
        }
    )
    provenance["normalizer"] = {
        "denominator_floor": 0.0,
        "output_standard_deviations": dict.fromkeys(task.output_names, 1.0),
    }


def _steady_artifact(
    tmp_path: Path,
    model: nn.Module,
    *,
    case_ids: tuple[str, ...] = (_CASE_ID,),
    target: torch.Tensor | None = None,
    call_case_ids: tuple[str, ...] | None = None,
    source_meta: dict[str, Any] | None = None,
    partition: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Generate one steady artifact of manufactured cases, batched by ``partition``, and return frame, root and request provenance."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    y_values = torch.linspace(0.0, 1.0, _STEADY_SHAPE[0])
    x_values = torch.linspace(0.0, 2.0, _STEADY_SHAPE[1])
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    zeros = torch.zeros_like(x_grid)
    coordinate_values = iter((x_grid, y_grid))
    input_by_field: dict[str, torch.Tensor] = {}
    for field in task.inputs:
        if field.role == "coordinate":
            input_by_field[field.name] = next(coordinate_values)
        elif field.role == "permeability" and "cross_component" in field.representation:
            input_by_field[field.name] = zeros
        elif field.role == "permeability":
            input_by_field[field.name] = torch.full_like(x_grid, -4.0)
        elif field.role == "porosity":
            input_by_field[field.name] = 0.25 + 0.25 * x_grid
        elif field.role == "boundary":
            input_by_field[field.name] = torch.where(y_grid == y_grid.amin(), _STEADY_INLET_PRESSURE, 0.0)
        else:
            msg = f"unsupported steady-flow fixture role: {field.role}"
            raise AssertionError(msg)
    count = len(case_ids)
    inputs = torch.stack([input_by_field[field] for field in task.input_names], dim=0).unsqueeze(0).repeat(count, 1, 1, 1)
    targets = torch.zeros((count, task.out_channels, *_STEADY_SHAPE)) if target is None else target.expand(count, -1, -1, -1)
    sample_ids = [f"manufactured-{index}" for index in range(count)]
    bounds = np.cumsum((0, *(partition or (count,)))).tolist()
    loader = [
        {
            "x": inputs[start:stop],
            "y": targets[start:stop],
            "source_index": torch.arange(start, stop),
            "split_local_index": torch.arange(start, stop),
            "meta": source_meta or {"sample_id": sample_ids[start:stop]},
        }
        for start, stop in itertools.pairwise(bounds)
    ]
    root = tmp_path / "steady"
    provenance = {
        "run": {"name": "steady", "task": task.id},
        "model": {"parameter_counts": {"total": 1, "trainable": 1}},
        "split_role": "eval",
        "dataset": _role_plan(case_ids).dataset_provenance(),
        "selection": {
            "effective_case_count": count,
            "effective_ordered_source_indices_sha256": analysis.artifacts.contracts.ordered_indices_sha256(range(count)),
        },
        "evaluator": {
            "input_fields": list(task.input_names),
            "input_units": {field.name: field.unit for field in task.inputs},
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
            "physics_kind": task.physics.kind,
        },
        "physics": analysis.artifacts.contracts.steady_physics_provenance(task),
    }
    _attach_objective_contract(provenance, task=task)
    frame, _parquet_path = analysis.artifacts.generation.generate_artifacts(
        task=task,
        model=model,
        loader=loader,
        processor=_IdentityProcessor(task),
        device=torch.device("cpu"),
        save_root=root,
        provenance=provenance,
        case_ids=list(case_ids if call_case_ids is None else call_case_ids),
    )
    return frame, root, provenance


def _role_plan(case_ids: tuple[str, ...]) -> Any:
    """Return a one-source ID role over ``case_ids`` in their given order."""
    return analysis.artifacts.plan.ArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="steady",
        source_dataset_ids=("steady",),
        source_identities=({"dataset_id": "steady", "case_ids": list(case_ids)},),
        case_ids_by_source=(tuple(case_ids),),
        membership_digests=("c" * 64,),
    )


def _staged_request(provenance: dict[str, Any], case_ids: tuple[str, ...]) -> Any:
    """Return the writer-side request that binds staged rows to their NPZ payloads."""
    return analysis.artifacts.service.ArtifactRequest(
        provenance=provenance,
        source_indices=tuple(range(len(case_ids))),
        batch_size=len(case_ids),
        case_ids=case_ids,
    )


def test_steady_artifact_stores_dimensionless_physics_semantics(tmp_path: Path) -> None:
    """
    Keep momentum, flux-balance, and pressure-boundary artifacts explicit.

    Every scalar must be the declared reduction of its stored residual array or
    boundary values, scaled by the prescribed inlet pressure.
    """
    task = domain.tasks.steady_flow.STEADY_FLOW
    groups_by_id = {group.id: group for group in task.output_groups}
    (pressure_field,) = groups_by_id["pressure"].fields
    velocity_fields = groups_by_id["velocity"].fields
    pressure_index = task.output_names.index(pressure_field)
    velocity_indices = [task.output_names.index(field) for field in velocity_fields]
    velocity_units = {task.field(field).unit for field in velocity_fields}
    assert len(velocity_units) == 1
    velocity_unit = next(iter(velocity_units))
    height, width = _STEADY_SHAPE
    inlet_pressure = _STEADY_INLET_PRESSURE
    frame, root, _provenance = _steady_artifact(tmp_path, _SteadyProjection(task))
    row = frame.iloc[0]
    required_scalars = {
        "momentum_residual_mse",
        "flux_balance_mse",
        "pressure_boundary_mse",
        "pressure_inlet_mse",
        "pressure_outlet_mean_square",
    }
    assert required_scalars.issubset(frame.columns)
    payload_path = analysis.artifacts.contracts.resolve_case_payload_path(
        root,
        row["npz_path"],
        expected_filename=f"{_CASE_ID}.npz",
    )
    with np.load(payload_path, allow_pickle=False) as payload:
        assert {"Rx", "Ry", "flux_balance", "coordinates"}.issubset(payload.files)
        for name, shape in (("Rx", (height, width)), ("Ry", (height, width)), ("flux_balance", (height,))):
            assert payload[name].shape == shape
            assert np.issubdtype(payload[name].dtype, np.floating)
            assert np.isfinite(payload[name]).all()
        crop = analysis.artifacts.contracts.EVAL_PAD
        interior = np.s_[crop:-crop, crop:-crop]
        expected_momentum = float(np.mean(payload["Rx"][interior] ** 2 + payload["Ry"][interior] ** 2))
        assert row["momentum_residual_mse"] == pytest.approx(expected_momentum)
        assert row["flux_balance_mse"] == pytest.approx(float(np.mean(payload["flux_balance"] ** 2)))
        assert row["flux_balance_mse"] > 0.0
        expected_speed = np.sqrt(np.square(payload["pred"][velocity_indices]).sum(axis=0))
        np.testing.assert_allclose(payload["pred"][-1], expected_speed)
        assert payload["artifact_units"][-1] == velocity_unit
        pressure = payload["pred"][pressure_index]
        pressure_boundary = payload["p_in_bc"][0]
        expected_inlet = float(np.mean((pressure[0] - pressure_boundary[0]) ** 2)) / inlet_pressure**2
        expected_outlet_mean_square = float(np.mean(pressure[-1]) / inlet_pressure) ** 2
        outlet_pointwise_mse = float(np.mean(pressure[-1] ** 2)) / inlet_pressure**2
    assert row["pressure_inlet_mse"] == pytest.approx(expected_inlet)
    assert row["pressure_outlet_mean_square"] == pytest.approx(expected_outlet_mean_square)
    assert row["pressure_outlet_mean_square"] != pytest.approx(outlet_pointwise_mse)
    assert row["pressure_boundary_mse"] == pytest.approx(row["pressure_inlet_mse"] + row["pressure_outlet_mean_square"])


def test_steady_artifact_rel_h1_is_the_training_h1_on_the_cropped_interior(tmp_path: Path) -> None:
    """
    Bind the artifact ``rel_h1`` to the shared NeuralOp H1 implementation.

    Each field is cropped to the EVAL_PAD interior first; the metric is the mean over
    fields of the loss-factory relative H1 with physical spacing (dy, dx) = (1/8, 1/5) m.
    """
    task = domain.tasks.steady_flow.STEADY_FLOW
    y_values = torch.linspace(0.0, 1.0, _STEADY_SHAPE[0])
    x_values = torch.linspace(0.0, 2.0, _STEADY_SHAPE[1])
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    target = torch.stack([1.0 + 0.3 * (index + 1) * torch.sin(2.0 * x_grid + y_grid) + y_grid**2 for index in range(task.out_channels)]).unsqueeze(0)
    model = _SteadyProjection(task)
    frame, _root, _provenance = _steady_artifact(tmp_path, model, target=target)

    inputs = {"x": x_grid[None, None], "y": y_grid[None, None]}
    prediction = model(torch.cat([inputs[name] if name in inputs else torch.zeros_like(x_grid)[None, None] for name in task.input_names], dim=1))
    crop = analysis.artifacts.contracts.EVAL_PAD
    relative_h1 = learning.losses.factory.build_relative_h1(2, reduction="mean", grid_spacing=(0.125, 0.2))
    expected = [
        float(relative_h1(prediction[:, c : c + 1, crop:-crop, crop:-crop], target[:, c : c + 1, crop:-crop, crop:-crop]))
        for c in range(task.out_channels)
    ]
    assert frame["rel_h1"].iloc[0] == pytest.approx(sum(expected) / len(expected), rel=1e-5)


@pytest.mark.parametrize("partition", [(2, 1), (1, 1, 1)], ids=("unequal", "single-case"))
def test_steady_artifact_rows_and_objective_do_not_depend_on_the_batch_partition(tmp_path: Path, partition: tuple[int, ...]) -> None:
    """
    Keep per-case rows and the group objective independent of how distinct cases are batched.

    Unequal batches must neither weight batches equally nor assign one case's metrics to another.
    """
    task = domain.tasks.steady_flow.STEADY_FLOW
    case_ids = (_CASE_ID, _OTHER_CASE_ID, _THIRD_CASE_ID)
    y_values = torch.linspace(0.0, 1.0, _STEADY_SHAPE[0])
    x_values = torch.linspace(0.0, 2.0, _STEADY_SHAPE[1])
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    channels = torch.arange(1, task.out_channels + 1, dtype=torch.float32)[:, None, None]
    target = torch.stack([1.0 + 0.4 * (case + 1) * torch.sin(channels * x_grid + y_grid) for case in range(len(case_ids))])
    model = _SteadyProjection(task)
    whole, _root, _provenance = _steady_artifact(tmp_path / "whole", model, case_ids=case_ids, target=target)
    split, _root, _provenance = _steady_artifact(tmp_path / "split", model, case_ids=case_ids, target=target, partition=partition)

    metric_columns = [column for column in whole.columns if whole[column].dtype.kind == "f"]
    assert whole["rel_l2"].nunique() == len(case_ids)
    pd.testing.assert_frame_equal(split[metric_columns], whole[metric_columns], rtol=1e-6)
    whole_value = whole.attrs["artifact_provenance"]["aggregate"]["value"]
    assert split.attrs["artifact_provenance"]["aggregate"]["value"] == pytest.approx(whole_value, rel=1e-9)


def test_non_steady_task_is_rejected_by_artifact_generation(tmp_path: Path, synthetic_task: domain.tasks.spec.TaskSpec) -> None:
    """Refuse every task other than steady flow before any payload is written."""
    with pytest.raises(ValueError, match="supports only task"):
        analysis.artifacts.generation.generate_artifacts(
            task=synthetic_task,
            model=nn.Identity(),
            loader=[],
            processor=_IdentityProcessor(synthetic_task),
            device=torch.device("cpu"),
            save_root=tmp_path / "target",
            provenance={},
            case_ids=[],
        )
    assert not (tmp_path / "target").exists()


def test_steady_artifact_rejects_case_ids_not_matching_membership(tmp_path: Path) -> None:
    """Reject case IDs that do not cover the saved membership one to one before any payload is written."""
    with pytest.raises(ValueError, match="exactly match"):
        _steady_artifact(tmp_path, _SteadyProjection(domain.tasks.steady_flow.STEADY_FLOW), call_case_ids=(_CASE_ID, _OTHER_CASE_ID))
    assert not (tmp_path / "steady").exists()


def test_steady_artifact_rejects_source_metadata_duplicating_artifact_identity(tmp_path: Path) -> None:
    """Reject a forged ``case_index`` in source metadata so it cannot replace the membership-derived identity."""
    with pytest.raises(KeyError, match="reserved artifact identity"):
        _steady_artifact(tmp_path, _SteadyProjection(domain.tasks.steady_flow.STEADY_FLOW), source_meta={"case_index": 99})


def test_steady_artifact_publishes_a_nonfinite_case_with_counts_instead_of_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classify a non-finite steady prediction, store it unchanged and report counts instead of metric values."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    case_ids = (_CASE_ID, _OTHER_CASE_ID)
    frame, root, provenance = _steady_artifact(tmp_path, _NonfiniteSteadyProjection(task), case_ids=case_ids)

    documents = [json.loads(text) for text in frame["prediction_validity"]]
    assert [document["status"] for document in documents] == [validity_contract.VALID, validity_contract.NONFINITE]
    assert documents[1]["channels"]["u"]["nan_count"] == 1
    assert documents[1]["first_invalid"] == {"kind": "NONFINITE", "channel": "u", "channel_index": 1, "spatial_index": list(_NONFINITE_INDEX)}
    record_metrics = ["rel_l2", "momentum_residual_mse", "physical_sse_p", "normalized_rmse_v", "normalized_velocity_vector_rmse"]
    assert np.isfinite(frame.loc[0, record_metrics].to_numpy(dtype=float)).all()
    assert np.isnan(frame.loc[1, record_metrics].to_numpy(dtype=float)).all()
    assert frame.loc[1, "physical_count_u"] == _STEADY_SHAPE[0] * _STEADY_SHAPE[1]
    with np.load(root / "npz" / f"{_OTHER_CASE_ID}.npz", allow_pickle=False) as payload:
        assert np.isnan(payload["pred"][1, *_NONFINITE_INDEX])

    stored = json.loads(analysis.artifacts.contracts.artifact_provenance_path(root).read_text(encoding="utf-8"))
    aggregate = stored["aggregate"]
    required = len(case_ids) * len(task.output_names) * _STEADY_SHAPE[0] * _STEADY_SHAPE[1]
    assert (aggregate["available"], aggregate["value"], aggregate["field_statistics"], aggregate["group_statistics"]) == (False, None, None, None)
    assert [aggregate[name] for name in ("required_value_count", "computed_value_count", "finite_value_count", "nonfinite_value_count")] == [
        required,
        required,
        required - 1,
        1,
    ]
    evidence = stored["evaluation"]["prediction_validity"]
    assert evidence["status_counts"] == {"VALID": 1, "FINITE_BUT_PHYSICALLY_INVALID": 0, "NONFINITE": 1}

    analysis.artifacts.service._validate_staged_case_payloads(root, request=_staged_request(provenance, case_ids))
    loaded = analysis.evaluation.artifact_loader._read_artifact_root(root)
    assert loaded.attrs[analysis.evaluation.dataframe.PRIMARY_OBJECTIVE_ID] == aggregate

    capsys.readouterr()
    validity_contract.report_invalid_predictions(evidence)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[WARNING] artifact completed with invalid predictions | valid=1 | finite_but_physically_invalid=0 | nonfinite=1"
    assert f"case={_OTHER_CASE_ID} | status=NONFINITE | channels=u | first_channel=u | first_index=[3, 4]" in lines[1]
    assert validity_contract.invalid_prediction_note([stored]) == "valid 1 of 2"


def test_steady_payload_contradicting_its_validity_document_is_refused_on_write_and_read(tmp_path: Path) -> None:
    """Corruption stays an error: changed arrays fail the writer check and the digest, relabelled documents fail the reader."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    case_ids = (_CASE_ID, _OTHER_CASE_ID)
    _frame, root, provenance = _steady_artifact(tmp_path, _NonfiniteSteadyProjection(task), case_ids=case_ids)
    for case_id, position, value, message in (
        (_OTHER_CASE_ID, (1, *_NONFINITE_INDEX), 0.0, "contradicts its prediction validity document"),
        (_CASE_ID, (0, 0, 0), np.inf, "pred contains non-finite values"),
    ):
        path = root / "npz" / f"{case_id}.npz"
        original = path.read_bytes()
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        arrays["pred"][position] = value
        np.savez_compressed(path, **arrays)
        with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match=message):
            analysis.artifacts.service._validate_staged_case_payloads(root, request=_staged_request(provenance, case_ids))
        # A read admits the structure only; the changed payload is refused by its digest when its case opens.
        frame = analysis.evaluation.artifact_loader._read_artifact_root(root)
        with pytest.raises(ValueError, match="Steady case payload digest changed"):
            analysis.evaluation.case.load_case(frame, position[0])
        path.write_bytes(original)

    provenance_path = analysis.artifacts.contracts.artifact_provenance_path(root)
    stored = json.loads(provenance_path.read_text(encoding="utf-8"))
    parquet_path = root / "artifacts.parquet"
    table = pd.read_parquet(parquet_path)
    table.loc[1, "prediction_validity"] = table.loc[0, "prediction_validity"]
    table.to_parquet(parquet_path, index=False)
    stored["outputs"] = analysis.artifacts.contracts.artifact_output_manifest(root)
    common.serialization.atomic_write_json(provenance_path, stored)
    with pytest.raises(ValueError, match="valid predictions"):
        analysis.evaluation.artifact_loader._read_artifact_root(root)


def test_steady_load_checks_structure_only_and_each_payload_digest_when_its_case_opens(tmp_path: Path) -> None:
    """A never-opened changed payload does not block the load; structure and Parquet changes do."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    _frame, root, _provenance = _steady_artifact(tmp_path, _SteadyProjection(task), case_ids=(_CASE_ID, _OTHER_CASE_ID))
    loader = analysis.evaluation.artifact_loader
    untouched = root / "npz" / f"{_OTHER_CASE_ID}.npz"
    original = untouched.read_bytes()
    untouched.write_bytes(original + b"changed")
    frame = loader._read_artifact_root(root)
    analysis.evaluation.case.load_case(frame, 0)
    with pytest.raises(ValueError, match="Steady case payload digest changed"):
        analysis.evaluation.case.load_case(frame, 1)
    untouched.write_bytes(original)

    extra = root / "npz" / "extra.npz"
    extra.write_bytes(original)
    with pytest.raises(ValueError, match="output manifest does not match"):
        loader._read_artifact_root(root)
    extra.unlink()

    moved = untouched.with_name("renamed.npz")
    untouched.rename(moved)
    with pytest.raises(ValueError, match="output manifest does not match"):
        loader._read_artifact_root(root)
    moved.rename(untouched)

    moved = root / "renamed.npz"  # outside npz/, so only the symlink itself can trigger the rejection
    untouched.rename(moved)
    untouched.symlink_to(moved)
    with pytest.raises(ValueError, match="output manifest does not match"):
        loader._read_artifact_root(root)
    untouched.unlink()
    moved.rename(untouched)

    parquet = root / "artifacts.parquet"
    parquet_bytes = parquet.read_bytes()
    parquet.write_bytes(parquet_bytes + b"changed")
    with pytest.raises(ValueError, match="output manifest does not match"):
        loader._read_artifact_root(root)
    parquet.write_bytes(parquet_bytes)
    loader._read_artifact_root(root)


def test_steady_case_replaced_after_its_digest_check_is_not_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The digest is taken on the handle the case is parsed from, so a path swapped before the parse changes nothing."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    _frame, root, _provenance = _steady_artifact(tmp_path, _SteadyProjection(task), case_ids=(_CASE_ID, _OTHER_CASE_ID))
    frame = analysis.evaluation.artifact_loader._read_artifact_root(root)
    baseline = analysis.evaluation.case.load_case(frame, 0)
    path = root / "npz" / f"{_CASE_ID}.npz"
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["source_index"] = arrays["source_index"] + 100
    sibling = root / "replacement.tmp"
    with sibling.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    original_load = np.load

    def replacing_load(*args: Any, **kwargs: Any) -> Any:
        sibling.replace(path)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", replacing_load)
    assert analysis.evaluation.case.load_case(frame, 0).source_index == baseline.source_index


def test_steady_case_changed_in_place_while_it_is_read_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-place change of the verified file during the parse is refused instead of returned as verified data."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    _frame, root, _provenance = _steady_artifact(tmp_path, _SteadyProjection(task), case_ids=(_CASE_ID,))
    frame = analysis.evaluation.artifact_loader._read_artifact_root(root)
    path = root / "npz" / f"{_CASE_ID}.npz"
    original_load = np.load

    def appending_load(*args: Any, **kwargs: Any) -> Any:
        loaded = original_load(*args, **kwargs)
        with path.open("ab") as appended:
            appended.write(b"\0")
        return loaded

    monkeypatch.setattr(np, "load", appending_load)
    with pytest.raises(ValueError, match="changed while it was read"):
        analysis.evaluation.case.load_case(frame, 0)


def test_cached_digest_is_not_reused_for_the_same_inode_on_another_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that differs from the cached one only in its device is hashed again, not served from the cache."""
    path = tmp_path / "payload.bin"
    path.write_bytes(b"current")
    common.serialization.cached_file_sha256(path)
    hashed: list[str] = []
    original_sha256 = common.serialization._stream_sha256
    real_fstat = os.fstat

    def counted_sha256(stream: Any) -> str:
        hashed.append(stream.name)
        return original_sha256(stream)

    def other_device(descriptor: int) -> Any:
        status = real_fstat(descriptor)
        return SimpleNamespace(st_dev=status.st_dev + 1, st_ino=status.st_ino, st_size=status.st_size, st_mtime_ns=status.st_mtime_ns)

    monkeypatch.setattr(common.serialization, "_stream_sha256", counted_sha256)
    monkeypatch.setattr(common.serialization.os, "fstat", other_device)
    common.serialization.cached_file_sha256(path)
    assert hashed == [str(path)]


def test_cached_digest_and_its_identity_come_from_one_open_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A path swapped between taking the identity and hashing never caches the new bytes' digest under the old identity."""
    path = tmp_path / "payload.bin"
    path.write_bytes(b"first")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"second")
    original_sha256 = common.serialization.file_sha256

    def swapping_sha256(target: Path | str) -> str:
        # A second open by path is where the swap could slip in between identity and digest.
        replacement.replace(path)
        return original_sha256(target)

    monkeypatch.setattr(common.serialization, "file_sha256", swapping_sha256)
    first_digest = original_sha256(path)
    assert common.serialization.cached_file_sha256(path) == first_digest
    assert common.serialization._FILE_DIGESTS[str(path)][1] == first_digest


def test_physical_cross_permeability_is_reconstructed_from_its_ratio() -> None:
    """
    Reconstruct physical cross-permeability from its encoded ratio.

    The original tensor must remain unchanged while ``kxy`` is scaled by the
    diagonal permeability convention into square metres for artifact consumers.
    """
    encoded = torch.tensor([[[[-2.0]], [[0.25]], [[-4.0]]]])
    permeability = analysis.artifacts.generation.extract_kappa(
        encoded,
        input_fields=["Kxx", "Kxy", "Kyy"],
        kappa_names=["Kxx", "Kxy", "Kyy"],
    )

    assert torch.equal(permeability["kappa_encoded"], encoded)
    assert permeability["kappa"][0, :, 0, 0].tolist() == pytest.approx([1e-2, 2.5e-4, 1e-4])


@pytest.mark.parametrize(
    "case_id",
    [
        "case_0003",
        "Lentil__case_0003__fe83e052fbf2",
        "lentil__case_3__fe83e052fbf2",
        "lentil__case_0003__fe83e052fbf",
        "lentil__case_0003__fe83e052fbf2.npz",
        "../lentil__case_0003__fe83e052fbf2",
    ],
)
def test_case_payload_filename_rejects_non_package_case_ids(case_id: str) -> None:
    """Name NPZ files only after the strict package case ID pattern accepted the ID."""
    assert analysis.artifacts.contracts.case_payload_filename(_CASE_ID) == f"{_CASE_ID}.npz"
    with pytest.raises(ValueError, match="package case ID"):
        analysis.artifacts.contracts.case_payload_filename(case_id)


def test_dataset_case_payload_filenames_drop_dataset_qualifier_and_reject_repeats() -> None:
    """Combined evaluation Datasets keep the package case ID as file name and cannot repeat one."""
    other = "field_pea__case_0001__0123456789ab"
    names = analysis.artifacts.contracts.dataset_case_payload_filenames([f"ood_a::{_CASE_ID}", other])
    assert names == (f"{_CASE_ID}.npz", f"{other}.npz")
    with pytest.raises(ValueError, match="duplicate case payload names"):
        analysis.artifacts.contracts.dataset_case_payload_filenames([f"ood_a::{_CASE_ID}", f"ood_b::{_CASE_ID}"])


def test_synthetic_task_flows_through_final_dataset_contract(
    tmp_path: Path,
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Final dataset loading and metadata flattening remain task-generic."""
    inputs = torch.stack([torch.full((2, 3), _SYNTHETIC_INPUT_VALUE + index) for index in range(synthetic_task.in_channels)]).unsqueeze(0)
    outputs = torch.stack([torch.full((2, 3), 20.0 + index) for index in range(synthetic_task.out_channels)]).unsqueeze(0)
    source_identity = {"token": "synthetic"}
    expected_metadata = {
        "generator": {"parameters": {"scalar_parameter": _SYNTHETIC_METADATA_VALUE}},
    }
    fingerprint = datasets.contracts.identity.compute_case_fingerprint(
        task=synthetic_task,
        case_id="case_0000",
        source_identity=source_identity,
        source_metadata=expected_metadata,
        inputs=inputs[0],
        outputs=outputs[0],
    )
    source_case = {
        "package_case_id": "case_0000",
        "source": source_identity,
        "design": {"origin": "synthetic"},
    }
    payload = datasets.contracts.identity.build_training_dataset_payload(
        task=synthetic_task,
        dataset_id="synthetic_train",
        sample_ids=("case_0000",),
        source_selection_identity={
            "schema_kind": "dataset_source_selection",
            "source_cases": [source_case],
        },
        source_identities=(source_identity,),
        source_metadata=(expected_metadata,),
        source_provenance={"source_case_identities": [source_case]},
        case_fingerprints=(fingerprint,),
        inputs=inputs,
        outputs=outputs,
    )
    loaded = datasets.runtime.steady.create_dataset(_save_dataset(tmp_path, payload), task=synthetic_task)
    with pytest.raises(RuntimeError, match="contradicts its saved split Dataset identity"):
        datasets.preprocessing.splits.require_source_identity(loaded, {"dataset_id": "synthetic_train"}, role="train", case_ids=("case_0000",))
    sample = loaded[0]
    assert sample["meta"] == expected_metadata
    generator = sample["meta"].get("generator")
    assert isinstance(generator, dict)
    parameters = generator.get("parameters")
    assert isinstance(parameters, dict)
    parameters["scalar_parameter"] = -1.0
    assert loaded[0]["meta"] == expected_metadata

    batch = next(iter(DataLoader(loaded, batch_size=1, shuffle=False)))
    artifact_metadata = analysis.artifacts.generation.meta_to_jsonable(batch["meta"])
    flattened_metadata = analysis.evaluation.dataframe.flatten_meta_scalars(artifact_metadata)

    assert loaded.input_fields == list(synthetic_task.input_names)
    assert loaded.output_fields == list(synthetic_task.output_names)
    assert flattened_metadata["generator_parameters_scalar_parameter"] == _SYNTHETIC_METADATA_VALUE
    assert loaded[0]["x"].shape == (3, 2, 3)
    assert torch.all(loaded[0]["x"][0] == _SYNTHETIC_INPUT_VALUE)
    assert synthetic_task.physics.kind == "none"
    assert tuple(field for group in synthetic_task.output_groups for field in group.fields) == synthetic_task.output_names


def test_artifact_batch_preserves_distinct_grouped_runs_with_equal_leaf_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep independent experiment/trial outputs separate despite equal scientific names, and record each processed run as a job log owner."""
    service = analysis.artifacts.service
    root = tmp_path / "experiments"
    runs = tuple(root / experiment / "fno" / ("a" * 16) / "single" for experiment in ("first", "second"))
    for run in runs:
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(json.dumps({"run": {"name": "same-scientific-name"}}))
    monkeypatch.setattr(service.experiments.run, "validate_evaluable_run", lambda _run: {"scientific_run_name": "same-scientific-name"})
    assert tuple(service.iter_run_dirs(root, run_names=("same-scientific-name",))) == runs
    monkeypatch.setattr(
        service,
        "load_run_artifact_plan",
        lambda run: analysis.artifacts.plan.ArtifactPlan(
            run_dir=run,
            run_name="same-scientific-name",
            task="steady_flow",
            id_role=_role_plan(("case",)),
            ood_roles={},
        ),
    )
    monkeypatch.setattr(service, "run_or_load_artifacts", lambda **kwargs: pd.DataFrame({"run": [str(kwargs["run_dir"])]}))
    monkeypatch.setattr(service, "_upload_published_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(service, "cleanup_runtime", lambda _device: None)
    owners_file = tmp_path / "owners"
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(owners_file))
    outputs = service.build_artifacts(runs_root=root, dataset_root=tmp_path / "datasets", device_policy="cpu")
    assert set(outputs) == {str(run.resolve()) for run in runs}
    for run in runs:
        assert outputs[str(run.resolve())]["eval"]["run"].tolist() == [str(run)]
    assert owners_file.read_text(encoding="utf-8").splitlines() == [str(run.resolve()) for run in runs]


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
@pytest.mark.parametrize("part_count", [0, 2])
def test_artifact_plan_projects_the_admitted_split_of_either_task(tmp_path: Path, task: str, part_count: int) -> None:
    """Plan the ID role and each OOD role over its split parts in split order, and no role for an empty part."""
    parts = (("ood_b", ("pea__case_0002__00000000000b",)), ("ood_a", ("chickpea__case_0001__00000000000a",)))[:part_count]
    parameter = ("lentil__case_0009__000000000009",)[: min(part_count, 1)]
    split = synthetic_split_payload(task, material_ood=parts, parameter_ood=parameter)
    completed = {"config": {"task": task, "run": {"name": "scientific-run"}, "data": {"train_dataset": "train_data"}}, "split_indices": split}
    plan = analysis.artifacts.plan.artifact_plan_from_completed(completed, run_dir=tmp_path)
    assert plan.id_role.source_dataset_ids == ("train_data",)
    assert plan.id_role.case_ids == tuple(split["roles"]["id_test"]["case_ids"])
    if not part_count:
        assert dict(plan.ood_roles) == {}
        assert plan.dataset_roles == ("id",)
        for split_name in ("material_ood", "parameter_ood"):
            with pytest.raises(FileNotFoundError, match=f"no saved {split_name} artifact role"):
                plan.role(split_name)
        return
    assert plan.dataset_roles == ("id", "material_ood", "parameter_ood")
    role = plan.role("material_ood")
    assert role.source_dataset_ids == ("ood_b", "ood_a")
    assert role.case_source_dataset_ids == ("ood_b", "ood_a")
    assert role.dataset_name == datasets.contracts.identity.combined_dataset_id(("ood_b", "ood_a"))
    assert [item["dataset_id"] for item in role.evaluation_assignments] == ["ood_b", "ood_a"]
    parameter_role = plan.role("parameter_ood")
    assert parameter_role.source_dataset_ids == ("train_data",)
    assert parameter_role.case_ids == parameter
    assert parameter_role.root(tmp_path) == tmp_path / "analysis" / "parameter_ood" / "train_data"


def test_artifact_plan_refuses_material_ood_parts_sharing_a_case_id(tmp_path: Path) -> None:
    """Prevent material-OOD parts from colliding in case payload and timing identities."""
    shared = ("chickpea__case_0001__00000000000a",)
    split = synthetic_split_payload("steady_flow", material_ood=(("ood_a", shared), ("ood_b", shared)))
    completed = {"config": {"task": "steady_flow", "run": {"name": "run"}, "data": {"train_dataset": "train_data"}}, "split_indices": split}
    with pytest.raises(ValueError, match="share package case IDs"):
        analysis.artifacts.plan.artifact_plan_from_completed(completed, run_dir=tmp_path)
