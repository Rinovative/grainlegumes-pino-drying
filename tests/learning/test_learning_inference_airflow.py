"""Protect the split-free steady model loader and the steady airflow service on Generation cases of both tasks.

Responsibilities:
  - Check that the steady model loader admits model and Train scaling without any split or Dataset, for steady runs only,
    with the model-code guard first, and that the split-aware loader still refuses a split before building a model
  - Check that the service predicts a steady case exactly as the steady artifact generator does, on the full grid
    whatever the training stride
  - Check that a batch of picks is admitted and Train-checked once and yields the one-pick results in pick order
  - Check the relative L2 per field against an independent formula, including a reference below its floor
  - Check that a transient case yields CPU float32 u, v, p arrays whose reference is the case's own COMSOL airflow
  - Check that another grid of the same shape, other stationary conditioning, a changed Train manifest and a
    non-finite prediction are refused

Design principles:
  - Fake-solver Sources and a published steady Dataset exercise the real admission, conversion and manifest chain
  - A test-owned linear model and constant Train scaling stand in for a trained run; the contract is wiring, not accuracy

This module does NOT:
  - Train a model, judge airflow accuracy or run the transient model on the predicted airflow
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
import torch
import yaml
from support import generation_fixtures, transient_package
from support import scaling as scaling_support
from support.generation_fixtures import successful_source_case
from torch import nn

from src import analysis, common, datasets, domain, experiments, generation, learning
from src.datasets.packages import dataset_packages_builder as builder
from src.datasets.packages import dataset_packages_selection as selection

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# The transient package's grid; the steady Train Source shares it, the other Source keeps its shape on a larger extent.
_TRAIN_GRID = {"nx": 16, "ny": 16, "Lx": 1.5, "Ly": 1.5}
_OTHER_GRID = {"nx": 16, "ny": 16, "Lx": 3.0, "Ly": 3.0}
_CANONICAL = (16, 16)
_SELECTOR_KEYS = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)
_STEADY = domain.tasks.steady_flow.STEADY_FLOW
_AIRFLOW = learning.inference.airflow


def _steady_source(
    factory: Callable[..., tuple[Path, Path]],
    solver: Path,
    storage: Path,
    work: Path,
    *,
    grid: dict[str, float],
    designs: dict[str, int],
    flow_temperature: float | None = None,
) -> str:
    """Run and catalog one complete fake-solver steady Source on ``grid`` and return its Source name."""
    config_path, _ = factory(simulation_profile="steady_flow", executable=solver, designs=designs, grid_overrides=grid)
    if flow_temperature is not None:
        common_path = config_path.parent / "common.yaml"
        common_config = yaml.safe_load(common_path.read_text(encoding="utf-8"))
        common_config["scientific_fixed_values"]["T_flow_ref"]["value"] = flow_temperature
        common_path.write_text(yaml.safe_dump(common_config, sort_keys=False), encoding="utf-8")
    campaign = generation.cases.config.load_campaign_config(config_path)
    generation.lifecycle.create_campaign(campaign, storage_root=storage)
    input_sources: dict[str, dict[str, Any]] = {}
    for batch in campaign.batches:
        inputs = generation.cases.input_generation.generate_input_cases(batch, len(batch.case_indices), storage_root=storage)
        for case_index in batch.case_indices:
            successful_source_case(campaign.source_name, batch, case_index, storage, work)
        input_sources[batch.batch_name] = {
            "input_generation_id": inputs.input_generation_id,
            "source_git_commit": "a" * 40,
            "case_indices": list(batch.case_indices),
        }
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = generation.publication.campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=work / "logs",
        storage_root=storage,
        input_sources=input_sources,
    )
    common.serialization.atomic_write_json(
        generation.publication.campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), run_manifest
    )
    return campaign.source_name


@pytest.fixture(scope="module")
def airflow_storage(_transient_package_template: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Add three steady Sources and a published steady Train Dataset to a private copy of the transient package storage."""
    template = _transient_package_template
    root = tmp_path_factory.mktemp("steady-airflow")
    storage = root / "scientific-storage"
    shutil.copytree(template["root"], storage, symlinks=True)
    steady_project = root / "project"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
        monkeypatch.setenv("PROJECT_ROOT", str(steady_project))
        transient_package.patch_dataset_producer_evidence(monkeypatch)
        factory = generation_fixtures.make_generation_config_factory(steady_project)
        solver = generation_fixtures.write_fake_comsol(root)
        work = root / "work"
        train_source = _steady_source(factory, solver, storage, work, grid=_TRAIN_GRID, designs={"train": 1, "validation": 1, "test": 1})
        other_source = _steady_source(factory, solver, storage, work, grid=_OTHER_GRID, designs={"test": 1})
        # Same grid as Train, but another stationary conditioning; the design counts keep the campaign name distinct.
        conditioned_source = _steady_source(factory, solver, storage, work, grid=_TRAIN_GRID, designs={"test": 2}, flow_temperature=305.0)
        rows = generation.publication.catalog.list_campaign_cases("steady_flow", train_source, storage_root=storage)
        request = {
            "schema_kind": selection.SELECTION_SCHEMA_KIND,
            "source_task": "steady_flow",
            "dataset_name": "steady_airflow_train",
            "dataset_view": "steady_flow",
            "selection": {"kind": "exact"},
            "source_cases": [{key: row[key] for key in _SELECTOR_KEYS} for row in rows],
        }
        dataset_id = builder.build_dataset_selection(request, storage_root=storage, material_roles={"lentil": "id"})["dataset_id"]
    manifest = datasets.packages.load_package_manifest("steady_flow", dataset_id, storage_root=storage)
    return {
        "storage": storage,
        "steady_project": steady_project,
        "transient_project": template["project_root"],
        "train_source": train_source,
        "train_case_ids": [str(row["case_id"]) for row in rows],
        "other_source": other_source,
        "conditioned_source": conditioned_source,
        "transient_source": template["source_campaign"],
        "transient_case_id": template["source_selectors"][0]["case_id"],
        "train_dataset": {"dataset_id": dataset_id, "dataset_digest": manifest["dataset_digest"]},
    }


def _linear_model() -> nn.Module:
    """Return a deterministic linear steady model."""
    model = nn.Conv2d(len(_STEADY.input_names), len(_STEADY.output_names), kernel_size=1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.linspace(-0.5, 0.5, model.weight.numel()).view_as(model.weight))
    return model


class _ConstantModel(nn.Module):
    """Predict one constant encoded state everywhere."""

    def __init__(self, value: float) -> None:
        """Hold the encoded value."""
        super().__init__()
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the value on the input grid for every output channel."""
        return torch.full((inputs.shape[0], len(_STEADY.output_names), *inputs.shape[2:]), self.value)


def _steady_context(
    train_dataset: dict[str, str], *, stride: int = 1, model: nn.Module | None = None
) -> learning.inference.context.SteadyModelContext:
    """Return a steady model whose Train scaling binds the published Train Dataset and was fitted at ``stride``."""
    effective = datasets.contracts.grid.resolve_spatial_view(_CANONICAL, stride)
    scaling = scaling_support._scaling(
        "steady_flow",
        view=scaling_support.spatial_view(effective, stride=stride, canonical=_CANONICAL),
        state_mean=0.1,
        state_std=2.0,
        membership_digest="a" * 64,
        dataset_identity=train_dataset,
    )
    return learning.inference.context.SteadyModelContext(
        run_name="airflow-fixture",
        model=_linear_model() if model is None else model,
        scaling=scaling,
        device=torch.device("cpu"),
        input_names=tuple(_STEADY.input_names),
        output_names=tuple(_STEADY.output_names),
    )


def _predict(
    airflow_storage: dict[str, Any], steady: Any, *, task: str = "steady_flow", source: str = "train_source", case_id: str | None = None
) -> Any:
    """Apply the service to one case of a fixture Source; the default is the last Train-Dataset case."""
    return _AIRFLOW.predict_case_airflow(
        steady,
        task=task,
        source_name=airflow_storage[source],
        material_family="lentil",
        case_id=airflow_storage["train_case_ids"][-1] if case_id is None else case_id,
        storage_root=airflow_storage["storage"],
    )


def _only_case_id(airflow_storage: dict[str, Any], source: str) -> str:
    """Return the case ID of a one-case steady fixture Source."""
    rows = generation.publication.catalog.list_campaign_cases("steady_flow", airflow_storage[source], storage_root=airflow_storage["storage"])
    return str(rows[0]["case_id"])


def _completed_run(model: nn.Module, *, task: str, model_sha256: str | None = None) -> dict[str, Any]:
    """Return the evidence evaluable-run validation would admit for one run of ``task``."""
    digest = learning.training.checkpoint.model_implementation_digest(task) if model_sha256 is None else model_sha256
    return {
        "scientific_run_name": "run",
        "config": {"task": task, "producer": {"training_sha256": "a" * 64, "model_sha256": digest}},
        "split_indices": {},
        "best_checkpoint": {"model_state_dict": model.state_dict()},
        "normalizer_state": scaling_support.steady_scaling((2, 2), state_mean=1.5, state_std=3.0).state_dict(),
    }


def _patch_model_build(monkeypatch: pytest.MonkeyPatch, model: nn.Module, completed: dict[str, Any]) -> None:
    """Admit ``completed`` without files and build ``model`` without a configuration."""
    context = learning.inference.context
    monkeypatch.setattr(experiments.run, "validate_evaluable_run", lambda _run_dir: completed)
    monkeypatch.setattr(context, "_field_contract", lambda _config: (list(_STEADY.input_names), list(_STEADY.output_names)))
    monkeypatch.setattr(experiments.run, "configure_reproducibility", lambda *_args, **_kwargs: {"model_init": 1})
    monkeypatch.setattr(experiments.run, "seed_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(context, "_build_model_from_config", lambda *_args, **_kwargs: model)


def test_steady_model_loads_without_split_or_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The split-free loader returns the admitted model and Train scaling, never touching a split or Dataset, for steady runs only."""
    context = learning.inference.context
    model = _linear_model()
    completed = _completed_run(model, task="steady_flow")
    _patch_model_build(monkeypatch, model, completed)
    monkeypatch.setattr(context, "_select_split", lambda **_kwargs: pytest.fail("the steady model loader selected a split"))
    monkeypatch.setattr(context, "_open_split_sources", lambda *_args, **_kwargs: pytest.fail("the steady model loader opened a Dataset"))
    resolution = learning.device.resolve_device("cpu", path="device")

    loaded = context.load_steady_model(run_dir=tmp_path / "run", device_resolution=resolution)

    assert loaded.model is model
    assert loaded.scaling.digest == datasets.preprocessing.normalization.TrainScaling.from_state_dict(completed["normalizer_state"]).digest
    assert (loaded.run_name, loaded.device, loaded.input_names, loaded.output_names) == (
        "run",
        torch.device("cpu"),
        tuple(_STEADY.input_names),
        tuple(_STEADY.output_names),
    )
    monkeypatch.setattr(experiments.run, "validate_evaluable_run", lambda _run_dir: _completed_run(model, task="transient_drying"))
    with pytest.raises(ValueError, match="steady_flow runs only"):
        context.load_steady_model(run_dir=tmp_path / "run", device_resolution=resolution)


def test_stale_model_code_is_reported_before_the_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient run whose recorded model code changed gets the model-code error, as before the steady-only guard."""
    stale = _completed_run(_linear_model(), task="transient_drying", model_sha256="0" * 64)
    monkeypatch.setattr(experiments.run, "validate_evaluable_run", lambda _run_dir: stale)

    with pytest.raises(ValueError, match="implementation changed"):
        learning.inference.context.load_inference_context(run_dir=tmp_path / "run", device_policy="cpu")


def test_split_aware_loader_refuses_a_split_before_building_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown split role is refused before the model is built, so no construction or RNG use precedes it."""
    _patch_model_build(monkeypatch, _linear_model(), _completed_run(_linear_model(), task="steady_flow"))
    monkeypatch.setattr(
        learning.inference.context, "_build_model_from_config", lambda *_args, **_kwargs: pytest.fail("built a model before the split")
    )

    with pytest.raises(ValueError, match="Unknown inference split"):
        learning.inference.context.load_inference_context(run_dir=tmp_path / "run", split="unknown", device_policy="cpu")


@pytest.mark.integration
@pytest.mark.parametrize("stride", [1, 3])
def test_steady_case_prediction_equals_the_artifact_generator_on_the_full_grid(
    airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch, stride: int
) -> None:
    """The service and the steady artifact generator predict one Train-Dataset case identically on the full grid, whatever the training stride."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))
    steady = _steady_context(airflow_storage["train_dataset"], stride=stride)

    result = _predict(airflow_storage, steady)

    request = datasets.runtime.factory.DatasetRequest(
        dataset_id=airflow_storage["train_dataset"]["dataset_id"], dataset_view="steady_flow", storage_root=airflow_storage["storage"]
    )
    dataset = datasets.runtime.factory.create_dataset(request)
    assert isinstance(dataset, datasets.runtime.steady.SteadyFlowDataset)
    position = dataset.identity.sample_ids.index(result.package_case_id)
    loader = learning.inference.context._build_eval_loader(learning.inference.context.IndexedSubset(dataset, torch.tensor([position])), batch_size=1)
    ((_, case),) = list(
        analysis.artifacts.generation._iter_inferred_artifact_cases(loader, model=steady.model, scaling=steady.scaling, device=steady.device)
    )
    order = [_STEADY.output_names.index(name) for name in _AIRFLOW.AIRFLOW_FIELDS]
    assert result.prediction.shape == (3, *_CANONICAL)
    assert torch.equal(result.prediction, case["prediction"][0][order])
    assert torch.equal(result.reference, case["y"][0][order])
    assert torch.equal(result.error, result.prediction - result.reference)


@pytest.mark.integration
def test_case_batch_is_admitted_once_and_predicts_like_single_cases(airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Two picks run one exact selection, one planner admission and one Train check, and yield the one-pick results in pick order."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))
    steady = _steady_context(airflow_storage["train_dataset"])
    storage = airflow_storage["storage"]
    case_ids = airflow_storage["train_case_ids"][:2]
    picks = [{"source_name": airflow_storage["train_source"], "material_family": "lentil", "case_id": case_id} for case_id in case_ids]
    counts: Counter[str] = Counter()

    def counted(name: str, original: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for module, name in (
        (datasets.packages, "select_exact_cases"),
        (datasets.packages, "load_package_manifest_evidence"),
        (datasets.packages.planning, "prepare_dataset_selection"),
    ):
        monkeypatch.setattr(module, name, counted(name, getattr(module, name)))

    batch = list(_AIRFLOW.predict_cases_airflow(steady, task="steady_flow", picks=picks, storage_root=storage))
    batch_counts = dict(counts)
    counts.clear()
    singles = [_predict(airflow_storage, steady, case_id=case_id) for case_id in case_ids]

    # The planner admits the picks once and the Train reference once; one-pick calls repeat all three per case.
    assert batch_counts == {"select_exact_cases": 1, "load_package_manifest_evidence": 1, "prepare_dataset_selection": 2}
    assert dict(counts) == {"select_exact_cases": 2, "load_package_manifest_evidence": 2, "prepare_dataset_selection": 4}
    assert [result.source_identity["case_id"] for result in batch] == case_ids
    for result, single in zip(batch, singles, strict=True):
        assert result.package_case_id == single.package_case_id
        assert (result.source_identity, result.relative_l2) == (single.source_identity, single.relative_l2)
        for name in ("reference", "prediction", "error"):
            assert torch.equal(getattr(result, name), getattr(single, name))


@pytest.mark.integration
def test_relative_l2_follows_its_definition_and_the_reference_floor(airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Each field's relative L2 is sqrt(sum e^2 / sum r^2) of its own channel; a reference below its floor gives None."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))
    convert = datasets.packages.generated_batch.interpret_admitted_source_case

    def zero_v_reference(*args: Any, **kwargs: Any) -> Any:
        shape, inputs, outputs, *rest = convert(*args, **kwargs)
        outputs = outputs.clone()
        outputs[_STEADY.output_names.index("v")] = 0.0
        return (shape, inputs, outputs, *rest)

    monkeypatch.setattr(datasets.packages.generated_batch, "interpret_admitted_source_case", zero_v_reference)

    result = _predict(airflow_storage, _steady_context(airflow_storage["train_dataset"]))

    prediction, reference = result.prediction.double().numpy(), result.reference.double().numpy()
    expected = {
        name: float(np.sqrt(np.sum((prediction[index] - reference[index]) ** 2) / np.sum(reference[index] ** 2)))
        for index, name in enumerate(("u", "v", "p"))
        if name != "v"
    }
    assert result.relative_l2["v"] is None
    assert result.relative_l2["u"] == pytest.approx(expected["u"], rel=1e-12)
    assert result.relative_l2["p"] == pytest.approx(expected["p"], rel=1e-12)
    assert expected["u"] != pytest.approx(expected["p"])


@pytest.mark.integration
def test_transient_case_yields_cpu_float32_uvp_with_its_own_comsol_reference(
    airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient case's arrays are CPU float32 [3, Y, X] in u, v, p order, and the reference is its case.h5 airflow."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["transient_project"]))
    storage = airflow_storage["storage"]

    result = _predict(
        airflow_storage,
        _steady_context(airflow_storage["train_dataset"]),
        task="transient_drying",
        source="transient_source",
        case_id=airflow_storage["transient_case_id"],
    )

    rows = generation.publication.catalog.list_campaign_cases("transient_drying", airflow_storage["transient_source"], storage_root=storage)
    (row,) = [row for row in rows if row["case_id"] == airflow_storage["transient_case_id"]]
    with h5py.File(storage / row["case_hdf5_relative"], "r") as handle:
        static = handle["static/fields"]
        assert isinstance(static, h5py.Dataset)
        raw_names = static.attrs["field_names"]
        names = json.loads(raw_names.decode() if isinstance(raw_names, bytes) else str(raw_names))
        expected = np.stack([np.asarray(static[names.index(name)], dtype=np.float32) for name in ("u", "v", "p")])
    assert result.fields == ("u", "v", "p")
    assert result.source_identity["task"] == "transient_drying"
    for array in (result.reference, result.prediction, result.error):
        assert (array.shape, array.dtype, array.device.type) == ((3, 16, 16), torch.float32, "cpu")
    assert np.array_equal(result.reference.numpy(), expected)


@pytest.mark.integration
def test_case_on_another_grid_of_the_same_shape_is_refused(airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A case whose coordinates differ from the run's Train grid is refused even though its shape is the same."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))
    steady = _steady_context(airflow_storage["train_dataset"])

    with pytest.raises(ValueError, match="differs from the grid"):
        _predict(airflow_storage, steady, source="other_source", case_id=_only_case_id(airflow_storage, "other_source"))


@pytest.mark.integration
def test_case_with_other_stationary_conditioning_is_refused(airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A case on the Train grid whose stationary conditioning (here T_flow_ref) differs from the Train Dataset's is refused."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))
    steady = _steady_context(airflow_storage["train_dataset"])

    with pytest.raises(ValueError, match="stationary conditioning"):
        _predict(airflow_storage, steady, source="conditioned_source", case_id=_only_case_id(airflow_storage, "conditioned_source"))


@pytest.mark.integration
def test_changed_train_manifest_is_refused(airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A Train Dataset whose current manifest digest is not the one the run recorded is refused before any prediction."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))
    recorded = {**airflow_storage["train_dataset"], "dataset_digest": "0" * 64}

    with pytest.raises(RuntimeError, match="manifest digest"):
        _predict(airflow_storage, _steady_context(recorded))


@pytest.mark.integration
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_prediction_is_refused(airflow_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """A model that diverges to NaN or infinity yields no airflow that external_airflow would reject later."""
    monkeypatch.setenv("PROJECT_ROOT", str(airflow_storage["steady_project"]))

    with pytest.raises(ValueError, match="non-finite"):
        _predict(airflow_storage, _steady_context(airflow_storage["train_dataset"], model=_ConstantModel(value)))
