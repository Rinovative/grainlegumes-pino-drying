"""Hold both Dataset views to one publication, admission, split and scaling contract.

Responsibilities:
  - Check that an interrupted publication leaves no Dataset folder and no staging residue for either view
  - Check that the folder renamed into place is already complete and that a published Dataset opens at once
  - Check that training refuses a Dataset without Train cases, and that a Source HDF5 changed after admission, a
    manifest with corrupted identity or changed payload bytes is refused, with one exception type and message for both views
  - Check that both training runtimes replay a saved split only under the same partition and manifest digest, and reuse
    the Train-only scaling cache only under its exact identity and recorded model digest, refusing a misplaced entry

Design principles:
  - Fake-solver Sources exercise the real admission, staging and publication chain of each view
  - Shared contracts are parametrized over the registered task instead of tested once per implementation

This module does NOT:
  - Judge physical validity of the fake-solver Sources or recompute scaling statistics
  - Test Dataset-plan orchestration or split recipes, which have their own tests
"""

from __future__ import annotations

import copy
import json
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support.generation_fixtures import successful_source_case

from src import common, domain, generation
from src.datasets.contracts.dataset_contracts_transient import TransientSamplingSpec
from src.datasets.packages import dataset_packages_builder as builder
from src.datasets.packages import dataset_packages_manifest as manifests
from src.datasets.packages import dataset_packages_planning as planning
from src.datasets.packages import dataset_packages_reproduction as reproduction
from src.datasets.packages import dataset_packages_selection as selection
from src.datasets.preprocessing import dataset_preprocessing_splits as splits
from src.datasets.runtime import dataset_runtime_factory as factory
from src.datasets.runtime import dataset_runtime_training as steady_training
from src.datasets.runtime import dataset_runtime_transient as transient_runtime
from src.datasets.runtime import dataset_runtime_transient_training as transient_training
from src.learning.learning_temporal import TemporalConditioningSpec
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("_dataset_producer_evidence")]

_SELECTOR_KEYS = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)
_CASES = 3
_THIRDS = {"kind": "seeded_material_fraction", "train_fraction": 1 / 3, "validation_fraction": 1 / 3, "id_test_fraction": 1 / 3}
_MODEL_SHA256 = "a" * 64


def _steady_source(request: pytest.FixtureRequest, root: Path) -> tuple[Path, list[dict[str, str]]]:
    """Run and catalog one complete fake-solver steady Source of three lentil cases."""
    make_config: Callable[..., tuple[Path, Path]] = request.getfixturevalue("generation_config_factory")
    config_path, _ = make_config(simulation_profile="steady_flow", executable=request.getfixturevalue("fake_comsol"), natural_count=_CASES)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = root / "scientific-storage"
    generation.lifecycle.create_campaign(campaign, storage_root=storage)
    batch = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(batch, _CASES, storage_root=storage)
    for case_index in batch.case_indices:
        successful_source_case(campaign.source_name, batch, case_index, storage, root / "work")
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = generation.publication.campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=root / "logs",
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
    return storage, [{key: row[key] for key in _SELECTOR_KEYS} for row in rows]


@pytest.fixture(params=["steady_flow", "transient_drying"])
def source(request: pytest.FixtureRequest, tmp_path: Path) -> dict[str, Any]:
    """Return the storage root and three successful Source cases of one registered task."""
    if request.param == "transient_drying":
        package = request.getfixturevalue("transient_package")
        return {"task": request.param, "storage": package["root"], "selectors": package["source_selectors"][:_CASES]}
    storage, selectors = _steady_source(request, tmp_path)
    return {"task": request.param, "storage": storage, "selectors": selectors}


def _publish(source: dict[str, Any], *, role: str = "id") -> dict[str, Any]:
    """Publish the three cases as one Dataset whose single material has ``role``."""
    request = {
        "schema_kind": selection.SELECTION_SCHEMA_KIND,
        "schema_version": selection.SELECTION_SCHEMA_VERSION,
        "source_task": source["task"],
        "dataset_name": "publication",
        "dataset_view": source["task"],
        "selection": {"kind": "exact"},
        "source_cases": source["selectors"],
    }
    partition_spec = {"seed_input": 5, "seed": 5, "recipe": _THIRDS, "material_roles": {"lentil": role}}
    return builder.build_dataset_selection(request, storage_root=source["storage"], partition_spec=partition_spec)


def _open_training(
    source: dict[str, Any], dataset_id: str, *, saved_split: Any = None, model_sha256: str = _MODEL_SHA256, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open the training loaders of the source's task and return the split payload and the scaling artifact."""
    if source["task"] == "steady_flow":
        *_, split, artifact = steady_training.create_dataloaders(
            train_dataset_id=dataset_id,
            additional_evaluation_dataset_ids=(),
            task=domain.tasks.registry.get_task("steady_flow"),
            model_sha256=model_sha256,
            storage_root=source["storage"],
            split_indices=saved_split,
            **kwargs,
        )
        return split, artifact
    loaders = transient_training.create_transient_training_loaders(
        train_dataset_id=dataset_id,
        additional_evaluation_dataset_ids=(),
        tensorizer=TransientTensorizerSpec(
            input_profile="canonical_physics_complete_v1", temporal_conditioning=TemporalConditioningSpec("normalized_current_time")
        ),
        train_sampling=TransientSamplingSpec(mode="one_step_transition"),
        loader_settings=factory.LoaderSettings(),
        model_sha256=model_sha256,
        storage_root=str(source["storage"]),
        saved_split=saved_split,
        **kwargs,
    )
    for loader in (loaders.train, loaders.evaluation, loaders.additional_evaluation, loaders.id_test):
        if loader is not None and isinstance(loader.dataset, transient_runtime.TransientPhysicalDataset):
            loader.dataset.close()
    return loaders.split, loaders.scaling_artifact


def test_interrupted_publication_publishes_nothing(source: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail after manifest and complete payload are staged: no Dataset folder appears and no staging folder remains."""
    storage = source["storage"]
    task_root = common.paths.get_datasets_root(storage_root=storage) / source["task"]
    state_root = common.paths.get_dataset_state_root(storage_root=storage)
    before = sorted(task_root.iterdir()) if task_root.exists() else []
    staged_part = "payload/transient_pt_shards/receipt.json" if source["task"] == "transient_drying" else "payload/dataset.pt"
    staged: list[list[Path]] = []

    def interrupt(*_args: Any, **_kwargs: Any) -> None:
        staged.append(sorted(state_root.glob(f"*.tmp/{staged_part}")))
        if len(staged) == 2:  # the producer check right before the publishing rename
            message = "interrupted before publication"
            raise RuntimeError(message)

    monkeypatch.setattr(reproduction, "verify_current_producer", interrupt)
    with pytest.raises(RuntimeError, match="interrupted before publication"):
        _publish(source)

    assert staged[1], "the complete payload must have been staged when the build was interrupted"
    assert (sorted(task_root.iterdir()) if task_root.exists() else []) == before
    assert not list(state_root.glob("*.tmp"))


def test_published_dataset_opens_without_a_second_step(source: dict[str, Any]) -> None:
    """Open the runtime Dataset directly after publication; a transient one is served from its published PT shards."""
    published = _publish(source)
    transient = source["task"] == "transient_drying"
    request = factory.DatasetRequest(
        dataset_id=published["dataset_id"],
        dataset_view=source["task"],
        transient_sampling=TransientSamplingSpec(mode="one_step_transition") if transient else None,
        storage_root=source["storage"],
    )
    dataset = factory.create_dataset(request)
    try:
        assert dataset[0] is not None
        assert isinstance(dataset, transient_runtime.TransientPTShardDataset) is transient
    finally:
        if isinstance(dataset, transient_runtime.TransientPhysicalDataset):
            dataset.close()


def test_training_refuses_a_dataset_without_train_cases(source: dict[str, Any]) -> None:
    """An evaluation-only Dataset (every case OOD) publishes, but training on it is refused with one message."""
    dataset_id = _publish(source, role="ood")["dataset_id"]
    with pytest.raises(ValueError, match="has no fixed case partition with Train cases"):
        _open_training(source, dataset_id)


def test_source_changed_after_admission_is_refused_alike(source: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Change one admitted Source HDF5 before its payload is built: both views refuse with the same type and message."""
    admit = planning.prepare_dataset_selection

    def admit_then_change(request: dict[str, Any], **kwargs: Any) -> planning.PreparedPackage:
        prepared = admit(request, **kwargs)
        path = prepared.cases[0].hdf5_path
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
        with path.open("ab") as handle:
            handle.write(b"\0")
        return prepared

    monkeypatch.setattr(planning, "prepare_dataset_selection", admit_then_change)
    with pytest.raises(ValueError, match="Source HDF5 changed after Dataset admission") as refused:
        _publish(source)
    assert type(refused.value) is ValueError


def test_manifest_with_corrupted_identity_is_refused(source: dict[str, Any]) -> None:
    """The one manifest validator refuses a published manifest whose content digest no longer binds its identity."""
    published = _publish(source)
    path = Path(published["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_digest"] = "0" * 64
    path.chmod(path.stat().st_mode | stat.S_IWUSR)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind exact identity"):
        manifests.load_package_manifest(source["task"], published["dataset_id"], storage_root=source["storage"])


def _files(directory: Path) -> set[str]:
    """Return every regular file below ``directory`` as a relative path."""
    return {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}


def test_published_folder_is_complete_at_the_rename(source: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The staged folder renamed into place already holds the admitted manifest, the payload and every derived part."""
    state_root = common.paths.get_dataset_state_root(storage_root=source["storage"])
    verify = reproduction.verify_current_producer
    at_rename: list[set[str]] = []

    def record(*args: Any, **kwargs: Any) -> None:
        # Only the producer check right before the publishing rename sees a staged manifest.
        for staged in state_root.glob(f"*.tmp/payload/{common.paths.DATASET_MANIFEST_FILENAME}"):
            manifest = json.loads(staged.read_text(encoding="utf-8"))
            manifests.read_dataset_manifest(staged.parent, task=source["task"], dataset_id=manifest["dataset_id"])
            at_rename.append(_files(staged.parent))
        verify(*args, **kwargs)

    monkeypatch.setattr(reproduction, "verify_current_producer", record)
    published = _publish(source)
    folder = Path(published["manifest_path"]).parent
    assert at_rename == [_files(folder)]
    assert not [name for name in at_rename[0] if name.endswith(".log")]


def test_manifest_refuses_changed_payload_bytes(source: dict[str, Any]) -> None:
    """The manifest binds the payload and, for the transient view, its PT shard receipt; changed bytes are refused alike."""
    published = _publish(source)
    transient = source["task"] == "transient_drying"
    folder = Path(published["manifest_path"]).parent
    manifest = manifests.load_package_manifest(source["task"], published["dataset_id"], storage_root=source["storage"])
    assert (manifest["derived_payload_sha256"] is not None) is transient
    target = folder / "transient_pt_shards" / "receipt.json" if transient else folder / manifest["payload_filename"]
    target.chmod(target.stat().st_mode | stat.S_IWUSR)
    with target.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="payload bytes disagree"):
        manifests.load_package_manifest(source["task"], published["dataset_id"], storage_root=source["storage"])


def test_saved_split_is_admitted_and_refused_on_a_changed_partition_digest(source: dict[str, Any]) -> None:
    """Both runtimes persist one split schema, replay it unchanged and refuse a changed partition digest."""
    dataset_id = _publish(source)["dataset_id"]
    split, _ = _open_training(source, dataset_id)
    admitted = splits.admit_split(split)
    assert admitted.task == source["task"]
    assert (admitted.extension is None) is (source["task"] == "steady_flow")
    assert _open_training(source, dataset_id, saved_split=split)[0] == split
    changed = copy.deepcopy(split)
    changed["case_partition_digest"] = "0" * 64
    with pytest.raises(ValueError, match="case partition digest mismatch"):
        _open_training(source, dataset_id, saved_split=changed)


def test_runtime_refuses_a_saved_split_of_another_manifest_digest(source: dict[str, Any]) -> None:
    """A split that is admissible on its own but binds another manifest digest is refused by both runtimes."""
    dataset_id = _publish(source)["dataset_id"]
    split = splits.admit_split(_open_training(source, dataset_id)[0])
    other = splits.build_split(
        task=split.task,
        train_dataset={**split.train.dataset, "dataset_digest": "0" * 64},
        case_partition=split.case_partition,
        additional_evaluation=[],
        extension=split.extension,
    )
    with pytest.raises(ValueError, match="does not match the current Dataset manifests"):
        _open_training(source, dataset_id, saved_split=other.as_payload())


def test_scaling_cache_hit_miss_and_refused_mismatch(source: dict[str, Any]) -> None:
    """Fit once on Train membership, reuse the cached artifact under its identity and model digest, and refuse misplaced or foreign entries."""
    dataset_id = _publish(source)["dataset_id"]
    events: list[str] = []
    phases: set[str] = set()

    def record(event: str, values: Any) -> None:
        phases.add(event)
        if event == "scaler_cache":
            events.append(str(values["state"]))
        elif event == "scaler_fit_start":
            events.append("fit")

    split, fitted = _open_training(source, dataset_id, startup_callback=record)
    shared = {"dataset_resolved", "scaler_cache", "scaler_fit_start", "scaler_fit_complete", "dataloader_ready"}
    # Both tasks report the same startup phases; only the transient fit adds PT-shard progress.
    assert shared <= phases
    assert phases - shared <= {"scaler_fit_progress"}
    _, cached = _open_training(source, dataset_id, startup_callback=record)
    assert events == ["miss", "fit", "hit"]
    assert cached["identity"] == fitted["identity"]
    train = splits.admit_split(split).train
    assert fitted["identity"]["train_membership_digest"] == train.membership_digest
    assert fitted["identity"]["dataset"]["dataset_digest"] == train.dataset["dataset_digest"]
    assert fitted["producer"] == {"model_sha256": _MODEL_SHA256}
    # Changed model code misses the entry and fits again; the artifact identity stays data-only.
    _, refitted = _open_training(source, dataset_id, startup_callback=record, model_sha256="b" * 64)
    assert events == ["miss", "fit", "hit", "miss", "fit"]
    assert refitted["identity"] == fitted["identity"]
    assert refitted["producer"] == {"model_sha256": "b" * 64}
    entries = list(common.paths.get_scaling_cache_root(storage_root=source["storage"]).glob("*.pt"))
    assert len(entries) == 2
    # An entry stored under another model's key is refused, never reused.
    for entry in entries:
        torch.save(fitted, entry)
    with pytest.raises(ValueError, match="produced under another key"):
        _open_training(source, dataset_id, model_sha256="b" * 64)
    identity = {**fitted["identity"], "train_membership_digest": "0" * 64}
    for entry in entries:
        torch.save({**fitted, "identity": identity, "identity_sha256": common.serialization.canonical_json_sha256(identity)}, entry)
    with pytest.raises(ValueError, match="Scaling artifact identity does not match"):
        _open_training(source, dataset_id)
