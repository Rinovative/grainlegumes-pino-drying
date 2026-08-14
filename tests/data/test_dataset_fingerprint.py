# ruff: noqa: S101
"""
Protect deterministic, content-bound dataset and saved-membership identity.

The tests vary case order, tensors, metadata, sample IDs, and split indices to
show that equivalent content is stable and tampering fails strict verification.
Direct-builder transactions are protected by the profile integration smoke. Large
production tensors are deliberately not loaded.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support.synthetic_task import build_synthetic_generated_batch_identity

from src import datasets, domain

_EXPECTED_DISTINCT_FINGERPRINTS = 4
_COMBINED_OOD_SAMPLE_COUNT = 8

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _reordered_payload(
    payload: dict[str, Any],
    *,
    order: list[int],
    task: domain.tasks.spec.TaskSpec,
) -> dict[str, Any]:
    """
    Rebuild a payload after applying one order to every sample-aligned component.

    Tensors, source evidence, metadata, and case fingerprints stay paired while
    generated-batch identity follows the reordered manifest membership.
    """
    return datasets.contracts.identity.build_training_dataset_payload(
        task=task,
        dataset_id=payload["dataset_id"],
        sample_ids=[payload["sample_ids"][index] for index in order],
        generated_batch_identity=build_synthetic_generated_batch_identity(
            batch_id=payload["generated_batch_identity"]["batch_id"],
            sample_ids=[payload["sample_ids"][index] for index in order],
        ),
        source_identities=[payload["source_identities"][index] for index in order],
        source_metadata=[payload["source_metadata"][index] for index in order],
        source_provenance=payload["source_provenance"],
        case_fingerprints=[payload["case_fingerprints"][index] for index in order],
        inputs=payload["inputs"][order],
        outputs=payload["outputs"][order],
    )


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """
    Save one strict payload below its logical dataset directory.

    This test helper intentionally uses direct ``torch.save`` rather than the
    production atomic publisher because callers exercise read-time identity only.
    """
    dataset_id = payload["dataset_id"]
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.pt"
    torch.save(payload, path)
    return path


def test_creation_computes_stable_content_identity(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Build identical strict final content twice, then verify one payload by content.

    Both constructions and strict validation must report the same fingerprint, proving
    deterministic identity for reproducible split and cache admission.
    """
    first = training_dataset_payload_factory()
    second = training_dataset_payload_factory()

    strict_identity = datasets.contracts.identity.validate_training_dataset_payload(
        first,
        task=steady_task,
        verify_content=True,
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert strict_identity.fingerprint == first["dataset_fingerprint"]


def test_dataset_reader_requires_actual_float32_tensors(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reject array coercion and non-float32 tensors at the persisted boundary."""
    array_payload = training_dataset_payload_factory()
    array_payload["inputs"] = array_payload["inputs"].numpy()
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        datasets.contracts.identity.validate_training_dataset_payload(array_payload, task=steady_task)

    with pytest.raises(TypeError, match=r"torch\.float32"):
        training_dataset_payload_factory(dtype=torch.float64)


def test_source_metadata_is_aligned_and_fingerprint_bound(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Change and misalign ordered source metadata around an otherwise fixed payload.

    Rebuilding must change the fingerprint, while post-build tampering or length
    drift must be rejected so provenance cannot detach from sample membership.
    """
    original = training_dataset_payload_factory()
    changed_metadata = copy.deepcopy(original["source_metadata"])
    changed_metadata[0]["case_id"] = "changed_case"
    changed = datasets.contracts.identity.build_training_dataset_payload(
        task=steady_task,
        dataset_id=original["dataset_id"],
        sample_ids=original["sample_ids"],
        generated_batch_identity=original["generated_batch_identity"],
        source_identities=original["source_identities"],
        source_metadata=changed_metadata,
        source_provenance=original["source_provenance"],
        case_fingerprints=original["case_fingerprints"],
        inputs=original["inputs"],
        outputs=original["outputs"],
    )

    assert original["schema_version"] == datasets.contracts.identity.TRAINING_DATASET_SCHEMA_VERSION
    assert original["source_metadata"][0]["case_id"] == "case_0000"
    assert changed["dataset_fingerprint"] != original["dataset_fingerprint"]

    tampered = copy.deepcopy(original)
    tampered["source_metadata"][0]["case_id"] = "tampered_case"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.contracts.identity.validate_training_dataset_payload(
            tampered,
            task=steady_task,
            verify_content=True,
        )

    misaligned = copy.deepcopy(original)
    misaligned["source_metadata"].pop()
    with pytest.raises(ValueError, match="source_metadata must align"):
        datasets.contracts.identity.validate_training_dataset_payload(
            misaligned,
            task=steady_task,
        )


def test_ordered_membership_changes_fingerprint(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Vary case order, membership, and source identity independently.

    Every family must produce a distinct fingerprint because each changes the
    scientific dataset consumed by a saved run.
    """
    original = training_dataset_payload_factory()
    reordered = _reordered_payload(
        original,
        order=[1, 0, 2, 3],
        task=steady_task,
    )
    missing = datasets.contracts.identity.build_training_dataset_payload(
        task=steady_task,
        dataset_id="tiny",
        sample_ids=original["sample_ids"][:-1],
        generated_batch_identity=build_synthetic_generated_batch_identity(
            batch_id=original["generated_batch_identity"]["batch_id"],
            sample_ids=original["sample_ids"][:-1],
        ),
        source_identities=original["source_identities"][:-1],
        source_metadata=original["source_metadata"][:-1],
        source_provenance=original["source_provenance"],
        case_fingerprints=original["case_fingerprints"][:-1],
        inputs=original["inputs"][:-1],
        outputs=original["outputs"][:-1],
    )
    changed = training_dataset_payload_factory(
        source_tokens=("case_0000", "replacement", "case_0002", "case_0003"),
    )

    fingerprints = {
        original["dataset_fingerprint"],
        reordered["dataset_fingerprint"],
        missing["dataset_fingerprint"],
        changed["dataset_fingerprint"],
    }
    assert len(fingerprints) == _EXPECTED_DISTINCT_FINGERPRINTS


def test_default_dataset_load_rejects_modified_tensor_content(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Mutate saved tensor content without recomputing its persisted fingerprint.

    The default dataset loader must reject the mismatch, proving ordinary
    consumers do not silently bypass content verification.
    """
    payload = copy.deepcopy(training_dataset_payload_factory())
    payload["inputs"][0, 0, 0, 0] += 1.0
    path = tmp_path / "modified.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.runtime.steady.create_dataset(path, task=steady_task)


def test_duplicate_sample_id_is_rejected(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Duplicate one sample identifier in an otherwise current final payload.

    Validation must reject the collision because ordered membership digests rely
    on each logical sample having a unique identity.
    """
    payload = copy.deepcopy(training_dataset_payload_factory())
    payload["sample_ids"][1] = payload["sample_ids"][0]
    with pytest.raises(ValueError, match="duplicate identifiers"):
        datasets.contracts.identity.validate_training_dataset_payload(payload, task=steady_task)


def test_multiple_ood_packages_form_one_ordered_identity_bound_loader(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Combine independent OOD packages without losing their exact ordered identities."""
    train_path = _save_dataset(tmp_path, training_dataset_payload_factory("combined_train"))
    first_path = _save_dataset(tmp_path, training_dataset_payload_factory("parameter_ood"))
    second_path = _save_dataset(tmp_path, training_dataset_payload_factory("family_ood"))
    _train_loader, _test_loaders, _processor, split_info = datasets.runtime.training.create_dataloaders(
        path_train=str(train_path),
        path_test_ood=(str(first_path), str(second_path)),
        task=steady_task,
        train_dataset_id="combined_train",
        ood_dataset_id=("parameter_ood", "family_ood"),
        batch_size=2,
        train_ratio=0.5,
        ood_fraction=1.0,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        split_seed=9,
    )

    split_contract = datasets.preprocessing.splits.admit_split_contract(split_info)
    assert split_contract.role("eval").source is split_contract.role("train").source
    assert split_contract.role("train").source.dataset_id == "combined_train"
    assert split_contract.role("ood").count == _COMBINED_OOD_SAMPLE_COUNT
    isolated_indices = split_contract.role("ood").indices
    isolated_indices.fill_(-1)
    assert min(split_contract.role("ood").index_values) >= 0
    rebuilt = split_contract.as_payload()
    assert list(rebuilt) == list(split_info)
    assert rebuilt["metadata"] == split_info["metadata"]
    for key in ("train_indices", "eval_indices", "ood_indices"):
        assert torch.equal(rebuilt[key], split_info[key])
        assert rebuilt[key].data_ptr() != split_info[key].data_ptr()

    expected_id = datasets.contracts.identity.combined_dataset_id(("parameter_ood", "family_ood"))
    assert split_info["metadata"]["datasets"]["ood"]["dataset_id"] == expected_id
    assert expected_id != datasets.contracts.identity.combined_dataset_id(("family_ood", "parameter_ood"))
    assert split_info["ood_indices"].numel() == _COMBINED_OOD_SAMPLE_COUNT
    assert len(set(split_info["metadata"]["datasets"]["ood"]["sample_ids"])) == _COMBINED_OOD_SAMPLE_COUNT


def test_saved_split_rejects_replaced_same_name_count_dataset(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Replace a saved training dataset with equal-name, equal-count new content.

    Reusing the original split must fail by fingerprint so path and cardinality
    cannot masquerade as the dataset identity used for training.
    """
    train_payload = training_dataset_payload_factory("train")
    ood_payload = training_dataset_payload_factory("ood")
    train_path = _save_dataset(tmp_path, train_payload)
    ood_path = _save_dataset(tmp_path, ood_payload)
    loader_args = {
        "path_train": str(train_path),
        "path_test_ood": str(ood_path),
        "task": steady_task,
        "train_dataset_id": "train",
        "ood_dataset_id": "ood",
        "batch_size": 1,
        "train_ratio": 0.5,
        "ood_fraction": 0.5,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "split_seed": 9,
    }
    *_, split_info = datasets.runtime.training.create_dataloaders(**loader_args)
    assert split_info["task_contract_digest"] == steady_task.contract_digest
    assert split_info["metadata"]["datasets"]["train"]["data_contract_digest"] == steady_task.data_contract_digest
    assert split_info["metadata"]["datasets"]["ood"]["data_contract_digest"] == steady_task.data_contract_digest
    stale_header = copy.deepcopy(split_info)
    stale_header["task_contract_digest"] = "8cdaf4de22d945e08783f118d5fa8374e37521f91b20b12c913230ba015ca91a"
    with pytest.raises(ValueError, match="current registered task"):
        datasets.preprocessing.splits.admit_split_contract(stale_header)

    replaced = training_dataset_payload_factory(
        "train",
        source_tokens=("same", "same", "replacement", "same"),
    )
    torch.save(replaced, train_path)
    with pytest.raises(ValueError, match="identity does not match"):
        datasets.runtime.training.create_dataloaders(
            **loader_args,
            split_indices=split_info,
        )


def _valid_normalizer_state() -> dict[str, torch.Tensor]:
    """Return one strict current 2D saved-normalizer state."""
    return {
        "in_normalizer.mean": torch.zeros(1, 2, 1, 1),
        "in_normalizer.std": torch.ones(1, 2, 1, 1),
        "out_normalizer.mean": torch.zeros(1, 3, 1, 1),
        "out_normalizer.std": torch.ones(1, 3, 1, 1),
    }


@pytest.mark.parametrize(
    ("key", "replacement", "error_type", "match"),
    [
        ("out_normalizer.std", -torch.ones(1, 3, 1, 1), ValueError, "non-negative"),
        ("out_normalizer.mean", torch.full((1, 3, 1, 1), float("inf")), ValueError, "non-finite"),
        ("in_normalizer.mean", torch.zeros(1, 2, 1, dtype=torch.complex64), TypeError, "real floating-point"),
        ("in_normalizer.mean", torch.zeros(1, 2, 1), ValueError, "must have shape"),
    ],
    ids=("negative-std", "non-finite", "complex", "wrong-rank"),
)
def test_saved_normalizer_state_fails_closed(
    key: str,
    replacement: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    """
    Vary one saved statistic across negative, non-finite, complex, and wrong-rank forms.

    Each parameter family must fail with its type/domain-specific error while
    valid BCHW keys remain fixed, protecting preprocessing reconstruction.
    """
    state = _valid_normalizer_state()
    state[key] = replacement

    with pytest.raises(error_type, match=match):
        datasets.preprocessing.normalization.data_processor_from_state(state)


def test_zero_variance_normalizer_uses_a_positive_denominator_floor() -> None:
    """
    Restore zero standard deviations and normalize constant BCHW tensors.

    Results must remain finite and zero through a positive denominator floor,
    protecting legitimate constant channels without falsifying saved statistics.
    """
    state = _valid_normalizer_state()
    state["in_normalizer.std"][0, 0] = 0.0
    state["out_normalizer.std"].zero_()

    processor = datasets.preprocessing.normalization.data_processor_from_state(state)
    in_normalizer = processor.in_normalizer
    out_normalizer = processor.out_normalizer
    assert in_normalizer is not None
    assert out_normalizer is not None
    inputs = torch.zeros(2, 2, 3, 4)
    outputs = torch.zeros(2, 3, 3, 4)
    normalized_inputs = in_normalizer.transform(inputs)
    normalized_outputs = out_normalizer.transform(outputs)

    assert in_normalizer.eps > 0.0
    assert out_normalizer.eps > 0.0
    assert in_normalizer.std[0, 0, 0, 0] == 0.0
    assert torch.isfinite(normalized_inputs).all()
    assert torch.isfinite(normalized_outputs).all()
    assert torch.equal(normalized_inputs, torch.zeros_like(normalized_inputs))
    assert torch.equal(normalized_outputs, torch.zeros_like(normalized_outputs))


def test_normalizer_artifact_binds_exact_dataset_and_train_membership(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reject raw or stale normalizers whose training data identity changed."""
    train_path = _save_dataset(tmp_path, training_dataset_payload_factory("normalizer_train"))
    ood_path = _save_dataset(tmp_path, training_dataset_payload_factory("normalizer_ood"))
    _train_loader, _test_loaders, processor, split_info = datasets.runtime.training.create_dataloaders(
        path_train=str(train_path),
        path_test_ood=str(ood_path),
        task=steady_task,
        train_dataset_id="normalizer_train",
        ood_dataset_id="normalizer_ood",
        batch_size=2,
        train_ratio=0.5,
        ood_fraction=0.5,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        split_seed=17,
    )
    split_contract = datasets.preprocessing.splits.admit_split_contract(split_info)
    artifact = datasets.preprocessing.normalization.build_normalizer_artifact(
        processor,
        task=steady_task,
        split_contract=split_contract,
    )
    restored = datasets.preprocessing.normalization.validate_normalizer_artifact(
        artifact,
        task=steady_task,
        split_contract=split_contract,
    )

    assert set(restored) == set(_valid_normalizer_state())
    assert artifact["dataset_id"] == "normalizer_train"
    assert artifact["train_sample_count"] == split_info["train_indices"].numel()
    for key in restored:
        assert torch.equal(restored[key], artifact["state"][key])
        assert restored[key].data_ptr() != artifact["state"][key].data_ptr()

    with pytest.raises(ValueError, match="artifact keys"):
        datasets.preprocessing.normalization.validate_normalizer_artifact(
            _valid_normalizer_state(),
            task=steady_task,
            split_contract=split_contract,
        )

    for key, replacement in (
        ("dataset_id", "replacement_dataset"),
        ("fingerprint", "f" * 64),
    ):
        stale_identity = replace(split_contract.role("train").source, **{key: replacement})
        stale_evidence = replace(split_contract.role("train"), source=stale_identity)
        stale_contract = replace(split_contract, train=stale_evidence)
        with pytest.raises(ValueError, match="does not match"):
            datasets.preprocessing.normalization.validate_normalizer_artifact(
                artifact,
                task=steady_task,
                split_contract=stale_contract,
            )

    stale_evidence = replace(split_contract.role("train"), membership_digest="e" * 64)
    stale_contract = replace(split_contract, train=stale_evidence)
    with pytest.raises(ValueError, match="does not match"):
        datasets.preprocessing.normalization.validate_normalizer_artifact(
            artifact,
            task=steady_task,
            split_contract=stale_contract,
        )


def test_training_loader_retains_a_partial_batch(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Request a batch larger than the small valid training split.

    The loader must retain one partial batch. Dropping it would turn a valid
    dataset into an empty training lifecycle.
    """
    train_path = _save_dataset(tmp_path, training_dataset_payload_factory("partial_train"))
    ood_path = _save_dataset(tmp_path, training_dataset_payload_factory("partial_ood"))

    train_loader, *_rest = datasets.runtime.training.create_dataloaders(
        path_train=str(train_path),
        path_test_ood=str(ood_path),
        task=steady_task,
        train_dataset_id="partial_train",
        ood_dataset_id="partial_ood",
        batch_size=8,
        train_ratio=0.5,
        ood_fraction=0.5,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        split_seed=13,
    )

    batch = next(iter(train_loader))
    assert 0 < batch["x"].shape[0] < train_loader.batch_size
    assert len(train_loader) == 1


def test_operational_provenance_is_excluded_from_scientific_fingerprint(
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Changing paths/timestamps/exact operational hashes must not change science."""
    first = training_dataset_payload_factory(
        source_provenance={
            "batch_manifest_sha256": "2" * 64,
            "diagnostic_path": r"C:\\Users\\first\\OneDrive\\batch",
            "conversion_timestamp": "2026-01-01T00:00:00Z",
        },
    )
    second = training_dataset_payload_factory(
        source_provenance={
            "batch_manifest_sha256": "3" * 64,
            "diagnostic_path": "/cluster/other/batch",
            "conversion_timestamp": "2030-02-03T04:05:06Z",
        },
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    datasets.contracts.identity.validate_training_dataset_payload(second, task=domain.tasks.registry.get_task("steady_flow"), verify_content=True)


def test_generated_source_identity_ignores_locator_and_receipt_provenance() -> None:
    """Bind template and HDF5 bytes while excluding paths and success receipts."""
    cases = [
        {
            "case_id": "case_0001",
            "material_family": "lentil",
            "case_input_id": "1" * 64,
            "simulation_case_id": "2" * 64,
            "case_hdf5_sha256": "3" * 64,
            "success_sha256": "4" * 64,
            "provenance_sha256": "5" * 64,
        }
    ]

    def build(*, path: str, template_sha256: str = "6" * 64, source_cases: list[dict[str, Any]] = cases) -> dict[str, Any]:
        return datasets.contracts.identity.build_generated_package_identity(
            dataset_name="generated-source",
            simulation_profile="steady_flow",
            campaign_digest="7" * 64,
            template={"relative_path": path, "sha256": template_sha256},
            export_contract_sha256="8" * 64,
            available_learning_views=("steady_flow",),
            airflow_source="comsol_steady_reference",
            cases=source_cases,
        )

    original = build(path="templates/original.mph")
    relocated = build(path="relocated/reference.mph")
    assert relocated["batch_identity"] == original["batch_identity"]
    assert relocated["batch_manifest_identity_sha256"] == original["batch_manifest_identity_sha256"]

    receipt_only_cases = copy.deepcopy(cases)
    receipt_only_cases[0]["success_sha256"] = "9" * 64
    receipt_only_cases[0]["provenance_sha256"] = "a" * 64
    receipt_only = build(path="templates/original.mph", source_cases=receipt_only_cases)
    assert receipt_only["batch_manifest_identity_sha256"] == original["batch_manifest_identity_sha256"]

    changed_template = build(path="templates/original.mph", template_sha256="b" * 64)
    assert changed_template["batch_identity"] != original["batch_identity"]
    changed_hdf5_cases = copy.deepcopy(cases)
    changed_hdf5_cases[0]["case_hdf5_sha256"] = "c" * 64
    changed_hdf5 = build(path="templates/original.mph", source_cases=changed_hdf5_cases)
    assert changed_hdf5["batch_identity"] != original["batch_identity"]
