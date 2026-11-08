"""Protect deterministic, content-bound dataset and saved-membership identity.

Responsibilities:
  - Check content fingerprints, ordered membership, saved splits, additional Evaluation parts, normalizer state, and partial batches

Design principles:
  - Operational provenance cannot replace scientific content identity

This module does NOT:
  - Load production-sized tensors or test direct-builder transactions
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support.synthetic_task import build_synthetic_selection_provenance, save_steady_package

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
    Dataset source selection follows the reordered exact membership.
    """
    provenance = copy.deepcopy(payload["source_provenance"])
    provenance["source_case_identities"] = [provenance["source_case_identities"][index] for index in order]
    return datasets.contracts.identity.build_training_dataset_payload(
        task=task,
        dataset_id=payload["dataset_id"],
        sample_ids=[payload["sample_ids"][index] for index in order],
        source_selection_identity={
            "schema_kind": "dataset_source_selection",
            "schema_version": 1,
            "source_cases": provenance["source_case_identities"],
        },
        source_identities=[payload["source_identities"][index] for index in order],
        source_metadata=[payload["source_metadata"][index] for index in order],
        source_provenance=provenance,
        case_fingerprints=[payload["case_fingerprints"][index] for index in order],
        inputs=payload["inputs"][order],
        outputs=payload["outputs"][order],
    )


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
        source_selection_identity=original["source_selection_identity"],
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
    missing_provenance = copy.deepcopy(original["source_provenance"])
    missing_provenance["source_case_identities"] = missing_provenance["source_case_identities"][:-1]
    missing = datasets.contracts.identity.build_training_dataset_payload(
        task=steady_task,
        dataset_id="tiny",
        sample_ids=original["sample_ids"][:-1],
        source_selection_identity={
            "schema_kind": "dataset_source_selection",
            "schema_version": 1,
            "source_cases": missing_provenance["source_case_identities"],
        },
        source_identities=original["source_identities"][:-1],
        source_metadata=original["source_metadata"][:-1],
        source_provenance=missing_provenance,
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


def _publish(root: Path, payload: dict[str, Any]) -> Path:
    """Publish one synthetic steady package with its manifest below ``02_datasets``."""
    return save_steady_package(root / "02_datasets", payload)


def _loaders(root: Path, task: domain.tasks.spec.TaskSpec, train: str, additional: tuple[str, ...], **kwargs: Any) -> tuple[Any, ...]:
    """Build steady loaders from published Dataset partitions."""
    return datasets.runtime.training.create_dataloaders(
        train_dataset_id=train,
        additional_evaluation_dataset_ids=additional,
        task=task,
        model_sha256="a" * 64,
        storage_root=root,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        **kwargs,
    )


def test_two_additional_evaluation_datasets_keep_their_own_identity_membership_and_assignment(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Record each additional Evaluation Dataset as its own split part and evaluate their declared cases in order."""
    _publish(tmp_path, training_dataset_payload_factory("combined_train", material_role="id"))
    _publish(tmp_path, training_dataset_payload_factory("parameter_ood", material_family="chickpea", material_role="ood"))
    _publish(tmp_path, training_dataset_payload_factory("family_ood", material_family="kidney_bean", material_role="ood"))
    _train_loader, loaders, _processor, split_info, _scaling = _loaders(
        tmp_path,
        steady_task,
        "combined_train",
        ("parameter_ood", "family_ood"),
        batch_size=2,
    )

    split = datasets.preprocessing.splits.admit_split(split_info)
    assert split.train.dataset_id == "combined_train"
    assert [part.dataset_id for part in split.additional_evaluation] == ["parameter_ood", "family_ood"]
    assert [assignment["dataset_id"] for assignment in split.evaluation_assignments] == ["parameter_ood", "family_ood"]
    assert len({part.dataset["dataset_digest"] for part in split.additional_evaluation}) == 2
    assert len({part.membership_digest for part in split.additional_evaluation}) == 2
    assert all(part.count == part.full_count for part in split.additional_evaluation)
    assert split.as_payload() == split_info
    evaluated = loaders["additional_evaluation"]
    assert evaluated is not None
    assert sum(len(batch["y"]) for batch in evaluated) == _COMBINED_OOD_SAMPLE_COUNT


def test_family_holdout_must_not_reuse_a_training_material(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reject an evaluation Dataset whose declared holdout repeats a training material."""
    _publish(tmp_path, training_dataset_payload_factory("holdout_train", material_role="id"))
    _publish(tmp_path, training_dataset_payload_factory("same_family_ood", material_role="ood"))
    with pytest.raises(ValueError, match="reuses a training material"):
        _loaders(tmp_path, steady_task, "holdout_train", ("same_family_ood",), batch_size=2)


def test_steady_training_requires_a_dataset_partition(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Refuse a role-neutral steady Dataset instead of drawing a run-seeded split."""
    _publish(tmp_path, training_dataset_payload_factory("unpartitioned_train"))
    with pytest.raises(ValueError, match="no fixed case partition"):
        _loaders(tmp_path, steady_task, "unpartitioned_train", (), batch_size=2)


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
    train_path = _publish(tmp_path, training_dataset_payload_factory("train", material_role="id"))
    _publish(tmp_path, training_dataset_payload_factory("ood", material_family="chickpea", material_role="ood"))
    *_, split_info, _scaling = _loaders(tmp_path, steady_task, "train", ("ood",), batch_size=1)
    assert split_info["task_contract_digest"] == steady_task.contract_digest
    stale_header = copy.deepcopy(split_info)
    stale_header["task_contract_digest"] = "8cdaf4de22d945e08783f118d5fa8374e37521f91b20b12c913230ba015ca91a"
    with pytest.raises(ValueError, match="current registered task"):
        datasets.preprocessing.splits.admit_split(stale_header)

    replaced = training_dataset_payload_factory(
        "train",
        source_tokens=("same", "same", "replacement", "same"),
        material_role="id",
    )
    train_path.unlink()
    save_steady_package(tmp_path / "02_datasets", replaced)
    with pytest.raises(ValueError, match="Saved split does not match the current Dataset manifests"):
        _loaders(tmp_path, steady_task, "train", ("ood",), batch_size=1, split_indices=split_info)


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
    _publish(tmp_path, training_dataset_payload_factory("partial_train", material_role="id"))

    train_loader, *_rest = _loaders(tmp_path, steady_task, "partial_train", (), batch_size=8)

    batch = next(iter(train_loader))
    assert 0 < batch["x"].shape[0] < train_loader.batch_size
    assert len(train_loader) == 1


def test_operational_provenance_is_excluded_from_scientific_fingerprint(
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Changing paths/timestamps/exact operational hashes must not change science."""
    base_provenance = training_dataset_payload_factory()["source_provenance"]
    first_provenance = copy.deepcopy(base_provenance)
    first_provenance.update(diagnostic_path=r"C:\\Users\\first\\OneDrive\\dataset", conversion_timestamp="2026-01-01T00:00:00Z")
    second_provenance = copy.deepcopy(base_provenance)
    second_provenance.update(diagnostic_path="/cluster/other/dataset", conversion_timestamp="2030-02-03T04:05:06Z")
    first = training_dataset_payload_factory(source_provenance=first_provenance)
    second = training_dataset_payload_factory(source_provenance=second_provenance)

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    datasets.contracts.identity.validate_training_dataset_payload(second, task=domain.tasks.registry.get_task("steady_flow"), verify_content=True)


def test_dataset_selection_identity_binds_exact_ordered_source_evidence() -> None:
    """Change exact source evidence and order to obtain distinct Dataset digests."""
    original = build_synthetic_selection_provenance(dataset_name="selected-source", sample_ids=("case_0001", "case_0002"))
    original_id, original_digest = datasets.contracts.identity.package_identity_from_provenance(original)
    assert datasets.contracts.identity.package_identity_from_provenance(original) == (original_id, original_digest)

    for changed in (
        {"case_hdf5_relative": "relocated/case.h5"},
        {"case_hdf5_sha256": "c" * 64},
        {"decision_sha256": "d" * 64},
    ):
        candidate = copy.deepcopy(original)
        candidate["source_case_identities"][0]["source"].update(changed)
        assert datasets.contracts.identity.package_identity_from_provenance(candidate)[1] != original_digest

    reordered = copy.deepcopy(original)
    reordered["source_case_identities"].reverse()
    assert datasets.contracts.identity.package_identity_from_provenance(reordered)[1] != original_digest
