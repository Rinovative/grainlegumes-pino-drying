"""Protect how a Dataset takes its cases, splits them and reads its transient samples.

Responsibilities:
  - Check that membership follows Generation's design order whatever order the catalog lists the rows in
  - Check that design order admits exactly one Source and names exact selection for several
  - Check that a design-order request admits only source cases of its declared Source, whoever reads it
  - Check that the fixed split depends on the split seed and that ordered membership alone
  - Check that a transient Dataset is read from its PT shards only, while the source HDF5 reader stays an explicit opt-in

Design principles:
  - A test-owned catalog and the compact published transient package replace Generation and publication
  - No selection seed exists, so nothing but the Source and the split seed can change a Dataset

This module does NOT:
  - Publish Datasets or exercise the Dataset plan, which have their own tests
  - Train models or judge the scientific quality of a split
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

import pytest

from src import datasets
from src.datasets.contracts import dataset_contracts_partition as partition
from src.datasets.packages import dataset_packages_selection as selection
from src.datasets.runtime.dataset_runtime_transient import TransientPhysicalDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

_ORIGINAL_POSITIONS = (1, 2, 3, 4, 5)
_FAILED_POSITION = 2
_REPLACEMENT_POSITION = 6
_RECIPE = {"kind": "seeded_material_fraction", "train_fraction": 0.6, "validation_fraction": 0.2, "id_test_fraction": 0.2}


def _row(position: int, *, origin: str, outcome: str = "successful", campaign: str = "source_a") -> dict[str, Any]:
    """Return one catalog row of the fake Source with the exact reference keys."""
    return {
        "campaign_name": campaign,
        "campaign_digest": "a" * 64,
        "material_family": "lentil",
        "case_id": f"case_{position:04d}",
        "case_input_id": f"input_{position}",
        "simulation_case_id": f"simulation_{position}",
        "case_hdf5_sha256": "b" * 64,
        "origin": origin,
        "candidate_position": position,
        "outcome": outcome,
    }


def _source_rows() -> list[dict[str, Any]]:
    """Return one failed original, its replacement and the other four originals."""
    outcomes = {position: "failed" if position == _FAILED_POSITION else "successful" for position in _ORIGINAL_POSITIONS}
    return [
        *(_row(position, origin="primary", outcome=outcome) for position, outcome in outcomes.items()),
        _row(_REPLACEMENT_POSITION, origin="replacement"),
    ]


def _select(rows: Sequence[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Return the selected case IDs when the catalog lists ``rows`` in the given order."""
    monkeypatch.setattr(selection.catalog, "list_campaign_cases", lambda *_args, **_kwargs: tuple(rows))
    picks = selection.select_ordered_cases("steady_flow", ["source_a"], {"lentil": 5})
    return [pick["case_id"] for pick in picks]


def test_membership_follows_design_order_not_catalog_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Originals by case index, then replacements by ordinal, and a pool that is too small is refused."""
    rows = _source_rows()
    expected = ["case_0001", "case_0003", "case_0004", "case_0005", "case_0006"]

    assert _select(rows, monkeypatch) == expected
    assert _select(rows[::-1], monkeypatch) == expected
    assert _select([*rows[3:], *rows[:3]], monkeypatch) == expected

    monkeypatch.setattr(selection.catalog, "list_campaign_cases", lambda *_args, **_kwargs: tuple(rows[:-1]))
    with pytest.raises(ValueError, match="lacks 5 successful lentil"):
        selection.select_ordered_cases("steady_flow", ["source_a"], {"lentil": 5})


def test_design_order_admits_exactly_one_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Several Sources are refused where the order is resolved and where it is declared; one Source still selects."""
    assert len(_select(_source_rows(), monkeypatch)) == len(_ORIGINAL_POSITIONS)

    with pytest.raises(ValueError, match=r"exactly one Source.*exact selection"):
        selection.select_ordered_cases("steady_flow", ["source_a", "source_b"], {"lentil": 5})
    declared = {"kind": "design_order", "source_names": ["source_a", "source_b"], "target_counts": {"lentil": 5}}
    with pytest.raises(ValueError, match=r"exactly one Source.*exact selection"):
        selection.materialize_selection_request({"source_task": "steady_flow", "selection": declared, "source_cases": []})


def test_design_order_request_admits_only_cases_of_its_declared_source() -> None:
    """Admission compares the campaigns the cases reference with the declared Source, also for a revised Source label."""

    def request(source: str, campaigns: Sequence[str]) -> dict[str, Any]:
        rows = [_row(position, origin="primary", campaign=campaign) for position, campaign in enumerate(campaigns, start=1)]
        return {
            "schema_kind": selection.SELECTION_SCHEMA_KIND,
            "schema_version": selection.SELECTION_SCHEMA_VERSION,
            "source_task": "steady_flow",
            "dataset_name": "ordered",
            "dataset_view": "steady_flow",
            "selection": {"kind": "design_order", "source_names": [source], "target_counts": {"lentil": len(rows)}},
            "source_cases": [{key: row[key] for key in selection.SOURCE_REFERENCE_ORDER} for row in rows],
        }

    for source, campaigns in (("source_a", ["source_a"] * 2), ("source_a__rev-fix", ["source_a"] * 2)):
        assert selection.validate_selection_request(request(source, campaigns))["selection"]["source_names"] == [source]
    for source, campaigns in (("source_a", ["source_a", "source_b"]), ("source_a", ["source_b"] * 2), ("source_a__rev-fix", ["source_b"] * 2)):
        with pytest.raises(ValueError, match="other than the declared Source"):
            selection.validate_selection_request(request(source, campaigns))


def test_split_depends_on_split_seed_and_membership_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same ordered membership and split seed give one split; another split seed gives another."""
    rows = _source_rows()
    material_by_case = dict.fromkeys(_select(rows, monkeypatch), "lentil")

    def split(case_ids: Sequence[str], seed: int) -> dict[str, Any]:
        return partition.build_fixed_partition(
            case_ids, material_by_case=material_by_case, material_roles={"lentil": "id"}, recipe=_RECIPE, seed_input=seed, seed=seed
        )

    first = split(_select(rows, monkeypatch), 9)
    assert split(_select(rows[::-1], monkeypatch), 9) == first
    assert len({tuple(split(_select(rows, monkeypatch), seed)["train"]) for seed in range(8)}) > 1


def test_transient_dataset_reads_pt_shards_only(transient_package: dict[str, Any]) -> None:
    """The training view reads PT shards, the HDF5 source reader is requested explicitly, and a folder without shards is refused."""
    root, dataset_id = transient_package["root"], transient_package["train_dataset"]
    request = datasets.runtime.factory.DatasetRequest(
        dataset_id=dataset_id,
        dataset_view="transient_drying",
        transient_sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        storage_root=root,
    )
    shards = datasets.runtime.factory.create_dataset(request)
    assert isinstance(shards, TransientPhysicalDataset)
    assert shards.storage_backend == "pt_shards"
    shards.close()
    source = datasets.runtime.factory.create_hdf5_source_dataset(request)
    assert source.storage_backend == "canonical_hdf5"
    source.close()

    shutil.rmtree(datasets.packages.transient_shards.transient_shard_directory(dataset_id, storage_root=root))

    # The manifest binds the shard receipt, so an incomplete publication is refused for every reader.
    for open_view in (datasets.runtime.factory.create_dataset, datasets.runtime.factory.create_hdf5_source_dataset):
        with pytest.raises(FileNotFoundError, match=dataset_id):
            open_view(request)
