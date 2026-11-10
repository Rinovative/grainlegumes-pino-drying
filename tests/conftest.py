"""Provide reusable strict task and dataset fixtures with isolated storage roots.

Responsibilities:
  - Provide registered/synthetic TaskSpec fixtures and compact fingerprinted Dataset payloads
  - Register the shared Generation and transient-package fixture plugins for every test directory

Design principles:
  - Autouse storage isolation protects production roots unless real-data access is explicitly
    selected

This module does NOT:
  - Supply benchmark data or test scientific equations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from support import real_data
from support.synthetic_task import build_synthetic_selection_provenance, build_synthetic_task

from src import datasets, domain

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

pytest_plugins = ("support.generation_fixtures", "support.source_tree", "support.transient_package")


@pytest.fixture(scope="session", autouse=True)
def isolate_session_storage_root(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Fence module- and session-scoped fixtures away from production data roots."""
    if real_data.real_data_tests_enabled():
        yield
        return
    root = tmp_path_factory.mktemp("application-data-roots")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    try:
        yield
    finally:
        monkeypatch.undo()


@pytest.fixture(autouse=True)
def isolate_application_storage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Keep generic tests' implicit data and coordination paths out of production roots."""
    if request.node.get_closest_marker("real_data") is not None:
        return
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))


@pytest.fixture
def steady_task() -> domain.tasks.spec.TaskSpec:
    """
    Return the registered immutable steady-flow contract used by production.

    This fixture is appropriate when a test must exercise the exact seven-input,
    three-output fields. Genericity tests should request ``synthetic_task`` instead.
    """
    return domain.tasks.registry.get_task("steady_flow")


@pytest.fixture
def synthetic_task() -> domain.tasks.spec.TaskSpec:
    """
    Return an unregistered task with different fields, units, and no physics.

    The fixture detects steady-flow constants leaking into generic consumers and
    must never be used as a production task or persisted default.
    """
    return build_synthetic_task()


@pytest.fixture
def training_dataset_payload_factory(
    steady_task: domain.tasks.spec.TaskSpec,
) -> Callable[..., dict[str, Any]]:
    """Return a factory for small version-1 final training datasets."""

    def factory(
        dataset_id: str = "tiny",
        *,
        sample_ids: Sequence[str] = ("case_0000", "case_0001", "case_0002", "case_0003"),
        dtype: torch.dtype = torch.float32,
        source_tokens: Sequence[str] | None = None,
        source_provenance: dict[str, Any] | None = None,
        material_family: str = "lentil",
        material_role: str | None = None,
    ) -> dict[str, Any]:
        tokens = tuple(source_tokens or sample_ids)
        inputs = torch.stack(
            [
                torch.stack([torch.full((2, 3), float(sample_index + channel), dtype=dtype) for channel in range(steady_task.in_channels)])
                for sample_index, _sample_id in enumerate(sample_ids)
            ]
        )
        outputs = torch.stack(
            [
                torch.stack([torch.full((2, 3), float(sample_index + 20 + channel), dtype=dtype) for channel in range(steady_task.out_channels)])
                for sample_index, _sample_id in enumerate(sample_ids)
            ]
        )
        identities = [{"case_id": sample_id, "token": tokens[index]} for index, sample_id in enumerate(sample_ids)]
        metadata = [{"case_id": sample_id, "parameters": {"sample_index": index}} for index, sample_id in enumerate(sample_ids)]
        fingerprints = [
            datasets.contracts.identity.compute_case_fingerprint(
                task=steady_task,
                case_id=sample_id,
                source_identity=identities[index],
                source_metadata=metadata[index],
                inputs=inputs[index],
                outputs=outputs[index],
            )
            for index, sample_id in enumerate(sample_ids)
        ]
        provenance = build_synthetic_selection_provenance(
            dataset_name=dataset_id,
            sample_ids=sample_ids,
            material_family=material_family,
            material_role=material_role,
        )
        return datasets.contracts.identity.build_training_dataset_payload(
            task=steady_task,
            dataset_id=dataset_id,
            sample_ids=sample_ids,
            source_selection_identity={
                "schema_kind": "dataset_source_selection",
                "source_cases": provenance["source_case_identities"],
            },
            source_identities=identities,
            source_metadata=metadata,
            source_provenance=source_provenance or provenance,
            case_fingerprints=fingerprints,
            inputs=inputs,
            outputs=outputs,
        )

    return factory
