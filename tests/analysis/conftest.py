"""Share compact Generation-input diagnostics and Dataset catalogs across analysis tests.

Responsibilities:
  - Generate and admit two cases per profile and three input Datasets once per session
  - Hand each test the shared frozen evidence under the session project root

Design principles:
  - Input generation runs once; diagnostics and catalogs are frozen dataclasses with read-only
    mappings, so tests share them without copying

This module does NOT:
  - Run a solver, publish Sources, or touch production storage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from support.generation_fixtures import lentil_generation_config, make_generation_config_factory

from src import generation
from src.analysis import generation_inputs

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

type ProfileRecords = dict[
    str, tuple[generation_inputs.diagnostics.GenerationInputDiagnostics, generation_inputs.diagnostics.GenerationInputDiagnostics]
]


def _profile_records(factory: Callable[..., tuple[Path, Path]], root: Path) -> ProfileRecords:
    """Generate and admit two compact canonical cases for each profile."""
    result = {}
    for profile_id in ("steady_flow", "transient_drying"):
        config = lentil_generation_config(factory, profile_id)
        records = []
        for case_index in (1, 2):
            bundle = generation.cases.case.generate_case_input_bundle(config, case_index, root / profile_id / f"case_{case_index:04d}")
            admitted = generation.cases.admission.admit_input_case(
                bundle.directory,
                source_id=f"input-{profile_id}",
                label="technical label is not a selector contract",
                batch_storage_name=config.batch_storage_name,
                campaign_purpose=str(config.scientific_values["campaign_purpose"]),
            )
            records.append(generation_inputs.diagnostics.build_case_diagnostics(admitted))
        result[profile_id] = (records[0], records[1])
    return result


def _dataset_catalog(factory: Callable[..., tuple[Path, Path]], storage: Path) -> generation_inputs.sources.GenerationInputDatasetCatalog:
    """Publish compact steady and two compatible transient input Datasets."""
    for profile_id, natural_count, cases in (("steady_flow", 4, 3), ("transient_drying", 4, 3), ("transient_drying", 5, 2)):
        config = lentil_generation_config(factory, profile_id, storage_root=storage, natural_count=natural_count)
        generation.cases.input_generation.generate_input_cases(config, cases, storage_root=storage)
    return generation_inputs.sources.discover_generation_input_datasets(storage)


@pytest.fixture(scope="session")
def _generation_input_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ProfileRecords, Any]:
    """Build the input evidence once under the environment the per-test fixtures provide."""
    root = tmp_path_factory.mktemp("generation-inputs")
    project_root = root / "project"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
        monkeypatch.setenv("STORAGE_ROOT", str(root / "storage"))
        monkeypatch.setenv("PROJECT_ROOT", str(project_root))
        factory = make_generation_config_factory(project_root)
        return project_root, _profile_records(factory, root / "records"), _dataset_catalog(factory, root / "catalog-storage")


@pytest.fixture
def profile_records(_generation_input_template: tuple[Path, ProfileRecords, Any], monkeypatch: pytest.MonkeyPatch) -> ProfileRecords:
    """Return the shared admitted per-profile case diagnostics."""
    project_root, records, _catalog = _generation_input_template
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    return records


@pytest.fixture
def dataset_catalog(
    _generation_input_template: tuple[Path, ProfileRecords, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> generation_inputs.sources.GenerationInputDatasetCatalog:
    """Return the shared discovered input Dataset catalog."""
    project_root, _records, catalog = _generation_input_template
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    return catalog
