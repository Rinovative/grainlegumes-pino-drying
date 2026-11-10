"""Protect separate sampled-schedule and actual-output horizons.

Responsibilities:
  - Check unchanged case inputs, explicit horizon provenance, and short Source publication

Design principles:
  - A bounded output prefix retains the full sampled boundary schedule and actual Dataset horizon

This module does NOT:
  - Run licensed COMSOL or establish full-duration physical feasibility
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
import yaml

from src import generation
from src.analysis.generation_inputs import generation_input_diagnostics as input_diagnostics
from src.datasets.packages import dataset_packages_trajectory as trajectory
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_schedule as schedule_service

if TYPE_CHECKING:
    from pathlib import Path


def _short_horizon(path: Path, *, stop: float = 3.0) -> None:
    common_path = path.parent / "common.yaml"
    value = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    value["schedule_time"] = copy.deepcopy(value["time"])
    value["schedule_time_provenance"] = copy.deepcopy(value["time_provenance"])
    value["time"]["stop"] = stop
    common_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_short_output_preserves_full_schedule_and_sampled_case_inputs(generation_config_factory: Any, tmp_path: Path) -> None:
    path, _template = generation_config_factory(simulation_profile="transient_drying", natural_count=1, grid_overrides={"nx": 9, "ny": 6})
    original = config_service.load_campaign_config(path).batches[0]
    _short_horizon(path)
    bounded = config_service.load_campaign_config(path).batches[0]
    assert "schedule_time" not in original.scientific_values
    assert original.scientific_values["time"] == bounded.scientific_values["schedule_time"]
    assert bounded.scientific_values["time"]["regular_times"] == [0.0, 1.0, 2.0, 3.0]
    assert original.case_seed(1) == bounded.case_seed(1)
    assert original.scientific_config_digest != bounded.scientific_config_digest
    assert original.case_input_config_digest != bounded.case_input_config_digest
    original_case = generation.cases.case.generate_case_input_bundle(original, 1, tmp_path / "original")
    bounded_case = generation.cases.case.generate_case_input_bundle(bounded, 1, tmp_path / "bounded")
    assert original_case.case_payload["sampled_values"] == bounded_case.case_payload["sampled_values"]
    original_diagnostics = copy.deepcopy(original_case.case_payload["schedule_diagnostics"])
    bounded_diagnostics = copy.deepcopy(bounded_case.case_payload["schedule_diagnostics"])
    original_handoff = original_diagnostics.pop("boundary_handoff")
    bounded_handoff = bounded_diagnostics.pop("boundary_handoff")
    assert original_diagnostics == bounded_diagnostics
    assert bounded_handoff.pop("regular_output_grid") == {"start_h": 0.0, "stop_h": 3.0, "interval_h": 1.0, "node_count": 4}
    assert bounded_handoff.pop("regular_output_time_policy") == schedule_service.SEPARATE_OUTPUT_TIME_POLICY
    original_handoff.pop("regular_output_time_policy")
    assert original_handoff == bounded_handoff
    admitted = generation.cases.admission.admit_input_case(bounded_case.directory, source_id="bounded-input", label="bounded input")
    record = input_diagnostics.build_case_diagnostics(admitted)
    assert record.canonical_schedule is not None
    assert record.regular_output_schedule is not None
    assert record.canonical_schedule.shape[0] == 169
    np.testing.assert_array_equal(record.regular_output_schedule[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert {p.name: p.read_bytes() for p in original_case.input_paths} == {p.name: p.read_bytes() for p in bounded_case.input_paths}


@pytest.mark.parametrize("invalid", ["missing_provenance", "insufficient_coverage", "different_interval", "steady_profile"])
def test_separate_schedule_contract_rejects_ambiguous_or_insufficient_support(generation_config_factory: Any, invalid: str) -> None:
    profile = "steady_flow" if invalid == "steady_profile" else "transient_drying"
    path, _template = generation_config_factory(simulation_profile=profile, natural_count=1)
    _short_horizon(path)
    common_path = path.parent / "common.yaml"
    value = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    if invalid == "missing_provenance":
        del value["schedule_time_provenance"]
    elif invalid == "insufficient_coverage":
        value["schedule_time"]["stop"] = 2.0
    elif invalid == "different_interval":
        value["schedule_time"]["interval"] = 2.0
    common_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(config_service.GenerationConfigError, match="schedule_time"):
        config_service.load_campaign_config(path)


def test_short_source_publication_retains_full_schedule_and_actual_dataset_horizon(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        natural_count=1,
        grid_overrides={"nx": 9, "ny": 6},
        retain_raw_csv=True,
        retain_solved_model=True,
    )
    _short_horizon(path, stop=4.0)
    campaign = config_service.load_campaign_config(path)
    batch = campaign.batches[0]
    storage = tmp_path / "storage"
    lifecycle.create_campaign(campaign, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage)
    monkeypatch.setenv("FAKE_COMSOL_TRANSIENT_TIME_MODE", "configured_horizon")
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", generation.campaign.campaign_run_id(campaign, git_commit="a" * 40))
    outcome = generation.runtime.run_case(batch, 1, cores_per_case=1, storage_root=storage, work_root=tmp_path / "work", scheduler_kind="local")
    assert outcome.status == "completed", outcome
    generation.runtime.validate_completed_case(batch, 1, storage_root=storage, validation_depth="routine")
    artifact = outcome.processed_directory / "case.h5"
    generation.publication.storage.validate_case_hdf5(artifact)
    with h5py.File(artifact, "r+") as handle:
        scientific_dataset = handle["provenance/scientific_config_json"]
        assert isinstance(scientific_dataset, h5py.Dataset)
        scientific_text = scientific_dataset.asstr()[()]
        assert isinstance(scientific_text, str)
        scientific = json.loads(scientific_text)
        assert scientific["time"]["stop"] == 4.0
        assert scientific["schedule_time"]["stop"] == 168.0
        schedule_dataset = handle["schedule/values"]
        assert isinstance(schedule_dataset, h5py.Dataset)
        temporal_text = schedule_dataset.attrs["temporal_provenance"]
        assert isinstance(temporal_text, str)
        temporal_provenance = json.loads(temporal_text)
        assert set(temporal_provenance) == {"output_time", "schedule_time"}
        handoff_text = schedule_dataset.attrs["boundary_handoff"]
        assert isinstance(handoff_text, str)
        handoff = json.loads(handoff_text)
        assert handoff["regular_output_time_policy"] == schedule_service.SEPARATE_OUTPUT_TIME_POLICY
        assert handoff["regular_output_grid"] == {"start_h": 0.0, "stop_h": 4.0, "interval_h": 1.0, "node_count": 5}
        canonical_rows, output_rows, _enabled, _duration = input_diagnostics._canonical_schedule(np.asarray(schedule_dataset), handoff)
        assert canonical_rows.shape[0] == 169
        np.testing.assert_array_equal(output_rows[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
        sampled_times = np.asarray(scientific["schedule_time"]["regular_times"])
        inconsistent_handoff = copy.deepcopy(handoff)
        inconsistent_handoff["regular_output_grid"]["stop_h"] = 5.0
        with pytest.raises(ValueError, match="exact prefix"):
            schedule_service.regular_output_times(inconsistent_handoff, sampled_times)
        inconsistent_handoff = copy.deepcopy(handoff)
        inconsistent_handoff["regular_output_time_policy"] = "common.time.regular_times_unchanged"
        with pytest.raises(ValueError, match="explicit regular output grid"):
            schedule_service.regular_output_times(inconsistent_handoff, sampled_times)
        np.testing.assert_array_equal(np.asarray(handle["time"]), [0.0, 1.0, 2.0, 3.0, 4.0])
        assert schedule_dataset[-1, 0] == 168.0
        layout = trajectory.resolve_transient_case_layout(handle, artifact, sample_id_prefix="physical_case")
        assert layout.configured_horizon == 4.0
        assert len(layout.samples) == 4
        assert layout.samples[0]["schedule_support_index"] is not None
        assert layout.samples[-1]["time_index_n_plus_1"] == 4
        final_schedule_row = np.asarray(schedule_dataset[-1], dtype=np.float64).copy()
        final_schedule_row[0] = 167.0
        schedule_dataset[-1] = final_schedule_row
        with pytest.raises(ValueError, match="complete configured schedule_time"):
            generation.publication.storage._validate_hdf5_schedule(handle, scientific)
