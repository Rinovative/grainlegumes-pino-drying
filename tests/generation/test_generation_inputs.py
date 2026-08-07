# ruff: noqa: S101, PLR2004
"""Scientific generation, profile, and generic COMSOL-adapter contracts."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from src import generation

if TYPE_CHECKING:
    from pathlib import Path


def test_profiles_select_templates_views_and_provenance(generation_config_factory: Any) -> None:
    """Protect exact profile registration, template binding, and unknown-profile rejection."""
    assert generation.profiles.available_profiles() == ("steady_flow", "transient_drying")
    expected = {
        "steady_flow": (("steady_flow",), "comsol_steady_reference", "template_brinkman.mph"),
        "transient_drying": (
            ("steady_flow", "transient_drying"),
            "comsol_coupled_reference",
            "template_brinkman_temp_moist.mph",
        ),
    }
    for profile_id, (views, airflow_source, filename) in expected.items():
        config_path, _template = generation_config_factory(simulation_profile=profile_id)
        config = generation.config.load_generation_config(config_path)
        assert config.profile.id == profile_id
        assert config.profile.available_learning_views == views
        assert config.profile.airflow_source == airflow_source
        assert config.template_path.name == filename
        assert config.template_sha256 == generation.profiles.get_profile(profile_id).template_sha256

    config_path, _template = generation_config_factory()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["simulation_profile"] = "unknown"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown simulation_profile"):
        generation.config.load_generation_config(config_path)


def test_spatial_and_generic_inputs_are_deterministic(generation_config_factory: Any, tmp_path: Path) -> None:
    """Protect deterministic ordering, tensors, porosity, scalars, schedules, and identities."""
    config_path, _template = generation_config_factory()
    config = generation.config.load_generation_config(config_path)
    first = generation.case.generate_case_input_bundle(config, 1, tmp_path / "first")
    second = generation.case.generate_case_input_bundle(config, 1, tmp_path / "second")

    assert config.batch_identity == generation.config.load_generation_config(config_path).batch_identity
    assert first.case_identity == second.case_identity
    for filename in ("case_0001.csv", "scalar_values.csv", "schedule.csv", "case.json"):
        assert (first.directory / filename).read_bytes() == (second.directory / filename).read_bytes()

    with (first.directory / "case_0001.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    assert rows[0] == ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"]
    numeric = np.asarray(rows[1:], dtype=np.float64)
    assert np.all(numeric[:4, 0] == 0.0)
    assert np.all(np.diff(numeric[:4, 1]) > 0.0)
    assert np.all(numeric[:, 2] * numeric[:, 4] - numeric[:, 3] ** 2 > 0.0)
    assert np.all((numeric[:, 5] >= 0.3) & (numeric[:, 5] <= 0.8))
    assert first.case_payload["available_learning_views"] == ["steady_flow", "transient_drying"]
    assert first.case_payload["airflow_source"] == "comsol_coupled_reference"

    sampled_path, _sampled_template = generation_config_factory(case_indices=[1, 2, 3])
    sampled_raw = yaml.safe_load(sampled_path.read_text(encoding="utf-8"))
    sampled_raw["sampling"] = {
        "method": "lhs",
        "variation": 0.4,
        "parameters": [
            {"target": "generator", "name": "ms_weight", "index": 0, "transform": "softmax", "group": "weights"},
            {"target": "generator", "name": "ms_weight", "index": 1, "transform": "softmax", "group": "weights"},
        ],
    }
    sampled_path.write_text(yaml.safe_dump(sampled_raw, sort_keys=False), encoding="utf-8")
    sampled_config = generation.config.load_generation_config(sampled_path)
    sampled = generation.sampling.sample_case_overrides(sampled_config)
    assert sampled == generation.sampling.sample_case_overrides(sampled_config)
    assert all(np.isclose(sum(case["generator"]["ms_weight"]), 1.0) for case in sampled.values())


def test_profile_and_adapter_errors_fail_at_preflight(generation_config_factory: Any, tmp_path: Path) -> None:
    """Protect optional inputs, exact profile adapters, schedules, and owned runtime controls."""
    empty_path, _template = generation_config_factory(scalar_entries=[])
    empty_config = generation.config.load_generation_config(empty_path)
    bundle = generation.case.generate_case_input_bundle(empty_config, 1, tmp_path / "empty")
    assert not (bundle.directory / "scalar_values.csv").exists()

    raw = yaml.safe_load(empty_path.read_text(encoding="utf-8"))
    raw["inputs"]["schedule_file"]["rows"] = [[0.0, 1.0], [0.0, 2.0]]
    invalid_schedule = tmp_path / "invalid_schedule.yaml"
    invalid_schedule.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        generation.config.load_generation_config(invalid_schedule)

    raw["inputs"]["schedule_file"]["rows"] = [[0.0, 1.0], [1.0, 2.0]]
    raw["inputs"]["spatial_files"][0]["filename"] = "other.csv"
    invalid_path = tmp_path / "invalid_path.yaml"
    invalid_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="requires one spatial adapter"):
        generation.config.load_generation_config(invalid_path)

    steady_path, _template = generation_config_factory(simulation_profile="steady_flow")
    steady = yaml.safe_load(steady_path.read_text(encoding="utf-8"))
    steady["inputs"]["schedule_file"] = {
        "filename": "schedule.csv",
        "delimiter": ";",
        "columns": ["time", "control"],
        "rows": [[0.0, 1.0], [1.0, 2.0]],
    }
    steady_path.write_text(yaml.safe_dump(steady, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not accept a schedule"):
        generation.config.load_generation_config(steady_path)

    owned_runtime = yaml.safe_load(empty_path.read_text(encoding="utf-8"))
    owned_runtime["execution"]["extra_arguments"] = ["-nn", "2"]
    owned_runtime_path = tmp_path / "invalid_owned_runtime.yaml"
    owned_runtime_path.write_text(yaml.safe_dump(owned_runtime, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="one-node execution"):
        generation.config.load_generation_config(owned_runtime_path)


def test_maintained_steady_flow_infrastructure_config_preflights() -> None:
    """Protect the maintained non-scientific steady-flow infrastructure smoke."""
    config = generation.config.load_generation_config("configs/generation/steady_flow/infrastructure_smoke.yaml")
    assert config.profile.id == "steady_flow"
    assert config.profile.available_learning_views == ("steady_flow",)
    assert config.profile.airflow_source == "comsol_steady_reference"
    assert config.values["cluster"]["cores_per_node"] == 32
