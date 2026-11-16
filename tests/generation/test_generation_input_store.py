"""Stored case-input container, canonical CSV rendering, and identity contracts.

Responsibilities:
  - Pin the CPU-independent config digests and scalar CSV of one steady and one transient fixture case, and check the
    CPU-dependent CSVs against a frozen copy of the CSV-storing writer
  - Check that the writer stores only the input container, whose rendered CSV bytes reproduce the recorded identities
  - Check that equivalent containers written independently give the same identities
  - Check that corrupt or malformed containers, rendered mismatches, and CSV-only case inputs are refused
  - Check that raw publication evidence refuses a container changed after admission

Design principles:
  - Identity comes from the rendered canonical CSV bytes, never from HDF5 serialization bytes
  - Corruptions are applied through h5py to small test-owned bundles

This module does NOT:
  - Launch COMSOL or test scientific sampling semantics
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from support.generation_fixtures import rewrite_input_store_equivalently

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.runtime import generation_runtime_batch as runtime_batch

if TYPE_CHECKING:
    from pathlib import Path

case_service = generation.cases.case

# Generated maps and schedules differ in the last digits between ICE CPU types (numpy SIMD loops, OpenBLAS kernels), so the
# identity pins below are only the CPU-independent ones: the config digests and the transient scalars.csv, both recorded with
# the CSV-storing writer that preceded the input store. The CPU-dependent CSVs are checked against a frozen copy of that writer.
_SCALARS_CSV = {"sha256": "840afa3510f5a8eddafa103002240d58a34972b1bd79629e51e0f74ff4dda4d6", "size_bytes": 388}
# (case_input_config_digest, scientific_config_digest) per profile.
_CONFIG_DIGESTS: dict[str, tuple[str, str]] = {
    "steady_flow": (
        "6eb659f0f162b583bbceb7ba9d4b1456a4a7d9936e534028268b24a36d6284a4",
        "624fdd0b6c312837a622ff984327e866080fa35e99344c660bd1c304893e0b9f",
    ),
    "transient_drying": (
        "51f615e958547e5329698d838a1b76f4da31ea0579c844be9c12cf6c12c89858",
        "60a630fc4d2919c8d0d7483aeb1cabacf4d6ebfef701e0d5dd6bb2ed547fcbdc",
    ),
}
_REGENERATE = r"no regular inputs\.h5.*regenerate"


def _config(generation_config_factory: Any, profile_id: str, *, natural_count: int = 4) -> Any:
    """Load one compact synthetic lentil batch."""
    config_path, _template = generation_config_factory(simulation_profile=profile_id, natural_count=natural_count)
    return generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(profile_id, "lentil", "test"),
    )


def _identities(rendered: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    """Return sha256 and size of each rendered CSV."""
    return {name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)} for name, data in rendered.items()}


def _legacy_case_ids(payload: dict[str, Any]) -> tuple[str, str]:
    """Project case_input_id and simulation_case_id like the code that preceded the input store (frozen reference copy)."""
    digest = common.serialization.canonical_json_sha256
    input_keys = (
        "case_input_config_digest",
        "material_family",
        "sampling_regime",
        "ood",
        "case_index",
        "seed_evidence",
        "block_provenance",
        "sampled_values",
        "coupled_selections",
        "input_files",
    )
    case_input_id = digest({key: payload[key] for key in input_keys})
    simulation_case_id = digest(
        {
            "case_input_id": case_input_id,
            "simulation_profile": payload["simulation_profile"],
            "template_sha256": payload["template"]["sha256"],
            "export_contract_sha256": payload["export_contract_sha256"],
            "steady_flow_conditioning_digest": payload["steady_flow_conditioning_digest"],
        }
    )
    return case_input_id, simulation_case_id


def _legacy_csv_bytes(content: Any, contract: dict[str, Any]) -> dict[str, bytes]:
    """Serialize generated inputs exactly like the CSV-storing writer that preceded the input store (frozen reference copy)."""
    number = generation.contracts.scalar_handoff.format_scalar_number

    def table(rows: list[list[str]], delimiter: str) -> bytes:
        stream = io.StringIO(newline="")
        csv.writer(stream, delimiter=delimiter, lineterminator="\n", quoting=csv.QUOTE_MINIMAL).writerows(rows)
        return stream.getvalue().encode("utf-8")

    spatial = contract["spatial"]
    columns = list(spatial["columns"])
    flattened = [content.fields[name].ravel(order="F") for name in columns]
    files = {
        spatial["filename"]: table(
            [columns, *([number(float(value)) for value in values] for values in zip(*flattened, strict=True))], spatial["delimiter"]
        )
    }
    if content.schedule is not None:
        scalar, schedule = contract["scalar"], contract["schedule"]
        files[scalar["filename"]] = table(
            [["name", "value", "unit"], *([name, number(value), unit] for name, value, unit in content.scalars)], scalar["delimiter"]
        )
        files[schedule["filename"]] = table(
            [list(schedule["columns"]), *([number(float(value)) for value in row] for row in content.schedule)], schedule["delimiter"]
        )
    return files


@pytest.mark.parametrize("profile_id", ["steady_flow", "transient_drying"])
def test_case_identities_equal_the_csv_writer_contract(generation_config_factory: Any, tmp_path: Path, profile_id: str) -> None:
    """Keep config digests and COMSOL CSV bytes identical to the CSV-storing writer, on every CPU, while storing only the container."""
    config = _config(generation_config_factory, profile_id)
    with patch.object(case_service, "_write_case_inputs", wraps=case_service._write_case_inputs) as writer:
        bundle = case_service.generate_case_input_bundle(config, 1, tmp_path / "case")
    payload = bundle.case_payload
    generated = SimpleNamespace(**writer.call_args.kwargs)

    assert (config.case_input_config_digest, config.scientific_config_digest) == _CONFIG_DIGESTS[profile_id]
    assert {path.name for path in bundle.directory.iterdir()} == {"case.json", "inputs.h5"}
    assert _identities(_legacy_csv_bytes(generated, payload["input_contract"])) == payload["input_files"]
    stored = case_service.read_case_inputs(bundle.directory, payload)
    assert _identities(case_service.render_input_files(stored, payload["input_contract"])) == payload["input_files"]
    assert (payload["case_input_id"], payload["simulation_case_id"]) == _legacy_case_ids(payload)
    if profile_id == "transient_drying":
        assert payload["input_files"]["scalars.csv"] == _SCALARS_CSV


def test_independently_written_equivalent_containers_give_the_same_identity(generation_config_factory: Any, tmp_path: Path) -> None:
    """Accept another HDF5 encoding of the same content: the rendered CSV identities, not the container bytes, bind the case."""
    bundle = case_service.generate_case_input_bundle(_config(generation_config_factory, "transient_drying"), 1, tmp_path / "case")
    original_store = case_service.measure_input_store(bundle.directory)
    original = case_service.read_case_inputs(bundle.directory, bundle.case_payload, store_identity=original_store)

    rewrite_input_store_equivalently(bundle.directory / "inputs.h5")
    rewritten_store = case_service.measure_input_store(bundle.directory)
    rewritten = case_service.read_case_inputs(bundle.directory, bundle.case_payload, store_identity=rewritten_store)

    assert rewritten_store.sha256 != original_store.sha256
    contract = bundle.case_payload["input_contract"]
    assert case_service.render_input_files(rewritten, contract) == case_service.render_input_files(original, contract)
    np.testing.assert_array_equal(rewritten.fields["X_0_db_field"], original.fields["X_0_db_field"])


def _corrupt(handle: h5py.File, corruption: str) -> None:
    """Apply one named corruption to a stored steady container."""
    maps = handle["fields"]
    assert isinstance(maps, h5py.Group)
    dataset = maps["Kxx"]
    assert isinstance(dataset, h5py.Dataset)
    values = np.asarray(dataset[()])
    if corruption == "non_finite":
        dataset[0, 0] = np.nan
    elif corruption == "tampered_value":
        dataset[0, 0] = 2.0 * values[0, 0]
    elif corruption != "wrong_hash":
        del maps["Kxx"]
        if corruption == "wrong_dtype":
            maps.create_dataset("Kxx", data=values.astype(np.float32))
        elif corruption == "wrong_shape":
            maps.create_dataset("Kxx", data=values[:, :-1])


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("wrong_hash", "recorded storage integrity"),
        ("missing_dataset", "recorded column order"),
        ("wrong_dtype", "must be float64 datasets"),
        ("wrong_shape", "one shared two-dimensional shape"),
        ("non_finite", "must be finite"),
        ("tampered_value", "do not render the canonical input identities"),
    ],
)
def test_corrupt_or_malformed_container_is_refused(generation_config_factory: Any, tmp_path: Path, corruption: str, match: str) -> None:
    """Refuse a container whose integrity, inventory, dtype, shape, finiteness, or rendered identity is wrong."""
    bundle = case_service.generate_case_input_bundle(_config(generation_config_factory, "steady_flow"), 1, tmp_path / "case")
    recorded = case_service.measure_input_store(bundle.directory)
    with h5py.File(bundle.directory / "inputs.h5", "r+") as handle:
        _corrupt(handle, corruption)
    store_identity = case_service.InputStoreIdentity(sha256="0" * 64, size_bytes=recorded.size_bytes) if corruption == "wrong_hash" else None

    with pytest.raises(ValueError, match=match):
        case_service.read_case_inputs(bundle.directory, bundle.case_payload, store_identity=store_identity)


def test_materialization_renders_identity_verified_csvs_and_refuses_a_mismatch(generation_config_factory: Any, tmp_path: Path) -> None:
    """Render every COMSOL CSV into the work directory, and write none when a rendering disagrees with case.json."""
    bundle = case_service.generate_case_input_bundle(_config(generation_config_factory, "transient_drying"), 1, tmp_path / "case")
    store_identity = case_service.measure_input_store(bundle.directory)
    work = tmp_path / "work"
    work.mkdir()

    paths = case_service.materialize_case_inputs(bundle.directory, work, bundle.case_payload, store_identity=store_identity)

    assert [path.name for path in paths] == sorted(bundle.case_payload["input_files"])
    assert _identities({path.name: path.read_bytes() for path in paths}) == bundle.case_payload["input_files"]
    payload = copy.deepcopy(bundle.case_payload)
    payload["input_files"]["schedule.csv"]["sha256"] = "0" * 64
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(ValueError, match="do not render the canonical input identities"):
        case_service.materialize_case_inputs(bundle.directory, refused, payload, store_identity=store_identity)
    assert list(refused.iterdir()) == []


def test_csv_only_case_inputs_are_refused_with_the_regeneration_instruction(generation_config_factory: Any, tmp_path: Path) -> None:
    """Refuse an input generation and a case directory that store CSVs instead of inputs.h5, and convert nothing."""
    config = _config(generation_config_factory, "steady_flow", natural_count=1)
    storage = tmp_path / "storage"
    lifecycle.create_campaign(generation.cases.config.load_campaign_config(config.source_path), storage_root=storage)
    batch = generation.cases.input_generation.generate_input_cases(config, 1, storage_root=storage)
    reference = generation.cases.input_generation.admit_configured_input_references(config, storage_root=storage)[1]
    payload = reference.case_payload()
    stored = case_service.read_case_inputs(reference.case_directory, payload)
    for filename, data in case_service.render_input_files(stored, payload["input_contract"]).items():
        (reference.case_directory / filename).write_bytes(data)
    (reference.case_directory / "inputs.h5").unlink()
    manifest_path = batch.metadata_directory / "input_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["cases"]:
        del record["input_store"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=_REGENERATE):
        generation.cases.admission.admit_input_batch_source(batch.metadata_directory, raw_directory=batch.raw_directory, validation_depth="evidence")
    with pytest.raises(FileNotFoundError, match=_REGENERATE):
        generation.cases.admission.admit_input_case(reference.case_directory, source_id="csv-only", label="csv-only case")
    assert not (reference.case_directory / "inputs.h5").exists()


def test_raw_publication_refuses_a_container_changed_after_admission(generation_config_factory: Any, tmp_path: Path) -> None:
    """Re-check the stored container against its admitted reference before it becomes raw publication evidence."""
    config = _config(generation_config_factory, "steady_flow", natural_count=1)
    storage = tmp_path / "storage"
    lifecycle.create_campaign(generation.cases.config.load_campaign_config(config.source_path), storage_root=storage)
    generation.cases.input_generation.generate_input_cases(config, 1, storage_root=storage)
    reference = generation.cases.input_generation.admit_configured_input_references(config, storage_root=storage)[1]
    evidence = runtime_batch._raw_publication_from_reference(reference)
    assert [artifact.relative_path for artifact in evidence.artifacts] == ["case.json", "inputs.h5"]

    rewrite_input_store_equivalently(reference.case_directory / "inputs.h5")

    with pytest.raises(RuntimeError, match="changed after admission"):
        runtime_batch._raw_publication_from_reference(reference)
