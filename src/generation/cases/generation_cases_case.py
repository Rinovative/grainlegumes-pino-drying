"""
===============================================================================
generation_cases_case.py
===============================================================================
Create deterministic scientific inputs and isolated COMSOL work directories.
Responsibilities:
  - Resolve one typed sample into spatial, scalar, and schedule adapter tables
  - Derive exact adapter-input and profile-bound simulation identities
  - Copy one immutable template into a fresh disposable work directory
Design principles:
  - Adapter bytes are deterministic, case-local, and identity-bound
  - Every stochastic stream and block row is explicit in case provenance
  - Source templates are read and copied but never opened as output targets
This module does NOT:
  - Run COMSOL, infer template mappings, or publish canonical HDF5 results
  - Treat CSV as a canonical learning-view artifact
===============================================================================
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src import common
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_paths as path_contract
from src.generation.contracts import generation_contracts_porosity as porosity_contract
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.contracts import generation_contracts_source as source_service

from . import generation_cases_config as config_contract
from . import generation_cases_fields as fields_service
from . import generation_cases_sampling as sampling_service
from . import generation_cases_schedule as schedule_service
from . import generation_cases_seeding as seeding

CASE_SCHEMA_KIND = "simulation_case"
CASE_SCHEMA_VERSION = 1
_TABLE_RANK = 2


_CASE_PAYLOAD_REQUIRED_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "scientific_config_digest",
        "case_input_config_digest",
        "case_id",
        "case_index",
        "generator_version",
        "git_commit",
        "material_family",
        "material_role",
        "evaluation_regime",
        "sampling_regime",
        "natural_support_state",
        "ood",
        "available_learning_views",
        "airflow_source",
        "steady_flow_conditioning",
        "steady_flow_conditioning_digest",
        "stationary_fixed_ownership",
        "stationary_fixed_values",
        "seed_evidence",
        "block_provenance",
        "sampled_values",
        "sampled_units",
        "coupled_selections",
        "spatial_diagnostics",
        "input_contract",
        "template",
        "export_contract_sha256",
        "input_files",
        "case_input_id",
        "simulation_case_id",
    }
)

_CASE_PAYLOAD_OPTIONAL_KEYS: Final = frozenset({"schedule_diagnostics", "pilot_check", "scalar_handoff", "scalars"})
CASE_CONTRACT_DIGEST: Final = common.serialization.canonical_json_sha256(
    {
        "schema_kind": CASE_SCHEMA_KIND,
        "schema_version": CASE_SCHEMA_VERSION,
        "generator_version": seeding.GENERATOR_VERSION,
        "required_case_fields": sorted(_CASE_PAYLOAD_REQUIRED_KEYS),
        "optional_case_fields": sorted(_CASE_PAYLOAD_OPTIONAL_KEYS),
        "transient_only_case_fields": ["scalar_handoff", "scalars", "schedule_diagnostics"],
        "porosity_diagnostic_fields": sorted(fields_service.POROSITY_DIAGNOSTIC_KEYS),
        "packing_scatter_truncation": [
            porosity_contract.PACKING_SCATTER_TRUNCATION_LOWER,
            porosity_contract.PACKING_SCATTER_TRUNCATION_UPPER,
        ],
    }
)


def validate_case_payload_schema(payload: dict[str, Any]) -> None:
    """Validate the exact active case.json schema and reject stale fields."""
    if not isinstance(payload, dict):
        msg = "case.json payload must be a mapping."
        raise TypeError(msg)
    missing, unknown = (
        sorted(_CASE_PAYLOAD_REQUIRED_KEYS.difference(payload)),
        sorted(set(payload).difference(_CASE_PAYLOAD_REQUIRED_KEYS | _CASE_PAYLOAD_OPTIONAL_KEYS)),
    )
    if missing or unknown:
        msg = f"case.json schema is invalid: missing={missing}, unknown={unknown}."
        raise ValueError(msg)
    if payload["schema_kind"] != CASE_SCHEMA_KIND or payload["schema_version"] != CASE_SCHEMA_VERSION:
        msg = "case.json schema kind or version is invalid."
        raise ValueError(msg)
    profile = payload["simulation_profile"]
    profiles.resolve_profile(profile)
    transient = profile == profiles.TRANSIENT_DRYING_PROFILE
    scalar_keys = {"scalar_handoff", "scalars"}
    scalar_fields_match = scalar_keys.issubset(payload) if transient else scalar_keys.isdisjoint(payload)
    if not scalar_fields_match:
        msg = "case.json scalar-handoff fields do not match its simulation profile."
        raise ValueError(msg)
    if transient != ("schedule_diagnostics" in payload):
        msg = "case.json schedule diagnostics do not match its simulation profile."
        raise ValueError(msg)
    fields_service.validate_porosity_diagnostics(payload["spatial_diagnostics"]["porosity"])


def compute_case_input_id(payload: dict[str, Any]) -> str:
    """Compute the exact generated-input identity from persisted evidence."""
    try:
        identity_payload = {
            "schema_version": payload["schema_version"],
            "case_input_config_digest": payload["case_input_config_digest"],
            "material_family": payload["material_family"],
            "sampling_regime": payload["sampling_regime"],
            "ood": payload["ood"],
            "case_index": payload["case_index"],
            "seed_evidence": payload["seed_evidence"],
            "block_provenance": payload["block_provenance"],
            "sampled_values": payload["sampled_values"],
            "coupled_selections": payload["coupled_selections"],
            "input_files": payload["input_files"],
        }
    except (KeyError, TypeError) as error:
        msg = "Case-input identity payload is incomplete or malformed."
        raise ValueError(msg) from error
    return common.serialization.canonical_json_sha256(identity_payload)


def compute_simulation_case_id(payload: dict[str, Any]) -> str:
    """Compute the profile-, template-, and export-bound simulation identity."""
    try:
        identity_payload = {
            "schema_version": payload["schema_version"],
            "case_input_id": payload["case_input_id"],
            "simulation_profile": payload["simulation_profile"],
            "template_sha256": payload["template"]["sha256"],
            "export_contract_sha256": payload["export_contract_sha256"],
            "steady_flow_conditioning_digest": payload["steady_flow_conditioning_digest"],
        }
    except (KeyError, TypeError) as error:
        msg = "Simulation-case identity payload is incomplete or malformed."
        raise ValueError(msg) from error
    return common.serialization.canonical_json_sha256(identity_payload)


@dataclass(frozen=True, slots=True)
class CaseBundle:
    """One generated case input bundle and its two canonical identities."""

    directory: Path
    case_id: str
    case_input_id: str
    simulation_case_id: str
    case_payload: dict[str, Any]
    input_paths: tuple[Path, ...]
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None


def _format_number(value: float) -> str:
    """Format one finite number with the shared scalar round-trip contract."""
    return scalar_handoff_contract.format_scalar_number(value)


def _table_text(rows: list[list[str]], *, delimiter: str) -> str:
    """Serialize one stable table with LF line endings."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return stream.getvalue()


def _write_spatial_file(destination: Path, fields: fields_service.SpatialFields, spec: dict[str, Any]) -> Path:
    """Write the exact spatial adapter in deterministic column-major order."""
    columns = list(spec["columns"])
    missing = [name for name in columns if name not in fields.columns]
    if missing:
        msg = f"Spatial adapter requests unknown generated fields: {missing}."
        raise ValueError(msg)
    flattened = [fields.columns[name].ravel(order="F") for name in columns]
    rows = [columns]
    rows.extend([_format_number(float(value)) for value in values] for values in zip(*flattened, strict=True))
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path


def _write_scalar_file(
    destination: Path,
    spec: dict[str, Any],
    values: dict[str, Any],
    units: dict[str, str],
) -> tuple[Path, tuple[scalar_handoff_contract.ScalarHandoffEntry, ...]]:
    """Write the sole profile-owned long-form scalar handoff."""
    entries = scalar_handoff_contract.build_transient_scalar_entries(values, units)
    rows = [["name", "value", "unit"]]
    rows.extend([[entry.name, scalar_handoff_contract.format_scalar_number(entry.value), entry.unit] for entry in entries])
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path, entries


def _write_schedule_file(destination: Path, spec: dict[str, Any], schedule: schedule_service.ComsolBoundarySchedule) -> Path:
    """Write the exact four-column COMSOL boundary interpolation table."""
    columns = list(spec["columns"])
    if tuple(columns) != profiles.SCHEDULE_FIELDS or schedule.values.ndim != _TABLE_RANK or schedule.values.shape[1] != len(columns):
        msg = "Generated schedule does not satisfy the configured adapter field contract."
        raise ValueError(msg)
    rows = [columns]
    rows.extend([_format_number(float(value)) for value in row] for row in schedule.values)
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path


def _require_empty_destination(
    destination: Path,
    *,
    allow_workspace_marker: bool,
) -> None:
    """Create one new bundle destination or require only its ownership marker."""
    if destination.exists() and not destination.is_dir():
        msg = f"Case bundle destination is not a directory: {destination}"
        raise FileExistsError(msg)
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {path_contract.CASE_WORKSPACE_MARKER} if allow_workspace_marker else set()
    actual = {entry.name for entry in destination.iterdir()}
    if actual != allowed:
        msg = f"Case bundle destination must contain exactly {sorted(allowed)} before generation: {destination}"
        raise FileExistsError(msg)


def _subseeds(
    config: config_contract.GenerationConfig,
    case_index: int,
) -> dict[str, int]:
    """Return profile substreams, sharing paired-smoke airflow only."""
    case_seed = config.case_seed(case_index)
    labels = ["bed", "pressure_bc", "packing_scatter"]
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        labels.extend(
            (
                "initial_moisture",
                "schedule_shared",
                "schedule_independent",
            )
        )
    result = {
        label: seeding.derive_seed(
            case_seed,
            "case_substream",
            label,
        )
        for label in labels
    }
    paired_seed = config.scientific_values.get("paired_equivalence_seed")
    if paired_seed is not None:
        assignment = config.case_assignment(case_index)
        paired_case_seed = seeding.derive_seed(
            int(paired_seed),
            "paired_equivalence_case",
            config.material_family,
            config.sampling_regime,
            str(assignment["assignment_role"]),
            str(case_index),
        )
        for label in ("bed", "pressure_bc", "packing_scatter"):
            result[label] = seeding.derive_seed(
                paired_case_seed,
                "paired_airflow_substream",
                label,
            )
    return result


def generate_case_input_bundle(
    config: config_contract.GenerationConfig,
    case_index: int,
    destination: Path | str,
    *,
    _allow_workspace_marker: bool = False,
) -> CaseBundle:
    """Generate one deterministic profile-specific scientific input bundle."""
    case_id = config.case_id(case_index)
    case_seed = config.case_seed(case_index)
    assignment = config.case_assignment(case_index)
    bundle_dir = Path(destination).expanduser().resolve()
    _require_empty_destination(
        bundle_dir,
        allow_workspace_marker=_allow_workspace_marker,
    )
    sample = sampling_service.sample_case(config, case_index)
    values = dict(sample.values)
    units = dict(sample.units)
    fixed = config.scientific_values["scientific_fixed_values"]
    stationary_fixed_entries: list[dict[str, Any]] = []
    for name, unit in zip(
        profiles.STATIONARY_FIXED_FIELDS,
        profiles.STATIONARY_FIXED_UNITS,
        strict=True,
    ):
        value = fixed[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            msg = f"Package-fixed stationary scalar {name!r} is unresolved."
            raise ValueError(msg)
        number = float(value)
        if config.profile.id == profiles.STEADY_FLOW_PROFILE:
            values[name] = number
            units[name] = unit
        stationary_fixed_entries.append(
            {
                "name": name,
                "value": number,
                "unit": unit,
                "owner": "package_fixed",
                "runtime_source": "canonical_template",
            }
        )
    subseeds = _subseeds(config, case_index)

    schedule: schedule_service.Schedule | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        schedule = schedule_service.generate_schedule(
            values,
            config.scientific_values["time"],
            fixed,
            seeds={name: subseeds[name] for name in ("schedule_shared", "schedule_independent")},
        )

    family_contract = config.scientific_values["material"]
    spatial_seed_names: tuple[str, ...] = ("bed", "pressure_bc", "packing_scatter")
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        spatial_seed_names = (*spatial_seed_names, "initial_moisture")
    initial_moisture_bounds = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        initial_moisture_bounds = materials.initial_moisture_generation_bounds(
            family_contract,
            values,
            active_ood_unit=sample.ood_provenance["active_unit_id"],
        )
    fields = fields_service.generate_spatial_fields(
        config.profile.id,
        config.scientific_values["grid"],
        values,
        seeds={name: subseeds[name] for name in spatial_seed_names},
        family_bounds=initial_moisture_bounds,
        porosity_coupling=family_contract["porosity_coupling"],
        active_ood_unit=sample.ood_provenance["active_unit_id"],
    )
    complete_case_retry = fields.metadata["complete_case_support_retry"]
    support_attempt = int(complete_case_retry["acceptance_attempt"])
    if schedule is not None and support_attempt > 1:
        schedule_seed_names = ("schedule_shared", "schedule_independent")
        schedule = schedule_service.generate_schedule(
            values,
            config.scientific_values["time"],
            fixed,
            seeds={
                name: seeding.derive_seed(
                    subseeds[name],
                    "complete_case_support_retry",
                    str(support_attempt),
                    name,
                )
                for name in schedule_seed_names
            },
        )
    boundary_schedule: schedule_service.ComsolBoundarySchedule | None = None
    if schedule is not None:
        boundary_schedule = schedule_service.build_comsol_boundary_schedule(
            schedule,
            config.scientific_values["boundary_schedule"]["startup_ramp"],
            initial_temperature=float(values["T_init"]),
            pressure=float(fixed["p_ref"]),
        )

    input_contract = config.scientific_values["input_contract"]
    spatial_path = _write_spatial_file(bundle_dir, fields, input_contract["spatial"])
    input_paths_list = [spatial_path]
    scalar_path: Path | None = None
    scalar_entries: tuple[scalar_handoff_contract.ScalarHandoffEntry, ...] | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        scalar_path, scalar_entries = _write_scalar_file(
            bundle_dir,
            input_contract["scalar"],
            values,
            units,
        )
        input_paths_list.append(scalar_path)
    if boundary_schedule is not None:
        input_paths_list.append(_write_schedule_file(bundle_dir, input_contract["schedule"], boundary_schedule))
    input_paths = tuple(sorted(input_paths_list, key=lambda item: item.name))
    input_files = {
        path.name: {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in input_paths
    }
    export_contract_sha256 = common.serialization.canonical_json_sha256(config.scientific_values["output_contract"])
    steady_flow_conditioning = config.scientific_values["steady_flow_conditioning"]
    steady_flow_conditioning_digest = common.serialization.canonical_json_sha256(steady_flow_conditioning)
    paired_seed = config.scientific_values.get("paired_equivalence_seed")
    seed_evidence = {
        "campaign_seed": config.scientific_values["campaign_seed"],
        "batch_seed": config.seed_base,
        "case_seed": case_seed,
        "case_seed_labels": {
            "simulation_profile": config.profile.id,
            "material_family": config.material_family,
            "sampling_regime": config.sampling_regime,
            "assignment_role": assignment["assignment_role"],
            "case_ordinal": case_index,
        },
        "paired_equivalence_seed": paired_seed,
        "subseed_origins": {
            name: ("paired_equivalence" if paired_seed is not None and name in {"bed", "pressure_bc", "packing_scatter"} else "profile_case")
            for name in subseeds
        },
        "derivation": ("sha256(generator_version|seed_namespace|semantic_labels)"),
        "subseeds": subseeds,
    }
    case_payload: dict[str, Any] = {
        "schema_kind": CASE_SCHEMA_KIND,
        "schema_version": CASE_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "scientific_config_digest": config.scientific_config_digest,
        "case_input_config_digest": config.case_input_config_digest,
        "case_id": case_id,
        "case_index": case_index,
        "generator_version": seeding.GENERATOR_VERSION,
        "git_commit": source_service.required_git_commit(),
        "material_family": assignment["material_family"],
        "material_role": config.material_role,
        "evaluation_regime": config.evaluation_regime,
        "sampling_regime": assignment["sampling_regime"],
        "natural_support_state": sample.ood_provenance["natural_support_state"],
        "ood": sample.ood_provenance,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "steady_flow_conditioning": steady_flow_conditioning,
        "steady_flow_conditioning_digest": steady_flow_conditioning_digest,
        "stationary_fixed_ownership": config.scientific_values["stationary_fixed_ownership"],
        "stationary_fixed_values": stationary_fixed_entries,
        "seed_evidence": seed_evidence,
        "block_provenance": sample.block_provenance,
        "sampled_values": values,
        "sampled_units": units,
        "coupled_selections": sample.coupled_selections,
        "spatial_diagnostics": fields.metadata,
        "input_contract": input_contract,
        "template": {
            "relative_path": config.template_relative_path,
            "filename": config.template_path.name,
            "sha256": config.template_sha256,
        },
        "export_contract_sha256": export_contract_sha256,
        "input_files": input_files,
    }
    pilot_case_kind = assignment.get("pilot_case_kind")
    if pilot_case_kind is not None:
        if pilot_case_kind not in config_contract.PILOT_CASE_KINDS:
            message = f"Unsupported pilot case kind {pilot_case_kind!r}."
            raise ValueError(message)
        case_payload["pilot_check"] = {
            "purpose": config_contract.PILOT_CAMPAIGN_PURPOSE,
            "case_kind": pilot_case_kind,
            "dataset_membership": "none",
        }
    if scalar_path is not None and scalar_entries is not None:
        scalar_entries_payload = [entry.as_dict() for entry in scalar_entries]
        case_payload["scalar_handoff"] = {
            "mechanism": "case_local_long_form_csv",
            "filename": scalar_path.name,
            "fresh_per_case": True,
            "runtime_validation": "required",
            "entries": scalar_entries_payload,
        }
        case_payload["scalars"] = scalar_entries_payload
    if boundary_schedule is not None:
        case_payload["schedule_diagnostics"] = boundary_schedule.metadata
    scalar_admission = (
        None
        if scalar_path is None
        else scalar_handoff_contract.admit_case_scalar_handoff(
            case_payload,
            bundle_dir,
        )
    )
    case_payload["case_input_id"] = compute_case_input_id(case_payload)
    case_payload["simulation_case_id"] = compute_simulation_case_id(case_payload)
    validate_case_payload_schema(case_payload)
    common.serialization.atomic_write_json(bundle_dir / "case.json", case_payload)
    return CaseBundle(
        directory=bundle_dir,
        case_id=case_id,
        case_input_id=case_payload["case_input_id"],
        simulation_case_id=case_payload["simulation_case_id"],
        case_payload=case_payload,
        input_paths=input_paths,
        scalar_handoff=scalar_admission,
    )
