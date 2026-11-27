"""Create deterministic scientific inputs, store them in one HDF5 container, and render their COMSOL CSVs.

Responsibilities:
  - Resolve one typed sample into spatial, scalar, and schedule inputs stored atomically in one HDF5 input container
  - Read the stored container through one fail-closed reader: inventory, dtype, shape, finite values, and coordinate order
  - Render the canonical COMSOL CSV bytes and derive exact adapter-input and profile-bound simulation identities from them
  - Materialize identity-verified CSVs into a COMSOL work directory and measure the container's storage integrity

Design principles:
  - Rendered adapter bytes are deterministic, case-local, and identity-bound; the container's own bytes are storage integrity only
  - One formatter and one renderer serve generation, admission, and runtime materialization
  - Column order comes from the case input contract and is recorded in the file, never taken from HDF5 member order
  - Every stochastic stream and block row is explicit in case provenance

This module does NOT:
  - Run COMSOL, infer template mappings, or publish canonical HDF5 results
  - Store CSV files, read CSV-only case directories, or treat CSV as a canonical learning-view artifact

Notes:
  Input container layout (format version 1, no timestamps):
    - Root attribute ``format_version``.
    - ``fields/``: one float64 ``(ny, nx)`` dataset per spatial column, ordered by the group attribute ``columns``;
      ``ravel(order="F")`` reproduces the CSV row order.
    - ``scalars`` (transient): compound table ``name, value, unit`` in handoff order.
    - ``schedule`` (transient): compound table with one float64 member per schedule column, in row order.
    - VS Code with the H5Web extension opens the container without code.

"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np

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

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CASE_SCHEMA_KIND = "generation_source_case"
INPUT_STORE_FORMAT_VERSION: Final = 1
_FORMAT_ATTRIBUTE: Final = "format_version"
_COLUMNS_ATTRIBUTE: Final = "columns"
_INPUT_TABLES: Final = {"spatial": "fields", "scalar": "scalars", "schedule": "schedule"}
_SCALAR_COLUMNS: Final = ("name", "value", "unit")
_TABLE_RANK = 2


_CASE_PAYLOAD_REQUIRED_KEYS: Final = frozenset(
    {
        "schema_kind",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "scientific_config_digest",
        "case_input_config_digest",
        "case_id",
        "case_index",
        "git_commit",
        "material_family",
        "domain_profile",
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

_CASE_PAYLOAD_OPTIONAL_KEYS: Final = frozenset({"schedule_diagnostics", "scalar_handoff", "scalars"})
CASE_CONTRACT_DIGEST: Final = common.serialization.canonical_json_sha256(
    {
        "schema_kind": CASE_SCHEMA_KIND,
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
    if payload["schema_kind"] != CASE_SCHEMA_KIND:
        msg = "case.json schema kind is invalid."
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
    """One generated case input bundle and its two canonical identities.

    ``scalar_handoff`` binds the transient scalar CSV that COMSOL reads; its source path is the case-local
    filename for a stored bundle and the rendered file for a prepared work directory.
    """

    directory: Path
    case_id: str
    case_input_id: str
    simulation_case_id: str
    case_payload: dict[str, Any]
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None


@dataclass(frozen=True, slots=True)
class InputStoreIdentity:
    """Storage-integrity evidence of one stored input container, outside every scientific identity."""

    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return the persisted manifest representation."""
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class StoredCaseInputs:
    """Validated stored case inputs in contract order; arrays are read-only."""

    fields: Mapping[str, np.ndarray]
    scalars: tuple[tuple[str, float, str], ...]
    schedule: np.ndarray | None


def _require(condition: bool, message: str) -> None:
    """Fail closed with one input-store contract message."""
    if not condition:
        raise ValueError(message)


def _format_number(value: float) -> str:
    """Format one finite number with the shared scalar round-trip contract."""
    return scalar_handoff_contract.format_scalar_number(value)


def _table_text(rows: list[list[str]], *, delimiter: str) -> str:
    """Serialize one stable table with LF line endings."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return stream.getvalue()


def require_input_store(directory: Path) -> Path:
    """Return the regular stored input container or refuse a directory without one, naming the regeneration fix."""
    name = common.paths.GENERATION_INPUT_STORE_FILENAME
    path = directory / name
    if path.is_symlink() or not path.is_file():
        message = f"Case inputs have no regular {name}: {directory}. Inputs stored as CSV files predate the HDF5 input store; regenerate them."
        raise FileNotFoundError(message)
    return path


def measure_input_store(directory: Path) -> InputStoreIdentity:
    """Return the sha256 and size of one directory's stored input container."""
    data = require_input_store(directory).read_bytes()
    return InputStoreIdentity(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _validate_inputs(content: StoredCaseInputs, input_contract: Mapping[str, Any]) -> None:
    """Require the contract inventory, float64 dtype, shared shape, finite values, and Cartesian F-order grid."""
    columns = list(input_contract["spatial"]["columns"])
    _require(list(content.fields) == columns and {"x", "y"}.issubset(columns), "Stored spatial inputs do not match the contract columns.")
    _require(
        len({values.shape for values in content.fields.values()}) == 1
        and all(values.dtype == np.float64 and values.ndim == _TABLE_RANK and values.size > 0 for values in content.fields.values())
        and all(np.isfinite(values).all() for values in content.fields.values()),
        "Stored spatial inputs must be finite non-empty float64 maps of one shared two-dimensional shape.",
    )
    x_values, y_values = content.fields["x"], content.fields["y"]
    ny, nx = x_values.shape
    # Physical coordinates need not ascend; x is constant down each column and y along each row, with distinct axis values.
    _require(
        bool(np.all(x_values == x_values[:1, :]))
        and bool(np.all(y_values == y_values[:, :1]))
        and np.unique(x_values[0]).size == nx
        and np.unique(y_values[:, 0]).size == ny,
        "Stored spatial coordinates are not one Cartesian grid in canonical F-order.",
    )
    transient = "scalar" in input_contract
    _require(
        ("schedule" in input_contract) == transient and bool(content.scalars) == transient and (content.schedule is None) != transient,
        "Stored scalar and schedule inputs do not match the case input contract.",
    )
    if content.schedule is None:
        return
    _require(tuple(input_contract["scalar"]["columns"]) == _SCALAR_COLUMNS, "Scalar input contract must name the columns name, value, and unit.")
    names = [name for name, _value, _unit in content.scalars]
    _require(
        all(names) and len(set(names)) == len(names) and all(math.isfinite(value) for _name, value, _unit in content.scalars),
        "Stored scalar inputs need unique non-empty names and finite values.",
    )
    schedule = content.schedule
    _require(
        schedule.dtype == np.float64
        and schedule.ndim == _TABLE_RANK
        and schedule.shape[0] > 0
        and schedule.shape[1] == len(input_contract["schedule"]["columns"])
        and bool(np.isfinite(schedule).all()),
        "Stored schedule must be a finite non-empty float64 table with one column per contract column.",
    )


def _text_dtype(values: Sequence[str]) -> np.dtype[Any]:
    """Return one fixed-length UTF-8 string type wide enough for every value."""
    return h5py.string_dtype("utf-8", max([1, *(len(value.encode("utf-8")) for value in values)]))


def _write_container(path: Path, content: StoredCaseInputs, input_contract: Mapping[str, Any]) -> None:
    """Write one complete input container to a fresh temporary path."""
    with h5py.File(path, "w") as handle:
        handle.attrs[_FORMAT_ATTRIBUTE] = INPUT_STORE_FORMAT_VERSION
        fields = handle.create_group(_INPUT_TABLES["spatial"])
        fields.attrs.create(_COLUMNS_ATTRIBUTE, list(content.fields), dtype=h5py.string_dtype())
        for name, values in content.fields.items():
            fields.create_dataset(name, data=values)
        if content.schedule is None:
            return
        scalar_type = np.dtype(
            [
                ("name", _text_dtype([name for name, _value, _unit in content.scalars])),
                ("value", np.float64),
                ("unit", _text_dtype([unit for _name, _value, unit in content.scalars])),
            ]
        )
        scalars = np.array([(name.encode(), value, unit.encode()) for name, value, unit in content.scalars], dtype=scalar_type)
        handle.create_dataset(_INPUT_TABLES["scalar"], data=scalars)
        columns = list(input_contract["schedule"]["columns"])
        schedule = np.empty(content.schedule.shape[0], dtype=np.dtype([(name, np.float64) for name in columns]))
        for index, name in enumerate(columns):
            schedule[name] = content.schedule[:, index]
        handle.create_dataset(_INPUT_TABLES["schedule"], data=schedule)


def _read_only(values: np.ndarray) -> np.ndarray:
    """Return one independent read-only float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _member(parent: h5py.Group, name: str, kind: type[h5py.Group | h5py.Dataset]) -> Any:
    """Return one named container member of the required HDF5 kind."""
    member = parent.get(name)
    _require(isinstance(member, kind), f"Stored case-input member {name!r} must be an HDF5 {kind.__name__}.")
    return member


def _read_container(handle: h5py.File, input_contract: Mapping[str, Any]) -> StoredCaseInputs:
    """Read one opened input container after checking its exact inventory and member types."""
    expected = {table for kind, table in _INPUT_TABLES.items() if kind in input_contract}
    version = handle.attrs.get(_FORMAT_ATTRIBUTE)
    _require(
        set(handle.attrs) == {_FORMAT_ATTRIBUTE} and isinstance(version, np.integer) and int(version) == INPUT_STORE_FORMAT_VERSION,
        f"Stored case inputs must carry format_version {INPUT_STORE_FORMAT_VERSION} and no other root attribute.",
    )
    _require(set(handle) == expected, f"Stored case inputs must contain exactly {sorted(expected)}.")
    group = _member(handle, _INPUT_TABLES["spatial"], h5py.Group)
    columns = list(input_contract["spatial"]["columns"])
    _require(
        set(group.attrs) == {_COLUMNS_ATTRIBUTE} and np.asarray(group.attrs[_COLUMNS_ATTRIBUTE]).tolist() == columns and set(group) == set(columns),
        "Stored spatial maps or their recorded column order do not match the contract columns.",
    )
    datasets = {name: _member(group, name, h5py.Dataset) for name in columns}
    _require(all(dataset.dtype == np.float64 for dataset in datasets.values()), "Stored spatial maps must be float64 datasets.")
    fields = {name: _read_only(dataset[()]) for name, dataset in datasets.items()}
    if _INPUT_TABLES["scalar"] not in expected:
        return StoredCaseInputs(fields=fields, scalars=(), schedule=None)
    scalar_table = _member(handle, _INPUT_TABLES["scalar"], h5py.Dataset)
    schedule_table = _member(handle, _INPUT_TABLES["schedule"], h5py.Dataset)
    schedule_columns = tuple(input_contract["schedule"]["columns"])
    _require(
        scalar_table.ndim == 1
        and scalar_table.dtype.names == _SCALAR_COLUMNS
        and scalar_table.dtype["value"] == np.float64
        and scalar_table.dtype["name"].kind == scalar_table.dtype["unit"].kind == "S",
        "Stored scalars must be one name, value, unit table with fixed-length text and float64 values.",
    )
    _require(
        schedule_table.ndim == 1
        and schedule_table.dtype.names == schedule_columns
        and all(schedule_table.dtype[name] == np.float64 for name in schedule_columns),
        "Stored schedule must be one table with one float64 member per contract column.",
    )
    scalar_rows = np.asarray(scalar_table[()])
    scalars = tuple((bytes(row["name"]).decode("utf-8"), float(row["value"]), bytes(row["unit"]).decode("utf-8")) for row in scalar_rows)
    rows = np.asarray(schedule_table[()])
    schedule = _read_only(np.column_stack([rows[name] for name in schedule_columns]).reshape(len(rows), len(schedule_columns)))
    return StoredCaseInputs(fields=fields, scalars=scalars, schedule=schedule)


def _load_inputs(directory: Path, input_contract: Mapping[str, Any], store_identity: InputStoreIdentity | None) -> StoredCaseInputs:
    """Read and validate one stored container from one byte snapshot, optionally against its recorded integrity."""
    path = require_input_store(directory)
    data = path.read_bytes()
    if store_identity is not None and (len(data) != store_identity.size_bytes or hashlib.sha256(data).hexdigest() != store_identity.sha256):
        message = f"Stored case inputs disagree with their recorded storage integrity: {path}"
        raise ValueError(message)
    try:
        with h5py.File(io.BytesIO(data), "r") as handle:
            content = _read_container(handle, input_contract)
    except OSError as error:
        message = f"Stored case inputs are not a readable HDF5 container: {path}"
        raise ValueError(message) from error
    _validate_inputs(content, input_contract)
    return content


def render_input_files(content: StoredCaseInputs, input_contract: Mapping[str, Any]) -> dict[str, bytes]:
    """
    Render the canonical COMSOL CSV bytes of one stored case; this is the sole CSV renderer.

    Parameters
    ----------
    content : StoredCaseInputs
        Validated stored inputs.
    input_contract : Mapping[str, Any]
        Case input contract owning filenames, delimiters, and column order.

    Returns
    -------
    dict[str, bytes]
        UTF-8 LF CSV bytes keyed by COMSOL filename, sorted by filename.

    """
    spatial = input_contract["spatial"]
    columns = list(spatial["columns"])
    flattened = [content.fields[name].ravel(order="F") for name in columns]
    rows = [columns]
    rows.extend([_format_number(float(value)) for value in values] for values in zip(*flattened, strict=True))
    rendered = {spatial["filename"]: _table_text(rows, delimiter=spatial["delimiter"])}
    if content.schedule is not None:
        scalar, schedule = input_contract["scalar"], input_contract["schedule"]
        scalar_rows = [list(scalar["columns"]), *([name, _format_number(value), unit] for name, value, unit in content.scalars)]
        rendered[scalar["filename"]] = _table_text(scalar_rows, delimiter=scalar["delimiter"])
        schedule_rows = [list(schedule["columns"]), *([_format_number(float(value)) for value in row] for row in content.schedule)]
        rendered[schedule["filename"]] = _table_text(schedule_rows, delimiter=schedule["delimiter"])
    return {filename: rendered[filename].encode("utf-8") for filename in sorted(rendered)}


def _input_file_identities(rendered: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    """Return the canonical ``input_files`` record of rendered CSV bytes."""
    return {filename: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)} for filename, data in sorted(rendered.items())}


def _require_rendered_identities(rendered: Mapping[str, bytes], case_payload: Mapping[str, Any], *, location: Path) -> None:
    """Require rendered CSV bytes to reproduce the case's canonical input identities."""
    if _input_file_identities(rendered) != case_payload["input_files"]:
        message = f"Stored case inputs do not render the canonical input identities of case.json: {location}"
        raise ValueError(message)


def _write_case_inputs(
    directory: Path,
    input_contract: Mapping[str, Any],
    *,
    fields: Mapping[str, np.ndarray],
    scalars: Sequence[tuple[str, float, str]],
    schedule: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    """Atomically store one case's inputs and return the ``input_files`` identities rendered from the file read back."""
    columns = list(input_contract["spatial"]["columns"])
    missing = [name for name in columns if name not in fields]
    _require(not missing, f"Spatial adapter requests unknown generated fields: {missing}.")
    content = StoredCaseInputs(
        fields={name: np.asarray(fields[name], dtype=np.float64) for name in columns},
        scalars=tuple((str(name), float(value), str(unit)) for name, value, unit in scalars),
        schedule=None if schedule is None else np.asarray(schedule, dtype=np.float64),
    )
    _validate_inputs(content, input_contract)
    common.serialization.atomic_path_write(
        directory / common.paths.GENERATION_INPUT_STORE_FILENAME,
        lambda path: _write_container(path, content, input_contract),
    )
    return _input_file_identities(render_input_files(_load_inputs(directory, input_contract, None), input_contract))


def read_case_inputs(directory: Path, case_payload: Mapping[str, Any], *, store_identity: InputStoreIdentity | None = None) -> StoredCaseInputs:
    """
    Read one case's stored inputs after proving that they render its canonical identities.

    Parameters
    ----------
    directory : Path
        Case input directory holding the stored input container.
    case_payload : Mapping[str, Any]
        Validated case payload with ``input_contract`` and ``input_files``.
    store_identity : InputStoreIdentity | None, optional
        Recorded storage integrity; the container bytes must match it when supplied.

    Returns
    -------
    StoredCaseInputs
        Validated read-only stored content.

    Raises
    ------
    FileNotFoundError
        If the directory has no container, for example CSV-only inputs that must be regenerated.
    ValueError
        If integrity, layout, dtype, shape, finiteness, grid order, or rendered identities disagree.

    """
    content = _load_inputs(directory, case_payload["input_contract"], store_identity)
    _require_rendered_identities(render_input_files(content, case_payload["input_contract"]), case_payload, location=directory)
    return content


def materialize_case_inputs(
    source_directory: Path,
    work_directory: Path,
    case_payload: Mapping[str, Any],
    *,
    store_identity: InputStoreIdentity,
) -> tuple[Path, ...]:
    """
    Render one integrity-verified stored case into its COMSOL CSV files before execution.

    Parameters
    ----------
    source_directory : Path
        Canonical case input directory holding the stored input container.
    work_directory : Path
        Isolated COMSOL work directory that receives the CSV files.
    case_payload : Mapping[str, Any]
        Validated case payload with ``input_contract`` and ``input_files``.
    store_identity : InputStoreIdentity
        Recorded storage integrity of the container.

    Returns
    -------
    tuple[Path, ...]
        Written CSV paths, sorted by filename.

    Raises
    ------
    ValueError
        If the container or its rendered identities disagree with the recorded evidence.
    RuntimeError
        If a written file does not hold its canonical identity bytes.

    """
    contract = case_payload["input_contract"]
    rendered = render_input_files(_load_inputs(source_directory, contract, store_identity), contract)
    _require_rendered_identities(rendered, case_payload, location=source_directory)
    paths = []
    for filename, data in rendered.items():
        path = common.serialization.atomic_write_bytes(work_directory / filename, data)
        identity = case_payload["input_files"][filename]
        if path.stat().st_size != identity["size_bytes"] or common.serialization.file_sha256(path) != identity["sha256"]:
            message = f"Rendered COMSOL input disagrees with its case identity: {path}"
            raise RuntimeError(message)
        paths.append(path)
    return tuple(paths)


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
    """Return profile substreams, with recorded case seed domains."""
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
    return {
        label: seeding.derive_seed(
            case_seed,
            "case_substream",
            label,
        )
        for label in labels
    }


def generate_case_input_bundle(
    config: config_contract.GenerationConfig,
    case_index: int,
    destination: Path | str,
    *,
    intervention: Mapping[str, Any] | None = None,
    _allow_workspace_marker: bool = False,
) -> CaseBundle:
    """Generate one deterministic profile-specific scientific input bundle, optionally with one declared sampling intervention."""
    case_id = config.case_id(case_index)
    case_seed = config.case_seed(case_index)
    assignment = config.case_assignment(case_index)
    bundle_dir = Path(destination).expanduser().resolve()
    _require_empty_destination(
        bundle_dir,
        allow_workspace_marker=_allow_workspace_marker,
    )
    sample = sampling_service.sample_case(config, case_index, intervention=intervention)
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
            config.scientific_values.get("schedule_time", config.scientific_values["time"]),
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
            config.scientific_values.get("schedule_time", config.scientific_values["time"]),
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
            source_air_temperature=float(values["T_amb"]),
            initial_dry_basis_moisture=fields.columns["X_0_db_field"],
            oswin_parameters={name: values[name] for name in ("A_osw", "B_osw", "C_osw")},
            pressure=float(fixed["p_ref"]),
            output_time_contract=config.scientific_values["time"] if "schedule_time" in config.scientific_values else None,
        )

    input_contract = config.scientific_values["input_contract"]
    scalar_entries: tuple[scalar_handoff_contract.ScalarHandoffEntry, ...] | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        scalar_entries = scalar_handoff_contract.build_transient_scalar_entries(values, units)
    if boundary_schedule is not None and tuple(input_contract["schedule"]["columns"]) != profiles.SCHEDULE_FIELDS:
        msg = "Generated schedule does not satisfy the configured adapter field contract."
        raise ValueError(msg)
    input_files = _write_case_inputs(
        bundle_dir,
        input_contract,
        fields=fields.columns,
        scalars=[(entry.name, entry.value, entry.unit) for entry in scalar_entries or ()],
        schedule=None if boundary_schedule is None else boundary_schedule.values,
    )
    export_contract_sha256 = common.serialization.canonical_json_sha256(config.scientific_values["output_contract"])
    steady_flow_conditioning = config.scientific_values["steady_flow_conditioning"]
    steady_flow_conditioning_digest = common.serialization.canonical_json_sha256(steady_flow_conditioning)
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
        "subseed_origins": dict.fromkeys(subseeds, "profile_case"),
        "derivation": "sha256(seed_base|semantic_labels)",
        "subseeds": subseeds,
    }
    case_payload: dict[str, Any] = {
        "schema_kind": CASE_SCHEMA_KIND,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "scientific_config_digest": config.scientific_config_digest,
        "case_input_config_digest": config.case_input_config_digest,
        "case_id": case_id,
        "case_index": case_index,
        "git_commit": source_service.required_git_commit(),
        "material_family": assignment["material_family"],
        "domain_profile": config.scientific_values["domain_profile"],
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
    scalar_admission: scalar_handoff_contract.ScalarHandoffAdmission | None = None
    if scalar_entries is not None:
        scalar_filename = input_contract["scalar"]["filename"]
        scalar_entries_payload = [entry.as_dict() for entry in scalar_entries]
        case_payload["scalar_handoff"] = {
            "mechanism": "case_local_long_form_csv",
            "filename": scalar_filename,
            "fresh_per_case": True,
            "runtime_validation": "required",
            "entries": scalar_entries_payload,
        }
        case_payload["scalars"] = scalar_entries_payload
        scalar_admission = scalar_handoff_contract.admit_transient_scalar_handoff(
            scalar_entries,
            source_path=Path(scalar_filename),
            source_filename=scalar_filename,
            sha256=input_files[scalar_filename]["sha256"],
            size_bytes=input_files[scalar_filename]["size_bytes"],
        )
    if boundary_schedule is not None:
        case_payload["schedule_diagnostics"] = boundary_schedule.metadata
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
        scalar_handoff=scalar_admission,
    )
