"""Load membership-filtered published reference cases and maintained settings without changing them.

Responsibilities:
  - Resolve the Train or validation membership of a Dataset from its manifest partition
  - Materialize steady-flow Dataset cases as task-ordered physical tensors
  - Rebuild exact float64 Cartesian coordinates from float32 Dataset grids
  - Read transient Dataset trajectories through the canonical HDF5 runtime
  - Read COMSOL global balance series and case constants from admitted sources
  - Merge maintained learning configurations over the task defaults for read-only inspection

Design principles:
  - Dataset runtimes own membership, conversion, and source admission
  - Only Train or validation membership is admitted, filtered before any case limit
  - Source HDF5 files are re-hashed against the Dataset index before any read
  - Field and unit order of every persisted array is checked, never assumed

This module does NOT:
  - Build, publish, repair, or delete Datasets, sources, artifacts, or configurations
  - Admit ID-test or OOD cases
  - Evaluate residuals or balances, or render figures
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

import h5py
import numpy as np
import torch

from src import common, datasets, domain, experiments, generation

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from torch import Tensor

Membership = Literal["train", "validation"]
MEMBERSHIPS: Final[tuple[str, ...]] = get_args(Membership)
_STEADY_TASK_ID: Final = "steady_flow"
_TRANSIENT_TASK_ID: Final = "transient_drying"
_COORDINATE_ULP_TOLERANCE = 1.0
_CASE_CONSTANTS = ("Lz", "M_v", "p_ref", "phi_clip_min", "phi_clip_max")


@dataclass(frozen=True, slots=True)
class MaintainedConfig:
    """
    Hold one maintained learning configuration merged over its task defaults.

    Attributes
    ----------
    name : str
        Path relative to the task configuration directory.
    kind : {"experiment", "optuna"}
        Authored experiment or Optuna study.
    config : dict[str, Any]
        Authored values merged over the task defaults; run-time values such as
        Dataset identities stay unresolved.
    search_space : dict[str, Any]
        The study search space; empty for experiments.

    """

    name: str
    kind: str
    config: dict[str, Any]
    search_space: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SteadyReference:
    """
    Hold published steady-flow reference cases as task-ordered tensors.

    Attributes
    ----------
    dataset_id : str
        Immutable published Dataset identifier.
    membership : {"train", "validation"}
        Dataset partition role of every case.
    case_ids, material_families : tuple[str, ...]
        Package case identifiers and their material families, in tensor order.
    input_fields, output_fields : tuple[str, ...]
        Task-owned channel names of ``inputs`` and ``outputs``.
    inputs, outputs : torch.Tensor
        Published float32 ``[case, channel, y, x]`` tensors exactly as training
        reads them before normalization.

    """

    dataset_id: str
    membership: Membership
    case_ids: tuple[str, ...]
    material_families: tuple[str, ...]
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    inputs: Tensor
    outputs: Tensor

    def tensors(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """
        Return ``(inputs, outputs)`` in one evaluation precision.

        Parameters
        ----------
        dtype : torch.dtype
            ``torch.float32`` returns the published tensors unchanged.
            ``torch.float64`` casts every channel and replaces ``x`` and ``y``
            with exact uniform float64 grids through the stored endpoints.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            New or shared tensors; callers must not mutate them.

        Raises
        ------
        ValueError
            If ``dtype`` is unsupported or a rebuilt coordinate differs from a
            stored float32 coordinate by more than one float32 ulp.

        """
        if dtype == torch.float32:
            return self.inputs, self.outputs
        if dtype != torch.float64:
            msg = f"Steady reference tensors support float32 or float64, got {dtype}."
            raise ValueError(msg)
        inputs = self.inputs.to(torch.float64)
        for name, axis in (("x", -1), ("y", -2)):
            index = self.input_fields.index(name)
            inputs[:, index] = uniform_float64_coordinate(self.inputs[:, index], axis=axis)
        return inputs, self.outputs.to(torch.float64)

    def extent(self, case_index: int) -> tuple[float, float, float, float]:
        """Return the physical ``(x_min, x_max, y_min, y_max)`` of one case in m."""
        x = self.inputs[case_index, self.input_fields.index("x")]
        y = self.inputs[case_index, self.input_fields.index("y")]
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    @property
    def case_labels(self) -> tuple[str, ...]:
        """Return compact ``material/case`` labels in tensor order."""
        return tuple(
            f"{material}/{case_id.split('__')[1] if '__' in case_id else case_id}"
            for material, case_id in zip(self.material_families, self.case_ids, strict=True)
        )


@dataclass(frozen=True, slots=True)
class TransientReferenceCase:
    """
    Hold one complete transient reference trajectory and its COMSOL balances.

    Attributes
    ----------
    dataset_id, case_id, material_family : str
        Dataset, package case, and material identity.
    membership : {"train", "validation"}
        Dataset partition role of the case.
    time_hours : numpy.ndarray
        Regular stored-state times ``[T]`` in h.
    states : dict[str, numpy.ndarray]
        ``T``, ``phi``, ``w_surf``, and ``w_int`` as float32 ``[T, y, x]``.
    static : dict[str, numpy.ndarray]
        ``x``, ``y``, ``u``, ``v``, ``p``, ``eps_bed``, and ``rho_bu_dry`` as
        float64 ``[y, x]`` fields.
    boundary : dict[str, numpy.ndarray]
        Per-transition inlet conditioning ``[T-1]`` keyed by Dataset name.
    scalars : dict[str, float]
        Case-level material scalars of the Dataset contract.
    global_series : dict[str, numpy.ndarray]
        COMSOL global series keyed by Generation name, including the
        diagnostic exact-stop row when present.
    constants : dict[str, float]
        Case-recorded ``Lz``, ``M_v``, ``p_ref``, and relative-humidity clips.
    exact_stop_time : float | None
        Irregular stop time in h, if the case stopped before the horizon.

    """

    dataset_id: str
    case_id: str
    material_family: str
    membership: Membership
    time_hours: np.ndarray
    states: dict[str, np.ndarray]
    static: dict[str, np.ndarray]
    boundary: dict[str, np.ndarray]
    scalars: dict[str, float]
    global_series: dict[str, np.ndarray]
    constants: dict[str, float]
    exact_stop_time: float | None

    @property
    def label(self) -> str:
        """Return a compact ``material/case`` label."""
        parts = self.case_id.split("__")
        return f"{self.material_family}/{parts[1] if len(parts) > 1 else self.case_id}"


def uniform_float64_coordinate(coordinate: Tensor, *, axis: int) -> Tensor:
    """
    Rebuild an exact uniform float64 coordinate grid from a float32 grid.

    Parameters
    ----------
    coordinate : torch.Tensor
        Stored float32 coordinate field ``[case, y, x]``.
    axis : int
        ``-1`` for x or ``-2`` for y.

    Returns
    -------
    torch.Tensor
        Float64 ``low + (high - low) * i / (n - 1)`` per case with the input shape.

    Raises
    ------
    ValueError
        If the axis is invalid or the rebuilt grid deviates from the stored
        values by more than one float32 ulp, i.e. the grid is not uniform.

    Notes
    -----
    The domain spacing admission adds a dtype-scaled roundoff allowance. A
    float32 grid cast to float64 keeps its float32 quantization but receives a
    float64 allowance and is rejected, so float64 evaluation needs exact grids.

    """
    if axis not in (-1, -2) or coordinate.ndim != 3:  # noqa: PLR2004 -- [case, y, x]
        msg = "Coordinate rebuild requires a [case, y, x] tensor and axis -1 or -2."
        raise ValueError(msg)
    work = coordinate.to(torch.float64)
    count = work.shape[axis]
    low = work.amin(dim=(-2, -1), keepdim=True)
    high = work.amax(dim=(-2, -1), keepdim=True)
    ramp = torch.arange(count, dtype=torch.float64, device=work.device)
    ramp = ramp.view(1, 1, count) if axis == -1 else ramp.view(1, count, 1)
    rebuilt = (low + (high - low) * ramp / (count - 1)).expand_as(work).clone()
    ulp = torch.finfo(torch.float32).eps * work.abs().amax().clamp_min(1.0)
    deviation = (rebuilt.to(torch.float32).to(torch.float64) - work).abs().amax()
    if bool((deviation > _COORDINATE_ULP_TOLERANCE * ulp).item()):
        msg = f"Stored coordinates are not a uniform grid: rebuilt values deviate by {float(deviation):.3e}."
        raise ValueError(msg)
    return rebuilt


def select_membership_cases(
    partition: Mapping[str, Any] | None,
    membership: str,
    *,
    dataset_id: str,
    max_cases: int | None = None,
) -> tuple[str, ...]:
    """
    Return the leading case IDs of one Train or validation membership.

    Parameters
    ----------
    partition : Mapping[str, Any] or None
        The Dataset manifest's ``case_partition``.
    membership : str
        ``"train"`` or ``"validation"``; ID-test and OOD roles are refused.
    dataset_id : str
        Dataset identifier for error messages.
    max_cases : int, optional
        Leading cases kept after the membership filter; all when omitted.

    Returns
    -------
    tuple[str, ...]
        Case IDs in stored partition order.

    Raises
    ------
    ValueError
        If the role is not admitted, ``max_cases`` is not positive, or the
        Dataset has no partition or no case of that membership.

    """
    if membership not in MEMBERSHIPS:
        msg = f"Component checks read only {list(MEMBERSHIPS)} membership, got {membership!r}; test and OOD cases stay unseen."
        raise ValueError(msg)
    if max_cases is not None and max_cases < 1:
        msg = "max_cases must be positive when given."
        raise ValueError(msg)
    cases = None if partition is None else partition.get(membership)
    if not cases:
        msg = f"Dataset {dataset_id!r} has no {membership} membership in its case partition."
        raise ValueError(msg)
    return tuple(str(case) for case in cases)[:max_cases]


def membership_caption(membership: str, labels: Sequence[str]) -> str:
    """Return a table caption naming the membership and its case count; the notebook shows the Dataset id once."""
    return f"{membership} membership · {len(labels)} cases"


def _learning_config_root(task: str, config_root: Path | str | None) -> Path:
    """Return the maintained learning configuration directory of one task."""
    return Path(config_root) if config_root is not None else common.paths.get_project_root() / "configs" / "03_learning" / task


def maintained_base_config(task: str) -> dict[str, Any]:
    """Return the task's shared ``base.yaml`` merged over the task defaults."""
    loader = experiments.config.loader
    base = loader.load_yaml(_learning_config_root(task, None) / loader.AUTHORED_BASE_FILENAME)
    return loader.deep_merge(experiments.config.defaults.get_task_defaults(task), base)


def iter_maintained_configs(task: str, *, config_root: Path | str | None = None) -> Iterator[MaintainedConfig]:
    """
    Yield every maintained experiment and Optuna study of one task, merged over the task defaults.

    Notes
    -----
    Experiments are merged with their task base through the authoritative
    authored-config loader and then over the task defaults with the loader's
    merge; run-time values such as Dataset identities are not resolved.

    """
    loader = experiments.config.loader
    root = _learning_config_root(task, config_root)
    for path in sorted(root.rglob("*.yaml")):
        if path.name == loader.AUTHORED_BASE_FILENAME:
            continue
        raw = loader.load_yaml(path)
        is_study = "experiment" in raw
        authored = loader.apply_authored_base(raw["experiment"] if is_study else raw, origin=path)
        yield MaintainedConfig(
            name=str(path.relative_to(root)),
            kind="optuna" if is_study else "experiment",
            config=loader.deep_merge(experiments.config.defaults.get_task_defaults(task), authored),
            search_space=dict(raw.get("search_space", {})),
        )


def load_steady_reference(
    dataset_id: str,
    *,
    membership: Membership,
    storage_root: Path | str | None = None,
    max_cases: int | None = None,
) -> SteadyReference:
    """
    Load published steady-flow cases of one membership through the Dataset runtime.

    Parameters
    ----------
    dataset_id : str
        Published steady-flow Dataset identifier.
    membership : {"train", "validation"}
        Partition role whose cases are loaded.
    storage_root : pathlib.Path or str, optional
        Scientific storage root; defaults to the canonical resolution.
    max_cases : int, optional
        Leading membership cases to keep; all when omitted.

    Returns
    -------
    SteadyReference
        Float32 task-ordered inputs and outputs of the selected cases.

    Raises
    ------
    ValueError
        If the membership is not admitted or absent, or ``max_cases`` is not positive.

    """
    factory = datasets.runtime.factory
    dataset = factory.create_dataset(factory.DatasetRequest(dataset_id=dataset_id, dataset_view=_STEADY_TASK_ID, storage_root=storage_root))
    if not isinstance(dataset, datasets.runtime.steady.SteadyFlowDataset) or dataset.manifest_binding is None:
        msg = f"Dataset {dataset_id!r} did not resolve to a manifest-bound steady-flow runtime."
        raise TypeError(msg)
    case_ids = select_membership_cases(dataset.manifest_binding.case_partition, membership, dataset_id=dataset_id, max_cases=max_cases)
    positions = datasets.preprocessing.splits.role_positions(dataset.identity, case_ids, label=membership)
    items = [dataset[index] for index in positions]
    metadata: list[Mapping[str, object]] = [item["meta"] for item in items]
    if tuple(str(meta["case_id"]) for meta in metadata) != case_ids:
        msg = f"Steady Dataset {dataset_id!r} returned cases that differ from its {membership} membership."
        raise ValueError(msg)
    task = domain.tasks.registry.get_task(_STEADY_TASK_ID)
    return SteadyReference(
        dataset_id=dataset_id,
        membership=membership,
        case_ids=case_ids,
        material_families=tuple(str(meta["material_family"]) for meta in metadata),
        input_fields=tuple(task.input_names),
        output_fields=tuple(task.output_names),
        inputs=torch.stack([item["x"] for item in items]),
        outputs=torch.stack([item["y"] for item in items]),
    )


def _named_array(handle: h5py.File, name: str, *, names: tuple[str, ...], units: tuple[str, ...]) -> np.ndarray:
    """Read one finite HDF5 array after checking its exact field and unit order."""
    dataset = handle.get(name)
    if not isinstance(dataset, h5py.Dataset):
        msg = f"Source HDF5 member {name!r} must be a dataset."
        raise TypeError(msg)
    decode = datasets.packages.trajectory.decode_json_string_list
    stored_names = tuple(decode(dataset.attrs.get("field_names"), label=f"{name}/field_names"))
    stored_units = tuple(decode(dataset.attrs.get("units"), label=f"{name}/units"))
    if stored_names != names or stored_units != units:
        msg = f"Source HDF5 member {name!r} has field order {stored_names} and units {stored_units}, expected {names} and {units}."
        raise ValueError(msg)
    values = np.asarray(dataset, dtype=np.float64)
    if not np.isfinite(values).all():
        msg = f"Source HDF5 member {name!r} contains non-finite values."
        raise ValueError(msg)
    return values


def _case_constants(handle: h5py.File) -> dict[str, float]:
    """Read case-recorded geometry and gas constants from digest-admitted source provenance."""
    dataset = handle.get("provenance/scientific_config_json")
    if not isinstance(dataset, h5py.Dataset):
        msg = "Source HDF5 lacks its scientific configuration provenance."
        raise TypeError(msg)
    raw = dataset[()]
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        msg = "Source scientific configuration provenance must be JSON text."
        raise TypeError(msg)
    config: Mapping[str, Any] = json.loads(raw)
    digest = handle.attrs.get("scientific_config_digest")
    digest = digest.decode("utf-8") if isinstance(digest, bytes) else digest
    if not isinstance(config, dict) or generation.cases.config.compute_scientific_config_digest(config) != digest:
        msg = "Source scientific configuration provenance disagrees with its HDF5 identity digest."
        raise ValueError(msg)
    fixed = config["scientific_fixed_values"]
    values = {"Lz": config["grid"]["Lz"], **{name: fixed[name] for name in _CASE_CONSTANTS if name != "Lz"}}
    constants = {name: float(values[name]) for name in _CASE_CONSTANTS}
    if not all(np.isfinite(value) and value > 0 for value in constants.values()):
        msg = "Source case constants must be finite and positive."
        raise ValueError(msg)
    return constants


def _file_stat(path: Path) -> tuple[int, int]:
    """Return the size and modification time used to detect a changed source."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _read_source_balances(path: Path, *, expected_sha256: str, admitted_stat: tuple[int, int]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read one admitted source's global series and constants, re-hashing it only if it changed."""
    if _file_stat(path) != admitted_stat:
        datasets.contracts.identity.require_admitted_source_bytes(path, expected_sha256)
    profiles = generation.contracts.profiles
    with h5py.File(path, "r") as handle:
        values = _named_array(handle, "global/values", names=tuple(profiles.GLOBAL_FIELD_NAMES), units=tuple(profiles.GLOBAL_FIELD_UNITS))
        constants = _case_constants(handle)
    return {name: np.ascontiguousarray(values[:, index]) for index, name in enumerate(profiles.GLOBAL_FIELD_NAMES)}, constants


def iter_transient_reference_cases(
    dataset_id: str,
    *,
    membership: Membership,
    storage_root: Path | str | None = None,
    max_cases: int | None = None,
) -> Iterator[TransientReferenceCase]:
    """
    Yield complete transient reference cases of one membership one at a time.

    Parameters
    ----------
    dataset_id : str
        Published transient Dataset identifier.
    membership : {"train", "validation"}
        Partition role whose cases are yielded.
    storage_root : pathlib.Path or str, optional
        Scientific storage root; defaults to the canonical resolution.
    max_cases : int, optional
        Leading membership cases to yield; all when omitted.

    Yields
    ------
    TransientReferenceCase
        One case with full-resolution states, statics, and COMSOL balances.

    Raises
    ------
    ValueError
        If the membership is not admitted or absent, ``max_cases`` is not
        positive, or a source fails its hash or contract.

    Notes
    -----
    Trajectories come from the canonical-HDF5 Dataset runtime at spatial
    stride one, which hashes every Dataset source on construction, also when
    fewer cases are selected. The COMSOL global series are not part of the
    Dataset contract; they are read from the same source, which is re-hashed
    only if its size or modification time changed since admission.

    """
    factory = datasets.runtime.factory
    contract = datasets.contracts.transient
    dataset = factory.create_hdf5_source_dataset(
        factory.DatasetRequest(
            dataset_id=dataset_id,
            dataset_view=_TRANSIENT_TASK_ID,
            transient_sampling=contract.TransientSamplingSpec(mode="trajectory"),
            storage_root=storage_root,
        )
    )
    step = contract.TRANSIENT_STEP_CONTRACT
    state_names = tuple(field.name for field in step.dynamic_state)
    static_names = tuple(field.name for field in step.static_spatial_conditioning)
    boundary_names = tuple(field.name for field in step.step_boundary_conditioning)
    scalar_names = tuple(field.name for field in step.scalar_conditioning)
    binding = dataset.manifest_binding
    try:
        partition = None if binding is None else binding.case_partition
        case_ids = select_membership_cases(partition, membership, dataset_id=dataset_id, max_cases=max_cases)
        by_id = {str(case["package_case_id"]): case for case in dataset.payload["cases"]}
        selected = [by_id[case_id] for case_id in case_ids]
        source_paths = [(dataset.source_root / str(case["source_relative_path"])).resolve() for case in selected]
        admitted_stats = [_file_stat(path) for path in source_paths]
        for case, source_path, admitted_stat in zip(selected, source_paths, admitted_stats, strict=True):
            arrays = dataset.read_case_arrays(str(case["package_case_id"]))
            global_series, constants = _read_source_balances(
                source_path, expected_sha256=str(case["source_hdf5_sha256"]), admitted_stat=admitted_stat
            )
            yield TransientReferenceCase(
                dataset_id=dataset_id,
                case_id=str(case["package_case_id"]),
                material_family=str(case["material_family"]),
                membership=membership,
                time_hours=np.asarray(arrays.time, dtype=np.float64),
                states={name: arrays.states[:, index] for index, name in enumerate(state_names)},
                static={name: np.asarray(arrays.static[index], dtype=np.float64) for index, name in enumerate(static_names)},
                boundary={name: np.asarray(arrays.boundary[:, index], dtype=np.float64) for index, name in enumerate(boundary_names)},
                scalars={name: float(arrays.scalars[index]) for index, name in enumerate(scalar_names)},
                global_series=global_series,
                constants=constants,
                exact_stop_time=arrays.exact_stop_time,
            )
    finally:
        dataset.close()
