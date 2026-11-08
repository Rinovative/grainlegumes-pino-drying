"""Materialize bounded task-aware EDA frames from admitted Source cases.

Responsibilities:
  - Read an exact campaign case prefix through Source admission
  - Preserve task field physical units, stored representations, roles, and identities
  - Derive speed magnitude when the task declares compatible velocity outputs

Design principles:
  - EDA reads Source evidence without assigning training or Evaluation roles
  - Bounded prefixes preserve campaign candidate order and disclose available case counts
  - Task contracts remain authoritative for field names and scientific metadata

This module does NOT:
  - Build final training datasets or fit preprocessing state
  - Read model outputs, checkpoints, or experiment run directories
  - Implement plots or notebook controls. EDA plot and UI modules own presentation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from src import domain

_SPATIAL_DIMENSIONS = 2
_REQUIRED_SOURCE_KEYS: Final = (
    "total_discovered_case_count",
    "failed_case_count",
    "incomplete_case_count",
    "invalid_case_count",
    "exclusion_reasons",
    "case_accounting_scope",
    "analysis_representation",
    "dataset_backend",
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.analysis.eda.eda_sources import GeneratedOutputEDABatch
    from src.domain.tasks.domain_task_spec import TaskSpec


def _require_source_keys(loaded: dict[str, Any], *, batch_name: str) -> None:
    """Require every source to report its case accounting and labels; a missing count is never read as zero."""
    missing = [key for key in _REQUIRED_SOURCE_KEYS if key not in loaded]
    if missing:
        message = f"EDA source {batch_name!r} does not report {missing}; a source without such cases must state 0 explicitly."
        raise ValueError(message)


def _transient_dataframe(
    loaded: dict[str, Any],
    *,
    batch_name: str,
    task: TaskSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one nested completed-case frame without flattening trajectories."""
    from . import eda_transient as transient  # noqa: PLC0415

    _require_source_keys(loaded, batch_name=batch_name)
    sample_ids = loaded["sample_ids"]
    rows = loaded["rows"]
    available = int(loaded["available_case_count"])
    frame = pd.DataFrame(rows, index=pd.Index(sample_ids, name="sample_id"))
    if frame.empty:
        message = "Transient completed-output EDA requires at least one admitted case."
        raise ValueError(message)
    contracts = frame.iloc[0]["contracts"]
    categories = {
        "dynamic_state": tuple(contracts["state_names"]),
        "static_spatial": tuple(contracts["static_names"]),
        "boundary_interval": tuple(contracts["boundary_names"]),
        "scalar_material": tuple(contracts["scalar_names"]),
        "complete_schedule": tuple(contracts["schedule_names"]),
        "global_series": tuple(contracts["global_names"]),
        "final_status": tuple(contracts["final_status_names"]),
    }
    unit_groups = (
        (contracts["state_names"], contracts["state_units"]),
        (contracts["static_names"], contracts["static_units"]),
        (contracts["boundary_names"], contracts["boundary_units"]),
        (contracts["scalar_names"], contracts["scalar_units"]),
        (contracts["schedule_names"], contracts["schedule_units"]),
        (contracts["global_names"], contracts["global_units"]),
        (contracts["final_status_names"], contracts["final_status_units"]),
    )
    field_units: dict[str, str] = {}
    for names, units in unit_groups:
        for name, unit in zip(names, units, strict=True):
            previous = field_units.setdefault(name, unit)
            if previous != unit:
                message = f"Transient EDA field {name!r} has inconsistent units."
                raise ValueError(message)
    field_roles = {name: category for category, names in categories.items() for name in names}
    state_names = categories["dynamic_state"]
    excluded = available - len(frame)
    frame.attrs.update(
        {
            "task_id": task.id,
            "task_contract_digest": task.contract_digest,
            "field_names": state_names,
            "field_units": field_units,
            "field_representations": dict.fromkeys(state_names, "absolute_physical_state"),
            "field_roles": field_roles,
            "field_categories": categories,
            "generated_batch_identity": loaded["generated_batch_identity"],
            "source_manifest_sha256": loaded["manifest_sha256"],
            "loaded_case_count": len(frame),
            "available_case_count": available,
            "total_discovered_case_count": int(loaded["total_discovered_case_count"]),
            "failed_case_count": int(loaded["failed_case_count"]),
            "incomplete_case_count": int(loaded["incomplete_case_count"]),
            "invalid_case_count": int(loaded["invalid_case_count"]),
            "exclusion_reasons": dict(loaded["exclusion_reasons"]) | ({} if excluded == 0 else {"bounded_prefix": excluded}),
            "case_accounting_scope": str(loaded["case_accounting_scope"]),
            "spatial_shape": tuple(np.asarray(frame.iloc[0]["state_trajectories"][state_names[0]]).shape[1:]),
            "analysis_representation": loaded["analysis_representation"],
            "dataset_backend": loaded["dataset_backend"],
        }
    )
    transient.validate_transient_frame(frame)
    root = loaded["generation_root"]
    loading_scope = f"the first {len(sample_ids)} of {available}" if len(sample_ids) < available else f"all {available}"
    logs = [
        f"[INFO] Loading {loading_scope} samples.",
        f"[INFO] Generated batch: {batch_name!r} for task {task.id!r} from {root}",
        f"[INFO] Final DataFrame contains {len(frame)} completed trajectories.",
        f"[INFO] Semantic columns: {', '.join(frame.columns)}",
        f"[INFO] Dynamic states: {', '.join(state_names)}",
    ]
    return frame, logs


def _steady_dataframe(
    loaded: dict[str, Any],
    *,
    batch_name: str,
    task: TaskSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one steady-flow frame with shared derived-field and accounting semantics."""
    _require_source_keys(loaded, batch_name=batch_name)
    sample_ids = loaded["sample_ids"]
    rows = loaded["rows"]
    available = int(loaded["available_case_count"])
    frame = pd.DataFrame(rows, index=pd.Index(sample_ids, name="sample_id"))
    declared_fields = (*task.inputs, *task.outputs)
    field_names = tuple(field.name for field in declared_fields)
    field_units = {field.name: field.unit for field in declared_fields}
    field_representations = {field.name: field.representation for field in declared_fields}
    field_roles: dict[str, str] = {field.name: field.role for field in declared_fields}
    if {"u", "v"}.issubset(task.output_names) and field_units["u"] == field_units["v"]:
        speed = [np.hypot(np.asarray(u, dtype=float), np.asarray(v, dtype=float)) for u, v in zip(frame["u"], frame["v"], strict=True)]
        frame["U"] = pd.Series(speed, index=frame.index, dtype=object)
        field_names = (*field_names, "U")
        field_units["U"] = field_units["u"]
        field_representations["U"] = "derived_speed_magnitude"
        field_roles["U"] = "derived_speed"
    excluded = available - len(frame)
    frame.attrs.update(
        {
            "task_id": task.id,
            "task_contract_digest": task.contract_digest,
            "field_names": field_names,
            "field_units": field_units,
            "field_representations": field_representations,
            "field_roles": field_roles,
            "generated_batch_identity": loaded["generated_batch_identity"],
            "source_manifest_sha256": loaded["manifest_sha256"],
            "loaded_case_count": len(frame),
            "available_case_count": available,
            "total_discovered_case_count": int(loaded["total_discovered_case_count"]),
            "failed_case_count": int(loaded["failed_case_count"]),
            "incomplete_case_count": int(loaded["incomplete_case_count"]),
            "invalid_case_count": int(loaded["invalid_case_count"]),
            "exclusion_reasons": dict(loaded["exclusion_reasons"]) | ({} if excluded == 0 else {"bounded_prefix": excluded}),
            "case_accounting_scope": str(loaded["case_accounting_scope"]),
            "analysis_representation": str(loaded["analysis_representation"]),
            "dataset_backend": str(loaded["dataset_backend"]),
            "spatial_shape": None if frame.empty else tuple(frame.iloc[0][task.input_names[0]].shape),
        }
    )
    root = loaded["generation_root"]
    loading_scope = f"the first {len(sample_ids)} of {available}" if len(sample_ids) < available else f"all {available}"
    logs = [
        f"[INFO] Loading {loading_scope} samples.",
        f"[INFO] Generated batch: {batch_name!r} for task {task.id!r} from {root}",
        f"[INFO] Final DataFrame contains {len(frame)} samples.",
        f"[INFO] Columns: {', '.join(frame.columns)}",
    ]
    if not frame.empty:
        shapes = {column: getattr(frame[column].iloc[0], "shape", None) for column in frame.columns}
        logs.append(f"[INFO] Example shapes: {shapes}")
    return frame, logs


def load_published_dataset_frame(
    manifest: dict[str, Any],
    *,
    storage_root: Path,
    max_cases: int | None,
) -> pd.DataFrame:
    """
    Read actual model-facing package values with full-grid scientific support.

    Steady PT admission necessarily materializes the complete payload once.
    Transient admission reads the index, then limits source-array reads to the
    selected complete cases. Fitted Training normalizers are never invented.
    """
    from src import datasets  # noqa: PLC0415
    from src.datasets.runtime import dataset_runtime_steady as steady_runtime  # noqa: PLC0415
    from src.datasets.runtime import dataset_runtime_transient as transient_runtime  # noqa: PLC0415

    from . import eda_transient as transient  # noqa: PLC0415

    view = datasets.contracts.views.get_view(str(manifest["dataset_view"]))
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition") if view.id == "transient_drying" else None
    identities = manifest["source_case_identities"]
    selected = identities if max_cases is None else identities[:max_cases]
    package_ids = tuple(str(case["package_case_id"]) for case in selected)
    request = datasets.runtime.factory.DatasetRequest(
        dataset_id=str(manifest["dataset_id"]),
        dataset_view=view.id,
        case_ids=package_ids,
        transient_sampling=sampling,
        storage_root=storage_root,
        spatial_stride=1,
    )
    runtime = (
        datasets.runtime.factory.create_hdf5_source_dataset(request)
        if view.id == "transient_drying"
        else datasets.runtime.factory.create_dataset(request)
    )
    task = domain.tasks.registry.get_task(view.id)
    if isinstance(runtime, transient_runtime.TransientPhysicalDataset):
        try:
            cases = {case["package_case_id"]: case for case in runtime.payload["cases"]}
            items = []
            for package_id in package_ids:
                case = cases[package_id]
                arrays = runtime.read_case_arrays(package_id)
                items.append(
                    {
                        "state": arrays.states[0],
                        "state_sequence": arrays.states,
                        "target": arrays.states[1:] - arrays.states[:-1],
                        "static": arrays.static,
                        "scalars": arrays.scalars,
                        "boundary": arrays.boundary,
                        "time": {"t_n": arrays.time[:-1], "t_n_plus_1": arrays.time[1:], "dt": np.diff(arrays.time)},
                        "metadata": {
                            **{
                                key: case[key]
                                for key in (
                                    "simulation_case_id",
                                    "case_input_id",
                                    "source_batch_id",
                                    "material_family",
                                    "source_simulation_profile",
                                )
                            },
                            "dataset_id": runtime.dataset_id,
                            "configured_horizon_hours": runtime.configured_regular_horizon,
                        },
                    }
                )
            frame = transient.frame_from_transient_items(items, backend=runtime.storage_backend)
        finally:
            runtime.close()
    elif isinstance(runtime, steady_runtime.SteadyFlowDataset | steady_runtime.SteadyDatasetSelection):
        positions = {sample_id: index for index, sample_id in enumerate(runtime.identity.sample_ids)}
        rows: list[dict[str, Any]] = []
        for case_id in package_ids:
            item = runtime[positions[case_id]]
            inputs, targets = item["x"].numpy(), item["y"].numpy()
            rows.append(
                {field: inputs[index] for index, field in enumerate(task.input_names)}
                | {field: targets[index] for index, field in enumerate(task.output_names)}
                | {"meta": dict(item["meta"])}
            )
        loaded = {
            "sample_ids": package_ids,
            "rows": rows,
            "available_case_count": len(identities),
            "total_discovered_case_count": len(identities),
            # A published package holds admitted cases only: this path defines no failed, incomplete or invalid ones.
            "failed_case_count": 0,
            "incomplete_case_count": 0,
            "invalid_case_count": 0,
            "exclusion_reasons": {},
            "generated_batch_identity": None,
            "manifest_sha256": manifest["dataset_digest"],
            "generation_root": storage_root,
            "analysis_representation": "steady_field_rows",
            "dataset_backend": "published_steady_pt",
            "case_accounting_scope": "published_package_membership",
        }
        frame, _logs = _steady_dataframe(loaded, batch_name=str(manifest["dataset_name"]), task=task)
        frame.attrs["source_physical_field_units"] = dict(frame.attrs["field_units"])
        frame.attrs["field_units"] = {
            name: "1" if frame.attrs["field_representations"][name].startswith("dimensionless_") else unit
            for name, unit in frame.attrs["field_units"].items()
        }
    else:
        message = f"Unsupported published runtime type: {type(runtime).__name__}."
        raise TypeError(message)
    frame.attrs.update(
        dataset_id=manifest["dataset_id"],
        dataset_manifest=manifest,
        dataset_view=view.id,
        available_case_count=len(identities),
        loaded_case_count=len(selected),
        case_accounting_scope="published_package_membership",
        normalization="Published values before train-fitted normalization; no fitted scaler is loaded.",
    )
    return frame


def generate_eda_dataframe_from_completed_cases(
    batch: GeneratedOutputEDABatch,
    *,
    task: TaskSpec,
    show_progress: bool = True,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Materialize exact successful Source cases from one current campaign view."""
    del show_progress  # Case-local EDA avoids notebook progress side effects.
    if task.id != batch.simulation_profile:
        message = "Generation EDA task contradicts its selected campaign."
        raise ValueError(message)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        message = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise ValueError(message)
    selected = batch.cases if max_cases is None else batch.cases[:max_cases]
    if not selected:
        message = f"Generation EDA campaign {batch.campaign_name!r} has no successful cases."
        raise ValueError(message)
    from src import datasets  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for case in selected:
        admitted = batch.admit_case(case)
        if task.id == "transient_drying":
            rows.append(datasets.packages.generated_batch.interpret_admitted_transient_source_case(admitted, task=task))
        elif task.id == "steady_flow":
            _shape, inputs, outputs, metadata, source, _fingerprint = datasets.packages.generated_batch.interpret_admitted_source_case(
                admitted,
                package_case_id=case.case_id,
                task=task,
            )
            rows.append(
                {
                    **{name: inputs[index].numpy() for index, name in enumerate(task.input_names)},
                    **{name: outputs[index].numpy() for index, name in enumerate(task.output_names)},
                    "meta": {**metadata, "source_design_provenance": admitted.design_provenance},
                    "source": source,
                }
            )
        else:
            message = f"Generation EDA does not support task {task.id!r}."
            raise ValueError(message)
    loaded = {
        "sample_ids": [case.case_id for case in selected],
        "rows": rows,
        "available_case_count": batch.available_case_count,
        "total_discovered_case_count": batch.discovered_case_count,
        "failed_case_count": batch.failed_case_count,
        "incomplete_case_count": batch.incomplete_case_count,
        "invalid_case_count": batch.invalid_case_count,
        "exclusion_reasons": {},
        "case_accounting_scope": "generation_source_campaign_candidates",
        "task": task,
        "generated_batch_identity": {
            "schema_kind": "analysis_generation_campaign_selection",
            "campaign_name": batch.campaign_name,
            "campaign_digest": batch.campaign_digest,
            "material_family": batch.material_family,
            "domain_profile": batch.domain_profile,
            "source_selectors": [case.selector for case in selected],
            "design_provenance": [
                row["source"]["design_provenance"] if task.id == "transient_drying" else row["meta"]["source_design_provenance"] for row in rows
            ],
        },
        "manifest_sha256": None,
        "generation_root": batch.generation_root,
        "analysis_representation": ("steady_field_rows" if task.id == "steady_flow" else "transient_complete_case_rows"),
        "dataset_backend": "canonical_generation_hdf5",
    }
    if task.id == "transient_drying":
        return _transient_dataframe(loaded, batch_name=batch.campaign_name, task=task)
    return _steady_dataframe(loaded, batch_name=batch.campaign_name, task=task)


@dataclass(frozen=True, slots=True)
class PublishedTensorValue:
    """Bind a stored model input or target to its authoritative tensor contract."""

    group: str
    purpose: str
    name: str
    unit: str
    representation: str
    support: str
    array: np.ndarray


def published_tensor_values(frame: pd.DataFrame, case_id: str, *, transition: int = 0) -> tuple[PublishedTensorValue, ...]:
    """Inspect actual package inputs and targets without fitting or decoding them."""
    from src import datasets  # noqa: PLC0415

    view_id = frame.attrs.get("dataset_view")
    if not isinstance(view_id, str) or case_id not in frame.index:
        message = "Published tensor inspection requires an admitted package and exact case."
        raise ValueError(message)
    inspection = datasets.contracts.views.inspect_contract(view_id)
    row = frame.loc[case_id]
    transient = view_id == "transient_drying"
    if transient:
        transition_times = np.asarray(row["time"]["transition_t_n_hours"])
        if isinstance(transition, bool) or not isinstance(transition, int) or not 0 <= transition < transition_times.size:
            message = "Transient tensor inspection requires an exact stored transition index."
            raise ValueError(message)
    elif transition != 0:
        message = "Steady tensor inspection has no transition dimension."
        raise ValueError(message)
    results = []
    for group in inspection.groups:
        if group.purpose == "archive":
            continue
        for field in group.fields:
            if not transient:
                values = np.asarray(row[field.name])
                support = "domain"
                if field.name == "p_in_bc":
                    x = np.asarray(row["x"])
                    values = values[x == np.min(x)]
                    support = "inlet boundary"
            elif group.name == "state":
                values = np.asarray(row["state_trajectories"][field.name][transition])
                support = "domain at current state time"
            elif group.name == "target":
                values = np.asarray(row["target_increments"][field.name][transition])
                support = "domain; next-state minus current-state"
            elif group.name == "static":
                values = np.asarray(row["static_fields"][field.name])
                support = "static domain"
            elif group.name == "boundary":
                values = np.asarray(row["boundary_intervals"][field.name][transition])
                support = "transition boundary conditioning"
            elif group.name == "scalars":
                values = np.asarray(row["scalar_conditioning"][field.name])
                support = "material scalar"
            else:
                message = f"Unsupported published tensor group: {group.name}."
                raise ValueError(message)
            representation = field.representation or "physical value"
            unit = "1" if representation.startswith("dimensionless_") else field.unit
            if values.ndim == _SPATIAL_DIMENSIONS and field.name in {"rho_bu_dry", "rho_gr_dry"} and np.all(values == values.flat[0]):
                values = np.asarray(values.flat[0])
                support = "constant density (metadata)"
            results.append(PublishedTensorValue(group.name, group.purpose, field.name, unit, representation, support, values))
    return tuple(results)
