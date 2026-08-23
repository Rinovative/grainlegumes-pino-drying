"""
eda_dataframe.py

Materialize bounded task-aware EDA frames from generated COMSOL batches.

Responsibilities:
  - Load a validated manifest-ordered generated-batch prefix
  - Preserve task field physical units, stored representations, roles, and identities
  - Derive speed magnitude when the task declares compatible velocity outputs

Design principles:
  - EDA reads the generation domain without resolving training datasets
  - Bounded prefixes preserve manifest order and disclose available case counts
  - Task contracts remain authoritative for field names and scientific metadata

This module does NOT:
  - Build final training datasets or fit preprocessing state
  - Read model outputs, checkpoints, or experiment run directories
  - Implement plots or notebook controls. EDA plot and UI modules own presentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.tasks.domain_task_spec import TaskSpec


class _CompletedCaseBatch(Protocol):
    """Describe one EDA-visible collection of Generation-admitted completed cases."""

    @property
    def source_kind(self) -> str:
        """Return terminal or partial source classification."""
        ...

    @property
    def batch_id(self) -> str:
        """Return immutable batch identity."""
        ...

    @property
    def batch_name(self) -> str:
        """Return the logical batch name."""
        ...

    @property
    def batch_storage_name(self) -> str:
        """Return the canonical batch storage locator."""
        ...

    @property
    def campaign_run_id(self) -> str | None:
        """Return the partial campaign identity when applicable."""
        ...

    @property
    def campaign_state(self) -> str | None:
        """Return the partial campaign state when applicable."""
        ...

    @property
    def campaign_sources(self) -> tuple[tuple[str, str], ...]:
        """Return every campaign run and state contributing partial evidence."""
        ...

    @property
    def generation_root(self) -> Path:
        """Return the canonical Generation root."""
        ...

    @property
    def interpreter_batch(self) -> Any:
        """Return the exact admitted batch view for Dataset interpretation."""
        ...

    @property
    def cases(self) -> tuple[Any, ...]:
        """Return ordered individually admitted cases."""
        ...

    @property
    def available_case_count(self) -> int:
        """Return the admitted valid-case count."""
        ...

    @property
    def discovered_case_count(self) -> int:
        """Return complete candidate-case accounting."""
        ...

    @property
    def failed_case_count(self) -> int:
        """Return authoritative failed-case count."""
        ...

    @property
    def incomplete_case_count(self) -> int:
        """Return running or incomplete-case count."""
        ...

    @property
    def invalid_case_count(self) -> int:
        """Return corrupt or identity-invalid case count."""
        ...


def _transient_dataframe(
    loaded: dict[str, Any],
    *,
    batch_name: str,
    task: TaskSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one nested completed-case frame without flattening trajectories."""
    from . import eda_transient as transient  # noqa: PLC0415

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
            "total_discovered_case_count": int(loaded.get("total_discovered_case_count", available)),
            "failed_case_count": int(loaded.get("failed_case_count", 0)),
            "incomplete_case_count": int(loaded.get("incomplete_case_count", 0)),
            "invalid_case_count": int(loaded.get("invalid_case_count", 0)),
            "exclusion_reasons": dict(loaded.get("exclusion_reasons", {})) | ({} if excluded == 0 else {"bounded_prefix": excluded}),
            "case_accounting_scope": str(loaded.get("case_accounting_scope", "admitted_terminal_manifest")),
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
            "total_discovered_case_count": int(loaded.get("total_discovered_case_count", available)),
            "failed_case_count": int(loaded.get("failed_case_count", 0)),
            "incomplete_case_count": int(loaded.get("incomplete_case_count", 0)),
            "invalid_case_count": int(loaded.get("invalid_case_count", 0)),
            "exclusion_reasons": dict(loaded.get("exclusion_reasons", {})) | ({} if excluded == 0 else {"bounded_prefix": excluded}),
            "case_accounting_scope": str(loaded.get("case_accounting_scope", "admitted_terminal_manifest")),
            "analysis_representation": str(loaded.get("analysis_representation", "steady_field_rows")),
            "dataset_backend": str(loaded.get("dataset_backend", "canonical_generation_hdf5")),
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


def generate_eda_dataframe_from_completed_cases(
    batch: _CompletedCaseBatch,
    *,
    task: TaskSpec,
    show_progress: bool = True,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Materialize one bounded EDA frame from independently admitted completed cases.

    Parameters
    ----------
    batch : completed-case batch evidence
        Case-local Generation evidence assembled by the EDA source catalog.
    task : TaskSpec
        Requested task contract.
    show_progress : bool, optional
        Enable bounded per-case progress feedback.
    max_cases : int | None, optional
        Positive per-batch manifest-order prefix bound.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Task-aware EDA frame and concise source-loading logs.

    Notes
    -----
    This path intentionally does not construct a terminal batch manifest or a
    Dataset generated-batch identity. It is an EDA-only view over case-local
    Generation admission evidence.

    """
    del show_progress  # Completed-case EDA avoids notebook progress side effects.
    if batch.source_kind != "partial":
        message = "Case-local EDA materialization accepts partial campaign evidence only."
        raise ValueError(message)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        message = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise ValueError(message)
    if not batch.cases:
        message = f"Generated-output EDA batch {batch.batch_id!r} has no admitted completed cases."
        raise ValueError(message)
    from src import datasets  # noqa: PLC0415

    selected = batch.cases if max_cases is None else batch.cases[:max_cases]
    rows: list[dict[str, Any]] = []
    interpreter_batch = batch.interpreter_batch
    for case in selected:
        if task.id == "transient_drying":
            rows.append(datasets.packages.generated_batch.interpret_generated_transient_case(interpreter_batch, case, task=task))
        elif task.id == "steady_flow":
            _shape, inputs, outputs, metadata, _source, _fingerprint = datasets.packages.generated_batch.interpret_generated_case(
                interpreter_batch,
                case,
                task=task,
            )
            rows.append(
                {
                    **{name: inputs[index].numpy() for index, name in enumerate(task.input_names)},
                    **{name: outputs[index].numpy() for index, name in enumerate(task.output_names)},
                    "meta": metadata,
                }
            )
        else:
            message = f"Generated-output EDA does not support task {task.id!r}."
            raise ValueError(message)
    available = batch.available_case_count
    exclusion_reasons = {} if batch.invalid_case_count == 0 else {"invalid_or_corrupt": batch.invalid_case_count}
    loaded = {
        "sample_ids": [case.case_id for case in selected],
        "rows": rows,
        "available_case_count": available,
        "total_discovered_case_count": batch.discovered_case_count,
        "failed_case_count": batch.failed_case_count,
        "incomplete_case_count": batch.incomplete_case_count,
        "invalid_case_count": batch.invalid_case_count,
        "exclusion_reasons": exclusion_reasons,
        "case_accounting_scope": "individually_admitted_campaign_cases",
        "task": task,
        "generated_batch_identity": {
            "source_kind": "individually_admitted_campaign_cases",
            "campaign_run_id": batch.campaign_run_id,
            "campaign_state": batch.campaign_state,
            "campaign_sources": [{"run_id": run_id, "state": state} for run_id, state in batch.campaign_sources],
            "batch_id": batch.batch_id,
            "case_ids": [case.case_id for case in batch.cases],
        },
        "manifest_sha256": None,
        "generation_root": batch.generation_root,
        "analysis_representation": ("steady_field_rows" if task.id == "steady_flow" else "transient_complete_case_rows"),
        "dataset_backend": "canonical_generation_hdf5",
    }
    if task.id == "transient_drying":
        return _transient_dataframe(loaded, batch_name=batch.batch_storage_name, task=task)
    return _steady_dataframe(loaded, batch_name=batch.batch_storage_name, task=task)


def generate_eda_dataframe(
    batch_name: str,
    *,
    task: TaskSpec,
    storage_root: str | Path | None = None,
    show_progress: bool = True,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Materialize one validated manifest-ordered generated-batch prefix.

    The shared generated-source reader enforces the same manifest hashes, unit
    headers, Cartesian grid, finite values, porosity, permeability, transforms,
    and task field ordering as the final dataset builder. No model-training path
    is resolved or opened.
    """
    from src import datasets  # noqa: PLC0415

    loaded: dict[str, Any] = datasets.packages.generated_batch.load_generated_batch(
        batch_name,
        task_id=task.id,
        storage_root=storage_root,
        show_progress=show_progress,
        max_cases=max_cases,
    )
    loaded_task = loaded["task"]
    if loaded_task.contract_digest != task.contract_digest:
        msg = f"Generated reader task contract does not match requested task {task.id!r}."
        raise ValueError(msg)
    if task.id == "transient_drying":
        return _transient_dataframe(loaded, batch_name=batch_name, task=task)
    return _steady_dataframe(loaded, batch_name=batch_name, task=task)
