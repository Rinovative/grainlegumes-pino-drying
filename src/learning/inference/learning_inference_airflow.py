"""Apply a completed steady run to one Generation case of either task and compare its airflow with COMSOL.

Responsibilities:
  - Admit one successful Generation case of either task by Source, material and case ID through the Dataset planner
  - Convert it into steady model inputs and reference fields with the steady Dataset builder's own conversion
  - Refuse a case whose coordinate grid or stationary conditioning differs from the run's Train Dataset
  - Predict its airflow on the full canonical grid with the run's model and Train scaling, refusing non-finite values
  - Return reference, prediction and error in u, v, p with the relative L2 per field of the shared case metric

Design principles:
  - Admission, case ID, grid identity, conditioning check, field transforms, scaling, metric and floors are their
    owners'; this module only wires them for one case
  - Inference follows the steady artifact generator: the full canonical grid whatever the training stride, encoded
    inputs, a forward in evaluation mode, decoded states
  - Channels are selected by field name, never by position

This module does NOT:
  - Load the run: ``learning.inference.context.load_steady_model`` owns that, once for many cases
  - Select a spatial view: a consumer such as the transient inference selects its own grid
  - Run the transient model or chain steady and transient inference
  - Publish artifacts, plots or any other file

Notes:
  Output contract:
    reference, prediction and error are finite CPU float32 tensors [3, Y, X] in the order u, v, p, in m/s and Pa,
    on the case's full canonical Generation grid, like the steady artifacts.
    ``prediction.unsqueeze(0)`` is one batch item of the transient inference's ``external_airflow``.
  Train evidence:
    The run's Train scaling binds the Train Dataset by ID and manifest digest. Its manifest records the grid; its
    first source case, re-admitted through the planner, carries the stationary conditioning the planner required of
    every Train case.
  The selection requests are only admitted, never published; their Dataset name satisfies the request schema.

"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import torch

from src import datasets, domain, learning
from src.datasets.packages import dataset_packages_selection as selection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from src.datasets.packages.dataset_packages_planning import PreparedPackage

    from .learning_inference import SteadyModelContext

AIRFLOW_FIELDS: Final = ("u", "v", "p")
_SELECTION_NAME: Final = "steady_airflow_case"


@dataclass(frozen=True, slots=True)
class CaseAirflow:
    """
    Hold the steady airflow of one Generation case beside its COMSOL reference.

    Attributes
    ----------
    run_name : str
        Scientific name of the steady run that predicted the airflow.
    package_case_id : str
        The Dataset builder's case ID of the Generation case.
    source_identity : Mapping[str, Any]
        The Generation catalog's exact identity of the case, including task, Source, material, case ID and HDF5 hash.
    fields : tuple[str, ...]
        Channel names of the arrays, ``AIRFLOW_FIELDS``.
    reference, prediction, error : torch.Tensor
        COMSOL fields, steady prediction and prediction minus reference, CPU float32 ``[3, Y, X]`` in physical units.
    relative_l2 : Mapping[str, float | None]
        Relative L2 per field, ``None`` where the reference lies below the field's floor.

    """

    run_name: str
    package_case_id: str
    source_identity: Mapping[str, Any]
    fields: tuple[str, ...]
    reference: torch.Tensor
    prediction: torch.Tensor
    error: torch.Tensor
    relative_l2: Mapping[str, float | None]


def _admit_selection(task: str, references: Sequence[Mapping[str, str]], *, storage_root: Path | str | None) -> PreparedPackage:
    """Admit exact successful Generation cases as one unpublished steady-view selection of the Dataset planner."""
    request = {
        "schema_kind": selection.SELECTION_SCHEMA_KIND,
        "source_task": task,
        "dataset_name": _SELECTION_NAME,
        "dataset_view": datasets.preprocessing.normalization.STEADY_TASK,
        "selection": {"kind": "exact"},
        "source_cases": [dict(reference) for reference in references],
    }
    return datasets.packages.planning.prepare_dataset_selection(request, storage_root=storage_root)


def _conditioning_digest(prepared: PreparedPackage) -> str:
    """Return the stationary-conditioning digest the planner validated for the selection's first case."""
    return str(prepared.cases[0].case.metadata_payload()["steady_flow_conditioning_digest"])


def _require_train_evidence(steady: SteadyModelContext, case: PreparedPackage, *, storage_root: Path | str | None) -> None:
    """Refuse a case whose grid or stationary conditioning is not that of the Train Dataset the run's scaling binds."""
    train = steady.scaling.dataset_identity
    manifest = datasets.packages.load_package_manifest_evidence(
        datasets.preprocessing.normalization.STEADY_TASK, str(train["dataset_id"]), storage_root=storage_root
    )
    if manifest["dataset_digest"] != train["dataset_digest"]:
        msg = f"Train Dataset {train['dataset_id']!r} of run {steady.run_name} no longer has the manifest digest the run recorded."
        raise RuntimeError(msg)
    if dict(manifest["grid"]) != dict(case.grid):
        msg = f"Case grid {dict(case.grid)} differs from the grid {manifest['grid']} run {steady.run_name} was trained on."
        raise ValueError(msg)
    reference = {key: manifest["source_case_identities"][0]["source"][key] for key in selection.SOURCE_REFERENCE_ORDER}
    if _conditioning_digest(_admit_selection(manifest["source_task"], [reference], storage_root=storage_root)) != _conditioning_digest(case):
        msg = f"Case stationary conditioning (T_flow_ref, p_ref, p_out) differs from that of run {steady.run_name}'s Train Dataset."
        raise ValueError(msg)


def predict_case_airflow(
    steady: SteadyModelContext,
    *,
    task: str,
    source_name: str,
    material_family: str,
    case_id: str,
    storage_root: Path | str | None = None,
) -> CaseAirflow:
    """
    Predict the steady airflow of one Generation case and compare it with the case's COMSOL airflow.

    Parameters
    ----------
    steady : SteadyModelContext
        Completed steady run from ``learning.inference.context.load_steady_model``.
    task : str
        Generation task of the case, ``"steady_flow"`` or ``"transient_drying"``.
    source_name, material_family, case_id : str
        The case's Generation Source, material and case folder.
    storage_root : Path | str | None, optional
        Storage root holding the Generation case, the run's Train Dataset and its Sources; defaults to the central
        resolution.

    Returns
    -------
    CaseAirflow
        Reference, prediction, error and relative L2 in the order u, v, p on the full canonical grid.

    Raises
    ------
    ValueError
        If the case is not one successful case offering the steady view, its coordinate grid or stationary
        conditioning differs from the run's Train Dataset, or the run predicts non-finite values for it.
    RuntimeError
        If the run's Train Dataset manifest changed since training.

    """
    steady_task = domain.tasks.registry.get_task(datasets.preprocessing.normalization.STEADY_TASK)
    pick = {"source_name": source_name, "material_family": material_family, "case_id": case_id}
    prepared = _admit_selection(task, datasets.packages.select_exact_cases(task, [pick], storage_root=storage_root), storage_root=storage_root)
    _require_train_evidence(steady, prepared, storage_root=storage_root)
    entry = prepared.source_case_identities[0]
    _shape, inputs, outputs, _metadata, _source, _fingerprint = datasets.packages.generated_batch.interpret_admitted_source_case(
        prepared.cases[0], package_case_id=entry["package_case_id"], task=steady_task
    )
    model_inputs = inputs[[steady_task.input_names.index(name) for name in steady.input_names]].unsqueeze(0).to(steady.device)
    steady.model.eval()
    with torch.no_grad():
        state_reference = steady.scaling.state_reference(model_inputs)
        encoded = steady.model(steady.scaling.encode_inputs(model_inputs, reference=state_reference))
        decoded = steady.scaling.decode_state(encoded, reference=state_reference)[0].cpu()
    if not bool(torch.isfinite(decoded).all()):
        msg = f"Run {steady.run_name} predicts non-finite airflow for case {entry['package_case_id']}."
        raise ValueError(msg)
    prediction = decoded[[steady.output_names.index(name) for name in AIRFLOW_FIELDS]]
    reference = outputs[[steady_task.output_names.index(name) for name in AIRFLOW_FIELDS]]
    sums = learning.metrics.metrics.field_statistics(prediction.unsqueeze(0), reference.unsqueeze(0))[0]
    relative_l2 = {
        name: learning.metrics.metrics.case_relative_norm(sums[index], norm="l2", floor=steady_task.reference_floor(name))
        for index, name in enumerate(AIRFLOW_FIELDS)
    }
    return CaseAirflow(
        run_name=steady.run_name,
        package_case_id=str(entry["package_case_id"]),
        source_identity=copy.deepcopy(entry["source"]),
        fields=AIRFLOW_FIELDS,
        reference=reference,
        prediction=prediction,
        error=prediction - reference,
        relative_l2=relative_l2,
    )
