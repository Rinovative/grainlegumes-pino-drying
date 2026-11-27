"""Run one field-test member as one case step: gate check, roster-local binding, bundle, COMSOL, and seal.

Responsibilities:
  - Check the roster, the member entry its submission binds, and for a variant its base's successful gate
  - Rebuild the base batch from the Source's frozen plan and generate the member's bundle with its declared intervention
  - Verify the roster-local binding: base digests, seeds and primitive values exactly, and the regenerated inputs the
    intervention leaves untouched against the published base inputs within a last-digit tolerance
  - Prepare a workspace from the explicit bundle, run the unchanged execute_prepared_case with the in-allocation licence window off
  - Keep the canonical case.h5 in the attempt, refuse a gate whose output diverges structurally from its published base,
    and record the outcome

Design principles:
  - The member lock covers the whole attempt, from the gate check to the seal; a sealed member never starts another attempt
  - Every failure becomes a classified outcome: licence, I/O and identity errors are infrastructure stops, never solver
    failures or broken bindings, and only a typed infeasible intervention is infeasible
  - Every attempt write path (bundle, base_regenerated, runtime, case.h5) passes the record's symbolic-link check; a
    refused one ends the attempt as ``binding_refused`` (``record_path``) before anything is written through it, except
    a runtime evidence copy, which is noted as missing in the outcome without changing the solve's classification
  - The CPU model is recorded as evidence only: inputs differ between ICE's CPU types only in their last digits (docs/07)
  - Nothing is written below 01_generation/: the Source is only read, through digests the roster binds

This module does NOT:
  - Submit node jobs, reconcile the scheduler, or decide which members run
  - Publish Sources, admit Dataset inputs, or replace a member
  - Judge numeric output differences; the analysis measures a gate's replicate envelope against the tolerances
"""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import socket
import sys
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np

from src import common
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_dispatch_identity as dispatch_identity
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_node as node_service
from src.generation.runtime import generation_runtime_preparation as preparation_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

from . import generation_field_tests_outputs as outputs
from . import generation_field_tests_record as record
from . import generation_field_tests_roster as roster_service

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.generation.cases import generation_cases_config as config_contract

# numpy SIMD loops and OpenBLAS change generated inputs in their last digits between ICE's CPU types (docs/07, Producer
# identity). Reproduced means: a field map within 1e-12 * max|published map|, each scalar within 1e-12 * |published scalar|
# (so zero stays zero), and each schedule column within 1e-12 * max|published column|.
_INPUT_RELATIVE_TOLERANCE: Final = 1e-12


class _Stop(Exception):  # noqa: N818 -- a control-flow carrier of one classified outcome, not an error type
    """Carry one classified non-success outcome out of the step."""

    def __init__(self, message: str, *, kind: str, code: str, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.classification = kind
        self.reason_code = code
        self.evidence = dict(evidence or {})


def _attempt_path(attempt_root: Path, *parts: str) -> Path:
    """Return one write path inside an attempt; a refused component, such as a symbolic link, ends the attempt."""
    try:
        return record.record_path(attempt_root, *parts)
    except ValueError as error:
        raise _Stop(str(error), kind="binding_refused", code="record_path") from error


def _require_gate(directory: Path, member: Mapping[str, Any]) -> None:
    """Refuse a variant whose base gate is not sealed successful with inputs reproduced within the last-digit tolerance."""
    if member["role"] == "gate":
        return
    gate_id = str(member["gate_member_id"])
    gate = record.read_seal(directory, gate_id)
    if gate is None:
        message = f"Gate {gate_id} of this base is not sealed."
        raise _Stop(message, kind="gate_blocked", code="gate_missing")
    if not record.gate_succeeded(gate):
        message = f"Gate {gate_id} sealed {gate['disposition']} without reproduced inputs."
        raise _Stop(message, kind="gate_blocked", code="gate_failed")


def _require_binding(
    payload: Mapping[str, Any], base_payload: Mapping[str, Any], member: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]]
) -> None:
    """Refuse a bundle whose seeds, design row, or undeclared primitive values differ from the base; scalars compare exactly."""
    if payload["seed_evidence"] != base_payload["seed_evidence"] or payload["block_provenance"] != base_payload["block_provenance"]:
        message = "Generated seeds or design row differ from the published base."
        raise _Stop(message, kind="binding_refused", code="seed_mismatch")
    try:
        sampling_service.require_intervention_scope(
            registry, base_payload["sampled_values"], payload["sampled_values"], declared=frozenset(member["intervention"])
        )
    except sampling_service.InterventionScopeError as error:
        raise _Stop(str(error), kind="binding_refused", code="undeclared_change") from error


def _input_arrays(content: case_service.StoredCaseInputs) -> dict[str, np.ndarray]:
    """Return every stored input as one named float64 array: each field map, the scalar values, and the schedule."""
    arrays = {f"fields/{name}": np.asarray(values, dtype=np.float64) for name, values in content.fields.items()}
    arrays["scalars"] = np.asarray([value for _name, value, _unit in content.scalars], dtype=np.float64)
    if content.schedule is not None:
        arrays["schedule"] = np.asarray(content.schedule, dtype=np.float64)
    return arrays


def _input_exceeds(name: str, regenerated: np.ndarray, published: np.ndarray) -> bool:
    """Return whether one regenerated input array leaves the last-digit tolerance of its kind (map, scalars, or schedule)."""
    delta, scale = np.abs(regenerated - published), np.abs(published)
    if name == "scalars":
        return bool(np.any(delta > _INPUT_RELATIVE_TOLERANCE * scale))
    if name == "schedule":
        return bool(np.any(delta.max(axis=0) > _INPUT_RELATIVE_TOLERANCE * scale.max(axis=0)))
    return bool(delta.size) and bool(delta.max() > _INPUT_RELATIVE_TOLERANCE * scale.max())


def _compare_inputs(
    regenerated: Mapping[str, np.ndarray], published: Mapping[str, np.ndarray], names: list[str]
) -> tuple[dict[str, float], list[str]]:
    """Return the maximum absolute difference per named array and the arrays beyond the last-digit tolerance or of another shape."""
    differences: dict[str, float] = {}
    exceeded: list[str] = []
    for name in names:
        left, right = regenerated[name], published.get(name)
        if right is None or left.shape != right.shape:
            exceeded.append(name)
            continue
        differences[name] = float(np.max(np.abs(left - right))) if left.size else 0.0
        if _input_exceeds(name, left, right):
            exceeded.append(name)
    return differences, exceeded


def _require_reproduced_inputs(
    config: config_contract.GenerationConfig,
    case_index: int,
    member: Mapping[str, Any],
    bundle: Any,
    base_inputs: Path,
    base_payload: Mapping[str, Any],
    attempt_root: Path,
) -> dict[str, Any]:
    """
    Compare the regenerated inputs the intervention leaves untouched with the published base inputs within the tolerance.

    A gate compares every input. A variant also regenerates its base on this node: arrays that differ bitwise between the
    two bundles are the ones the intervention touched; every other array must match the published base within tolerance.
    """
    published_content = case_service.read_case_inputs(base_inputs, base_payload)
    regenerated_content = case_service.read_case_inputs(bundle.directory, bundle.case_payload)
    published, regenerated = _input_arrays(published_content), _input_arrays(regenerated_content)
    touched: list[str] = []
    if member["role"] != "gate":
        base_bundle = _generate_bundle(config, case_index, {**member, "intervention": {}}, _attempt_path(attempt_root, "base_regenerated"))
        unchanged = _input_arrays(case_service.read_case_inputs(base_bundle.directory, base_bundle.case_payload))
        shutil.rmtree(base_bundle.directory)
        touched = sorted(name for name in regenerated if name not in unchanged or not np.array_equal(regenerated[name], unchanged[name]))
    differences, exceeded = _compare_inputs(regenerated, published, [name for name in sorted(regenerated) if name not in touched])
    if "scalars" not in touched and [item[0] for item in regenerated_content.scalars] != [item[0] for item in published_content.scalars]:
        exceeded.append("scalars")
    comparison = {"relative_tolerance": _INPUT_RELATIVE_TOLERANCE, "max_abs_diff": differences, "touched": touched}
    if exceeded:
        message = f"Regenerated inputs {sorted(set(exceeded))} differ from the published base beyond the last-digit tolerance."
        code = "gate_inputs_differ" if member["role"] == "gate" else "untouched_inputs_differ"
        raise _Stop(message, kind="binding_refused", code=code, evidence={"input_comparison": comparison})
    return comparison


def _numeric_datasets(handle: h5py.File) -> list[str]:
    """Return the paths of every numeric dataset of one case.h5."""
    names: list[str] = []

    def visit(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
            names.append(name)

    handle.visititems(visit)
    return names


def _compare_outputs(produced: Path, published: Path) -> tuple[dict[str, float], list[str]]:
    """
    Return the largest finite difference per shared numeric dataset and every structural divergence from the published case.

    Divergences: a numeric dataset missing or extra, another shape, a non-finite value where the published one is finite,
    and other completion evidence (the reached flag, or the stored hour axis beyond the published file's time classification
    tolerance). A drift of the exact stop time is no divergence by itself: the transient solve does not reproduce it bitwise,
    and the analysis measures it in the replicate envelope as the duration when both cases reached the target. A drift that
    changes the layout (the exact-stop datasets, the diagnostics rows or the stored hours) still diverges through the dataset
    and shape checks.
    """
    differences: dict[str, float] = {}
    divergences: list[str] = []
    with h5py.File(produced, "r") as left, h5py.File(published, "r") as right:
        mine, theirs = set(_numeric_datasets(left)), set(_numeric_datasets(right))
        divergences += [f"missing:{name}" for name in sorted(theirs - mine)] + [f"extra:{name}" for name in sorted(mine - theirs)]
        for name in sorted(mine & theirs):
            a, b = (np.asarray(outputs.dataset(handle, name)[()], dtype=np.float64) for handle in (left, right))
            if a.shape != b.shape:
                divergences.append(f"shape:{name}")
                continue
            if bool(np.any(~np.isfinite(a) & np.isfinite(b))):
                divergences.append(f"nonfinite:{name}")
            delta = np.abs(a - b)[np.isfinite(a) & np.isfinite(b)]
            if delta.size:
                differences[name] = float(delta.max())
        produced_completion, published_completion = outputs.completion(left), outputs.completion(right)
        atol = float(outputs.dataset(right, "time").attrs.get("classification_atol", math.nan)) if "time" in theirs else 0.0
    if not math.isfinite(atol):
        message = f"{published} lacks a finite time classification tolerance."
        raise KeyError(message)
    if differences.get("time", 0.0) > atol:
        divergences.append("hour_axis")
    if produced_completion is not None and published_completion is not None and produced_completion.reached != published_completion.reached:
        divergences.append("reached")
    return differences, divergences


def _classify_execution_error(error: BaseException) -> tuple[str, str]:
    """Return the classification and reason code of one failure raised by preparation or execution."""
    if isinstance(error, license_service.TemporaryLicenseCapacityError):
        return "infrastructure_stop", "license_capacity"
    if isinstance(error, runtime_service.CaseInterruptedError):
        return "infrastructure_stop", "interrupted"
    if isinstance(error, runtime_service.CaseExecutionError):
        if error.runtime_reason is not None:
            return "solver_failure", error.runtime_reason
        if error.validation_reason is not None:
            return "solver_failure", error.validation_reason
        if error.exit_code is None:
            return "infrastructure_stop", "solver_not_started"
        return "solver_failure", f"{error.failure_stage}_unclassified"
    if isinstance(error, preparation_service.CasePreparationError):
        return "infrastructure_stop", "preparation_failed"
    return "infrastructure_stop", type(error).__name__.lower()


def _copy_runtime_evidence(runtime_directory: Path, evidence_directory: Path) -> list[str]:
    """Copy the runtime files into the attempt; a refused or failed copy becomes a returned note, never a raised error."""
    notes: list[str] = []
    for path in sorted(runtime_directory.glob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                shutil.copyfile(path, _attempt_path(evidence_directory, "runtime", path.name))
            except (_Stop, OSError) as error:
                notes.append(f"runtime evidence missing: {path.name} ({error})")
    return notes


def _execute(
    config: config_contract.GenerationConfig,
    bundle_directory: Path,
    *,
    cores: int,
    storage_root: Path,
    work_root: Path | str | None,
    evidence_directory: Path,
    notes: list[str],
) -> Path:
    """
    Prepare and run one explicit bundle with the in-allocation licence window off; return the canonical case.h5 in the record.

    A runtime file that cannot be copied into the attempt is appended to ``notes`` as missing evidence; it never replaces
    the solve's result or failure, and the scratch cleanup always runs.
    """
    execution = dict(config.execution_values)
    execution["runtime"] = {**execution["runtime"], "license_wait": {**execution["runtime"]["license_wait"], "enabled": False}}
    config = dataclasses.replace(config, execution_values=execution)
    prepared = preparation_service.prepare_bundle_work_directory(config, bundle_directory, storage_root=storage_root, work_root=work_root)
    try:
        result = runtime_service.execute_prepared_case(
            config,
            prepared,
            cores_per_case=cores,
            worker_slot=1,
            # Licence retry is off above, and campaign progress needs a dispatch identity a step never carries.
            scheduler_kind="slurm",
            allocated_node=socket.gethostname(),
        )
        target = _attempt_path(evidence_directory, "case.h5")
        shutil.copyfile(result.canonical_case.path, target)
        return target
    finally:
        notes.extend(_copy_runtime_evidence(prepared.runtime_directory, evidence_directory))
        try:
            workspace_service.cleanup_case_workspace(
                prepared.work_directory,
                allowed_root=prepared.work_root,
                storage_root=prepared.storage_root,
                expected_run_id=prepared.workspace_run_id,
                expected_case_id=prepared.bundle.case_id,
                allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Field-test scratch cleanup failed for {prepared.work_directory}: {error}", file=sys.stderr)


def _generate_bundle(config: config_contract.GenerationConfig, case_index: int, member: Mapping[str, Any], target: Path) -> Any:
    """Generate the member's bundle; only a typed infeasible intervention is infeasible, any other refusal is a broken binding."""
    try:
        return case_service.generate_case_input_bundle(config, case_index, target, intervention=member["intervention"] or None)
    except sampling_service.InfeasibleInterventionError as error:
        raise _Stop(str(error), kind="infeasible", code="intervention_infeasible") from error
    except sampling_service.InterventionError as error:
        raise _Stop(str(error), kind="binding_refused", code="intervention_refused") from error
    except OSError as error:
        raise _Stop(str(error), kind="infrastructure_stop", code="bundle_io") from error
    except (ValueError, RuntimeError) as error:
        raise _Stop(str(error), kind="binding_refused", code="bundle_generation") from error


def _attempt(
    directory: Path,
    roster: Mapping[str, Any],
    member: Mapping[str, Any],
    attempt: int,
    *,
    submission: Mapping[str, Any],
    cores: int,
    storage_root: Path,
    work_root: Path | str | None,
    fields: dict[str, Any],
) -> None:
    """Run every stage of one attempt, filling ``fields`` with the outcome evidence; raise ``_Stop`` for a non-success."""
    task = str(roster["task"])
    member_id = str(member["member_id"])
    attempt_root = record.attempt_directory(directory, member_id, attempt)
    if member["role"] == "gate":
        fields["gate"] = {"inputs_equal": False, "output_max_abs_diff": {}, "output_divergences": None}
    if submission.get("member_sha256") != roster_service.member_sha256(member):
        message = f"Member {member_id!r} differs from the entry its submission binds (roster version {submission.get('roster_version')})."
        raise _Stop(message, kind="binding_refused", code="roster_binding")
    _require_gate(directory, member)
    if os.environ.get(dispatch_identity.CAMPAIGN_RUN_ID_ENVIRONMENT) is not None:
        message = f"{dispatch_identity.CAMPAIGN_RUN_ID_ENVIRONMENT} is set; a field-test step never runs as a campaign."
        raise _Stop(message, kind="infrastructure_stop", code="campaign_identity_present")
    try:
        base_payload, base_case_h5 = roster_service.verified_base_payload(task, roster, member["base"], storage_root=storage_root)
        campaign = roster_service.load_base_campaign(task, roster["base_source"], storage_root=storage_root)
        config, case_index = roster_service.base_batch(campaign, member["base"])
    except OSError as error:
        raise _Stop(str(error), kind="infrastructure_stop", code="base_io") from error
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        raise _Stop(str(error), kind="binding_refused", code="base_binding") from error
    bundle_directory = _attempt_path(attempt_root, "bundle")
    bundle = _generate_bundle(config, case_index, member, bundle_directory)
    payload = bundle.case_payload
    fields["bundle_sha256"] = bundle.simulation_case_id
    fields["spatial_support_attempt"], fields["schedule_acceptance_attempt"] = roster_service.acceptance_attempts(payload)
    _require_binding(payload, base_payload, member, config.scientific_values["material"]["parameter_registry"])
    base_inputs = roster_service.base_inputs_directory(base_case_h5)
    try:
        fields["input_comparison"] = _require_reproduced_inputs(config, case_index, member, bundle, base_inputs, base_payload, attempt_root)
    except OSError as error:
        raise _Stop(str(error), kind="infrastructure_stop", code="input_io") from error
    except ValueError as error:
        raise _Stop(str(error), kind="binding_refused", code="input_comparison") from error
    if member["role"] == "gate":
        fields["gate"]["inputs_equal"] = True
    _attempt_path(attempt_root, "runtime").mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    fields["notes"] = notes
    try:
        produced = _execute(
            config,
            bundle_directory,
            cores=cores,
            storage_root=storage_root,
            work_root=work_root,
            evidence_directory=attempt_root,
            notes=notes,
        )
    except _Stop:
        raise
    except Exception as error:
        kind, code = _classify_execution_error(error)
        raise _Stop(str(error), kind=kind, code=code) from error
    if member["role"] == "gate":
        try:
            fields["gate"]["output_max_abs_diff"], fields["gate"]["output_divergences"] = _compare_outputs(produced, base_case_h5)
        except OSError as error:
            raise _Stop(str(error), kind="infrastructure_stop", code="gate_comparison") from error
        except (KeyError, TypeError, ValueError) as error:
            raise _Stop(str(error), kind="binding_refused", code="gate_outputs_differ") from error
        if fields["gate"]["output_divergences"]:
            message = f"The gate's output diverges structurally from its published base case: {fields['gate']['output_divergences']}"
            raise _Stop(message, kind="binding_refused", code="gate_outputs_differ")
    # The output stays in the attempt; settling links it into the member record only after the outcome is committed.
    fields["case_h5_sha256"] = common.serialization.file_sha256(produced)


def run_member(
    task: str,
    test_id: str,
    member_id: str,
    *,
    attempt: int,
    cores: int,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """
    Run one attempt of one field-test member under its member lock and record its outcome.

    Parameters
    ----------
    task, test_id, member_id
        The field test and its roster member.
    attempt : int
        The attempt number its submission record names; a step without that record refuses to start.
    cores : int
        Cores of the case step.
    storage_root, work_root : Path | str | None, optional
        Storage root and disposable scratch root (default: the Slurm job's TMPDIR).

    Returns
    -------
    dict | None
        The member's seal when it is final after this attempt (or was already sealed), else None (a retry is due).

    Raises
    ------
    FileNotFoundError, FileExistsError
        If the attempt has no submission record or already has an outcome.
    common.locking.FileLockUnavailableError
        If another process holds the member.

    """
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage)
    roster = roster_service.load_roster(directory)
    member = roster_service.member(roster, member_id)
    with record.member_lock(directory, member_id):
        sealed = record.read_seal(directory, member_id)
        if sealed is not None:
            return sealed
        submission = record.read_submission(directory, member_id, attempt)
        if submission is None:
            message = f"Member {member_id!r} attempt {attempt} has no submission record."
            raise FileNotFoundError(message)
        if record.read_outcome(directory, member_id, attempt) is not None:
            message = f"Member {member_id!r} attempt {attempt} already has an outcome."
            raise FileExistsError(message)
        fields: dict[str, Any] = {
            "cpu_model": node_service.read_cpu_model(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "started_at": record.utc_now(),
        }
        classification = "succeeded"
        try:
            _attempt(
                directory,
                roster,
                member,
                attempt,
                submission=submission,
                cores=cores,
                storage_root=storage,
                work_root=work_root,
                fields=fields,
            )
        except _Stop as stop:
            classification = stop.classification
            fields.update(stop.evidence)
            fields["failure"] = {"reason_code": stop.reason_code, "message": str(stop)}
            fields["case_h5_sha256"] = None
        outcome = record.new_outcome(member_id, attempt, classification, recorded_by="step", **fields)
        return record.record_outcome(directory, roster, member, outcome)
