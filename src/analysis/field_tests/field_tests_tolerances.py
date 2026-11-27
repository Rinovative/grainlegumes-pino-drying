"""Build, freeze and admit the field-test tolerance table from trained runs on their validation population.

Responsibilities:
  - Pair each validation case's Source reference with a trained run's prediction through the maintained inference
    services: the steady airflow service and the transient autonomous rollout
  - Exclude non-finite predictions and physically invalid rollouts (the maintained transient validity classification)
    and count them per reason
  - Measure the envelope endpoints of the valid pairs with the field-test endpoint code (every endpoint and the increment
    windows at every shared window end) and take the median per endpoint
  - Record each model run's admitted identity (its path below STORAGE_ROOT, effective config digest and selected
    checkpoint digest), the cases and excluded cases by run path, and the endpoint code digest with the table
  - Write the table once and refuse a changed table or a table of other endpoint code on admission

Design principles:
  - The yardstick uses the same endpoint function, masks, units and stored windows as the field-test comparisons
  - Only the validation membership is used; test and OOD cases never set a tolerance
  - A run's validation Sources and steady predictions are admitted in one batch call each, so campaign-wide reads do not
    grow with the case count while every case keeps its own hash and admission checks
  - An invalid prediction is excluded and recorded, never repaired; censored durations never set a tolerance
  - A run named twice (by its resolved admitted path) is refused, so no validation case counts twice in a median
  - The freeze is write-once; its self digest and file digest bind every verdict built on it, in stage B too

This module does NOT:
  - Train models, build analysis artifacts or choose which runs form the yardstick
  - Decide verdicts or read field-test members
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from src import common, datasets, experiments, generation, learning
from src.analysis.artifacts import analysis_artifact_transient as artifact_transient
from src.analysis.evaluation import evaluation_transient_rollout as rollout
from src.analysis.evaluation import evaluation_transient_validity as validity

from . import field_tests_endpoints as endpoints

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

TOLERANCES_SCHEMA_KIND: Final = "field_test_tolerances"
SCHEMA_VERSION: Final = 1
TOLERANCES_FILENAME: Final = "tolerances.json"
_SOURCE_SELECTOR_KEYS: Final = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)
_REFERENCE_RTOL: Final = 1.0e-6


@dataclass(frozen=True, slots=True)
class ValidationPair:
    """One validation case of one run: the run's admitted identity, the Source reference and the run's prediction on its grid and hours."""

    run: Mapping[str, str]
    case_id: str
    reference: endpoints.CaseFields
    prediction: endpoints.CaseFields


@dataclass(frozen=True, slots=True)
class FrozenTolerances:
    """An admitted frozen tolerance table, its file digest and its tolerance per endpoint."""

    payload: Mapping[str, Any]
    sha256: str
    tolerances: Mapping[str, float]


def _validation_sources(
    split: datasets.preprocessing.splits.SplitContract, *, storage_root: Path | str | None
) -> tuple[str, dict[str, tuple[Mapping[str, Any], Path]]]:
    """Return the Generation task and the Source identity and ``case.h5`` of every validation member, re-admitted in one call."""
    part = split.role("validation")
    manifest = datasets.packages.load_package_manifest_evidence(split.task, part.dataset_id, storage_root=storage_root)
    by_case = {entry["package_case_id"]: entry for entry in manifest["source_case_identities"]}
    sources = [by_case[case_id]["source"] for case_id in part.case_ids]
    admitted = generation.publication.catalog.resolve_raw_case_references(
        manifest["source_task"], [{key: source[key] for key in _SOURCE_SELECTOR_KEYS} for source in sources], storage_root=storage_root
    )
    return str(manifest["source_task"]), {
        case_id: (source, case.hdf5_path) for case_id, source, case in zip(part.case_ids, sources, admitted, strict=True)
    }


def _run_identity(completed: Mapping[str, Any], *, storage_root: Path | str | None) -> dict[str, str]:
    """
    Return an admitted run's identity: its path below STORAGE_ROOT, effective config digest and selected checkpoint digest.

    Raises
    ------
    ValueError
        If the run lies outside the storage root, where no path names it.

    """
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    run_dir = Path(completed["run_dir"]).resolve()
    if not run_dir.is_relative_to(storage):
        msg = f"Run {run_dir} is not below the storage root {storage}."
        raise ValueError(msg)
    return {
        "path": run_dir.relative_to(storage).as_posix(),
        "effective_config_digest": str(completed["effective_config_digest"]),
        "best_checkpoint_sha256": str(completed["selected_checkpoint_sha256"]),
    }


def steady_validation_pairs(run_dir: Path | str, *, storage_root: Path | str | None = None, device: str = "cpu") -> Iterator[ValidationPair]:
    """
    Yield each validation case of a completed steady run with its prediction from the steady airflow service.

    Raises
    ------
    ValueError
        If the service's reference airflow differs from the Source ``case.h5`` fields, or a prediction is non-finite.

    """
    run = _run_identity(experiments.run.validate_evaluable_run(run_dir), storage_root=storage_root)
    steady = learning.inference.context.load_steady_model(run_dir=run_dir, device_resolution=learning.device.resolve_device(device))
    split = datasets.preprocessing.splits.read_saved_split(run_dir)
    source_task, sources = _validation_sources(split, storage_root=storage_root)
    picks = [
        {"source_name": str(source["campaign_name"]), "material_family": str(source["material_family"]), "case_id": str(source["case_id"])}
        for source, _path in sources.values()
    ]
    airflows = learning.inference.airflow.predict_cases_airflow(steady, task=source_task, picks=picks, storage_root=storage_root)
    for (case_id, (_source, hdf5_path)), airflow in zip(sources.items(), airflows, strict=True):
        reference = endpoints.read_case_fields(hdf5_path)
        predicted: dict[str, np.ndarray] = {}
        for index, name in enumerate(airflow.fields):
            served = airflow.reference[index].numpy().astype(np.float64)
            if not np.allclose(served, reference.fields[name], rtol=_REFERENCE_RTOL, atol=0.0):
                msg = f"Steady airflow service reference of {case_id!r} differs from its Source case.h5 field {name!r}."
                raise ValueError(msg)
            predicted[name] = airflow.prediction[index].numpy().astype(np.float64)
        predicted["speed"] = np.hypot(predicted["u"], predicted["v"])
        yield ValidationPair(run, case_id, reference, endpoints.CaseFields(task="steady_flow", fields=predicted))


def _predicted_completion(case: rollout.TransientEvaluationCase, states: np.ndarray, reference: endpoints.Completion) -> endpoints.Completion | None:
    """Return the regular-grid completion of a predicted trajectory with the reference's target definition."""
    fraction = rollout.wet_fraction_series(
        states, static=case.static_conditioning, scalars=case.scalar_conditioning, target_wet_basis=reference.target_wet_basis
    )
    if not bool(np.isfinite(fraction).all()):
        return None
    reached = np.flatnonzero(fraction <= reference.fraction_limit)
    stop = int(reached[0]) if reached.size else len(fraction) - 1
    return endpoints.Completion(
        t_stop=float(case.physical_times[stop]),
        reached=bool(reached.size),
        wet_fraction_final=float(fraction[stop]),
        horizon=reference.horizon,
        fraction_limit=reference.fraction_limit,
        target_wet_basis=reference.target_wet_basis,
    )


def transient_validation_pairs(run_dir: Path | str, *, storage_root: Path | str | None = None, device: str = "cpu") -> Iterator[ValidationPair]:
    """
    Yield each validation case of a completed transient run with its full autonomous rollout on the canonical grid.

    The rollout starts from the stored initial state and runs over the stored hours, as the Evaluation artifacts'
    full autonomous record does; the table builder excludes a non-finite or physically invalid rollout.
    """
    completed = experiments.run.validate_evaluable_run(run_dir)
    run = _run_identity(completed, storage_root=storage_root)
    _training, evaluation = artifact_transient.resolve_transient_artifact_spatial_representations(completed)
    context = learning.inference.transient.load_transient_inference_context(
        run_dir=run_dir, device=device, evaluation_spatial_shape=evaluation.represented_shape
    )
    split = datasets.preprocessing.splits.admit_split(completed["split_indices"])
    part = split.role("validation")
    _source_task, sources = _validation_sources(split, storage_root=storage_root)
    request = datasets.runtime.factory.DatasetRequest(
        dataset_id=part.dataset_id,
        dataset_view="transient_drying",
        transient_sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        storage_root=storage_root,
        spatial_stride=evaluation.spatial_stride,
    )
    dataset = datasets.runtime.factory.create_dataset(request)
    if not isinstance(dataset, datasets.runtime.transient.TransientPhysicalDataset):
        msg = "The transient Dataset factory returned a non-transient runtime."
        raise TypeError(msg)
    datasets.preprocessing.splits.require_source_identity(dataset, part.dataset, role="validation", case_ids=part.case_ids)
    selected = datasets.runtime.transient.select_transient_cases(dataset, part.case_ids)
    for spec in artifact_transient.case_materialization_specs((selected,)):
        case_id, _record, case, _pt = artifact_transient.materialize_case(spec, dataset_role="id")
        reference = endpoints.read_case_fields(sources[case_id][1])
        prepared = rollout.prepare_transient_evaluation_case(context, case)
        states = learning.inference.transient.rollout_prepared_transient_autonomous(context, prepared.request).states[0].cpu().numpy()
        trajectory = np.concatenate([case.reference_states[:1], states], axis=0)
        prediction = endpoints.CaseFields(
            task="transient_drying",
            times=np.asarray(case.physical_times, dtype=np.float64),
            states=trajectory,
            # The reader refuses a transient case without completion evidence.
            completion=_predicted_completion(case, trajectory, cast("endpoints.Completion", reference.completion)),
        )
        yield ValidationPair(run, case_id, reference, prediction)


def _exclusion(prediction: endpoints.CaseFields) -> str | None:
    """Return why a prediction cannot set a tolerance: a non-finite value, or a rollout state outside the physical domains."""
    arrays = [*prediction.fields.values(), *([] if prediction.states is None else [prediction.states])]
    if not all(bool(np.isfinite(array).all()) for array in arrays):
        return "nonfinite_prediction"
    # The stored initial state is the rollout input; the maintained classification judges the predicted states after it.
    if prediction.states is not None and len(prediction.states) > 1 and bool(validity.prediction_physical_invalid_mask(prediction.states[1:]).any()):
        return "physically_invalid_rollout"
    return None


def build_tolerance_table(task: str, pairs: Iterable[ValidationPair]) -> dict[str, Any]:
    """
    Return the tolerance table: the median of every verdict endpoint over the valid validation pairs.

    A non-finite prediction or a physically invalid rollout is excluded and counted per reason. Missing values and
    censored duration bounds contribute nothing; an endpoint no pair supports has no entry, so the verdict builder treats
    it as missing. Transient increment windows enter at every shared window end. The model runs are the pairs' admitted
    run identities, and every case names its run by path. The payload carries its own canonical digest as
    ``table_sha256``.

    Raises
    ------
    ValueError
        If a pair belongs to another task, a run is named twice (its admitted path, resolved, repeats a validation case,
        which would count twice in the median), or no valid pair remains.

    """
    values: dict[str, list[float]] = defaultdict(list)
    runs: dict[str, dict[str, str]] = {}
    seen: set[tuple[str, str]] = set()
    cases: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for pair in pairs:
        if pair.reference.task != task:
            msg = f"Validation case {pair.case_id!r} belongs to {pair.reference.task}, not {task}."
            raise ValueError(msg)
        if (pair.run["path"], pair.case_id) in seen:
            msg = f"Run {pair.run['path']!r} is named twice: its validation case {pair.case_id!r} would count twice in the median."
            raise ValueError(msg)
        seen.add((pair.run["path"], pair.case_id))
        runs.setdefault(pair.run["path"], dict(pair.run))
        reason = _exclusion(pair.prediction)
        if reason is not None:
            excluded.append({"run": pair.run["path"], "case_id": pair.case_id, "reason": reason})
            continue
        cases.append({"run": pair.run["path"], "case_id": pair.case_id})
        for name, endpoint in endpoints.envelope_endpoints(pair.reference, pair.prediction).items():
            if endpoint.verdict and not endpoint.censored and endpoint.value is not None and math.isfinite(endpoint.value):
                values[name].append(endpoint.value)
    if not cases:
        msg = "No valid validation pair supports a tolerance."
        raise ValueError(msg)
    payload: dict[str, Any] = {
        "schema_kind": TOLERANCES_SCHEMA_KIND,
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "statistic": "median",
        "population": "validation",
        "endpoint_code_sha256": endpoints.endpoint_code_sha256(),
        "model_runs": list(runs.values()),
        "cases": cases,
        "excluded_cases": excluded,
        "excluded_counts": dict(sorted(Counter(item["reason"] for item in excluded).items())),
        "entries": {name: {"tolerance": float(np.median(found)), "contributors": len(found)} for name, found in sorted(values.items())},
    }
    payload["table_sha256"] = common.serialization.canonical_json_sha256(payload)
    return payload


def freeze_tolerances(path: Path | str, table: Mapping[str, Any]) -> str:
    """Write the tolerance table once and return its file digest; an existing table is never replaced."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{uuid.uuid4().hex}.staged")
    common.serialization.atomic_write_json(staged, table)
    try:
        # A hard link publishes the complete file and fails when the target exists, so concurrent freezes cannot both win.
        os.link(staged, target)
    except FileExistsError as error:
        msg = f"Tolerance table {target} is frozen; it is written once and never replaced."
        raise FileExistsError(msg) from error
    finally:
        staged.unlink(missing_ok=True)
    return common.serialization.file_sha256(target)


def load_frozen_tolerances(path: Path | str) -> FrozenTolerances:
    """
    Admit a frozen tolerance table measured with the running endpoint code.

    Raises
    ------
    ValueError
        If the schema, the self digest or the endpoint code digest does not match.

    """
    target = Path(path)
    sha256 = common.serialization.file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_kind") != TOLERANCES_SCHEMA_KIND or payload.get("schema_version") != SCHEMA_VERSION:
        msg = f"{target} is not a field-test tolerance table of schema version {SCHEMA_VERSION}."
        raise ValueError(msg)
    body = {key: value for key, value in payload.items() if key != "table_sha256"}
    if common.serialization.canonical_json_sha256(body) != payload.get("table_sha256"):
        msg = f"Tolerance table {target} changed after it was frozen."
        raise ValueError(msg)
    if payload["endpoint_code_sha256"] != endpoints.endpoint_code_sha256():
        msg = f"Tolerance table {target} was measured with other endpoint code; freeze a new table in a new field test."
        raise ValueError(msg)
    tolerances = {name: float(entry["tolerance"]) for name, entry in payload["entries"].items()}
    return FrozenTolerances(payload=payload, sha256=sha256, tolerances=tolerances)
