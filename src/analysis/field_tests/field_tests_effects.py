"""Measure the exploratory physical effect of every stage-A field-test row on top of the frozen verdicts.

Responsibilities:
  - Bind the frozen stage-A results (roster version and digest, admitted tolerance table, verdict file and its header) and
    refuse unless the verdict module's own assembly reproduces the persisted inventory, its threshold sides and row verdicts
  - Admit the yardstick runs' identities and their saved Train membership against the Dataset manifest, and measure case
    quantities of interest in physical units after checking each case's contract: transient duration, progress, moisture
    spread, exhaust air and bed temperature at the case's own drying phase, steady pressure drop, outlet velocity,
    non-uniformity and wall-core contrast
  - Reduce every persisted comparison to signed differences and effect sizes against the Train scale, with resolution
    flags from the gate replicates, a locality index per state field, phase support, and level, base and row summaries
  - Derive the exploratory disposition, merge and range flags (range claims checked against the frozen Source registry),
    family shares, stall attribution of reproduced failures with a read-only verdict sensitivity, and the steady
    reachability of the rows screened remove
  - Publish ``effects.json``, ``effects.md``, ``effects_families.png``, the range-evidence copies and ``manifest.json``
    once in a digest-named directory, and combine both admitted task results into ``recommendations.md``

Design principles:
  - Exploratory and read-only: no call reaches ``build_verdicts`` or ``judge_field_test``, and nothing is written into a
    verdict, tolerance table, member record or code projection; every output says so first or in its metadata
  - Every number is a measured value; the resolution floor and the scale resolution only set flags
  - Fail closed on any binding: a differing reconstruction, run identity, Train membership, case contract, range claim or
    digest refuses; the result digest binds every input, the stall check and the complete definition set
  - A committed directory (one holding ``manifest.json``) is only verified, never written; an uncommitted one is filled
    by exclusive creation with ``manifest.json`` last; no symbolic link is followed

This module does NOT:
  - Change, replace or reinterpret the frozen verdicts, or decide which rows are kept
  - Run COMSOL, models or Generation, or read the exact-stop state of a transient case
  - Register itself in a package initializer; it is imported by its module path

Notes:
  The statistic is a tested-range response at the stage-A bases (plan v4 of item 230), not a global sensitivity index.
  Effect sizes are ``d = |delta| / scale``, the scale 1.4826 x MAD over the material's ID Train cases (the pooled ID
  benchmark only for a base of a declared material-OOD material). Transient phase quantities are interpolated linearly
  between stored hours at ``tau = t / t_stop``, a censored case included; a phase outside the stored window is missing.
  ``DEFINITIONS`` holds every threshold, band, rule order and table; its digest is part of the result digest. Floats are
  persisted with 12 significant digits.

"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src import common, datasets, domain, experiments, generation

from . import field_tests_endpoints as endpoints
from . import field_tests_tolerances as tolerances
from . import field_tests_verdicts as verdicts

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

LABEL: Final = "exploratory; does not change the frozen verdicts"
SCHEMA_KIND: Final = "field_test_effects"
STEADY: Final = "steady_flow"
TRANSIENT: Final = "transient_drying"
POOLED: Final = "pooled_id"
PHASES: Final = (0.25, 0.5, 0.75)
MAD_SCALE: Final = 1.4826
NEGLIGIBLE: Final = 0.05
SUBSTANTIAL: Final = 0.2
SCALE_FLOOR_FACTOR: Final = 20.0
SPEED_FLOOR: Final = 1.0e-12
STALL_MARGIN: Final = 0.8
ASSUMED_CHECK_SECONDS: Final = 900.0
INPUT_INVARIANT_RTOL: Final = 1.0e-12
INVENTORY_RTOL: Final = 1.0e-9
FLOAT_DIGITS: Final = 12
MIN_SUPPORT: Final = 2
RESOLUTION_UNKNOWN: Final = "resolution unknown"
RUN_LEVEL: Final = "run-level, includes the duration effect"
BASE_OR_HARDWARE: Final = "base or hardware (challenge contribution not excluded)"
PARAMETER_RELATED: Final = "parameter-related"
MIXED: Final = "mixed/unknown"
CROSSING_CLOCK: Final = "crossing clock"
INTERVAL_LEVELS: Final = ("low", "high")
EVIDENCE_DOCUMENTS: Final = ("DECISION_TABLE.md", "PARAMETER_EVIDENCE.md")
TASK_FILES: Final = ("effects.json", "effects.md", "effects_families.png")
# The ID Train population of the yardstick runs per material (plan «Population for scales»); kidney bean has none.
TRAIN_POPULATION: Final = {TRANSIENT: {"chickpea": 32, "lentil": 32}, STEADY: {"chickpea": 512, "lentil": 512}}
ANISOTROPY_PAIR: Final = ("permeability.anisotropy.max_ratio", "permeability.anisotropy.strength")
ANISOTROPY_COORDINATE: Final = "permeability.anisotropy.strength*(max_ratio-1)"
FINE_TEXTURE: Final = tuple(
    f"{block}.structure.{name}" for block in ("bed", "initial_moisture") for name in ("fine_len_rel", "fine_ani_x", "fine_ani_y")
)
FAMILIES: Final = (
    ("Bettstruktur", ("bed.structure.", "bed.perturbations.")),
    ("Anfangsfeuchte", ("initial_moisture.structure.",)),
    ("Permeabilität", ("permeability.anisotropy.", "permeability.orientation.")),
    ("Porosität", ("porosity.",)),
    ("Einlass", ("pressure_bc.",)),
    ("Betriebsplan", ("schedule.", "omega_in_amp")),
    ("Materialgrössen", ("k_gr", "cp_gr_dry")),
    ("Umgebung", ("T_amb",)),
)
OWNER_DECISIONS: Final = {"permeability.orientation.jitter": "retained by Rino on 27.11. 06:41 without stage B"}
QUALIFICATIONS: Final = (
    "coupled-stream rows: generator effects including reassigned downstream draws",
    "inlet rows (pressure_bc.*): provisional until template 63 (item 229)",
    "all statements: ID supports of the tested materials and today's template only",
    "tested-range response at the stage-A bases; not a global sensitivity index and blind to interactions",
    "family shares are a descriptive index of the tested challenges and depend on family size and coverage",
)
# How each steady input enters the steady equations (Brinkman momentum residual and inlet boundary).
STEADY_INPUT_PATHS: Final = {
    "Kxx": "Darcy term mu K^-1 u, src/domain/physics/domain_physics_brinkman.py:25",
    "Kxy": "Darcy term mu K^-1 u, src/domain/physics/domain_physics_brinkman.py:25",
    "Kyy": "Darcy term mu K^-1 u, src/domain/physics/domain_physics_brinkman.py:25",
    "eps_bed": "Brinkman stress tau = (mu/eps)(...), src/domain/physics/domain_physics_brinkman.py:26, docs/02_physical_model.md:24-26",
    "p_in_bc": "inlet pressure boundary condition on y = 0, docs/02_physical_model.md:24",
}
_DIAGNOSTICS: Final = ("f_wet_dm", "X_wb_bulk", "T_out_mean", "phi_out_mean")
_PHASE_QOIS: Final = (
    ("T2", "f_wet_dm"),
    ("T2", "X_wb_bulk"),
    ("T3", "moisture_spread"),
    ("T4", "T_out_mean"),
    ("T4", "phi_out_mean"),
    ("T5", "T_bed_max"),
)
DURATION: Final = "T1/duration"
QOIS: Final = {
    TRANSIENT: (DURATION, *(f"{code}/{name}@tau={tau:g}" for tau in PHASES for code, name in _PHASE_QOIS)),
    STEADY: ("S1/pressure_drop", "S2/outlet_v", "S3/flow_nonuniformity", "S4/wall_core_speed"),
}
LOCALITY_FIELDS: Final = {
    TRANSIENT: tuple(f"{name}@tau={tau:g}" for tau in PHASES for name in endpoints.TRANSIENT_STATES),
    STEADY: endpoints.STEADY_OUTPUTS,
}
# Every rule order and table the result depends on; the labels below are taken from these tables.
DISPOSITION_RULES: Final = (
    ("not kept in this task", "V is remove or unresolved; the bands are reported"),
    ("indeterminate", "V is keep, and a QoI headline median is undefined, trigger evidence is missing, or P, N and F are all empty"),
    ("keep rests on failures only", "V is keep, F is true, P and N are empty"),
    ("acts globally", f"V is keep, Gmax >= {NEGLIGIBLE}"),
    ("acts only locally", f"V is keep, N is empty, P is non-empty, Gmax < {NEGLIGIBLE} with resolved scales; F is noted if true"),
    ("small non-local effect", f"V is keep, N is non-empty (named), Gmax < {NEGLIGIBLE} with resolved scales; F is noted if true"),
    ("insufficient evidence", "everything else, with the reason"),
)
NOT_KEPT, INDETERMINATE, FAILURES_ONLY, ACTS_GLOBALLY, ACTS_LOCALLY, NON_LOCAL, INSUFFICIENT = (label for label, _rule in DISPOSITION_RULES)
REACHABILITY_ORDER: Final = (
    ("input invariant", f"every comparison holds all five input deltas, each max|delta| <= {INPUT_INVARIANT_RTOL:g} x max|reference input|"),
    (RESOLUTION_UNKNOWN, "a gate replicate difference is missing"),
    ("unknown", "a comparison other than an infeasible contrast, an endpoint value or its ratio is missing"),
    ("within resolution", "every endpoint difference <= the gate replicate difference of that endpoint"),
    ("resolved, max ratio < 0.5", "the largest endpoint ratio is below the removal margin"),
    ("resolved, 0.5 <= max ratio < 1", "the largest endpoint ratio is below the keep ratio"),
    ("resolved, max ratio >= 1", "otherwise"),
)
INVARIANT, _, UNKNOWN, WITHIN, RESOLVED_LOW, RESOLVED_MID, RESOLVED_HIGH = (label for label, _rule in REACHABILITY_ORDER)
ATTRIBUTION_RULES: Final = (
    (BASE_OR_HARDWARE, "every failing attempt has a matched reference attempt that failed or crossed at or after the margin"),
    (PARAMETER_RELATED, "every failing attempt has a matched reference attempt that succeeded and crossed before the margin"),
    (MIXED, "otherwise, including a failing attempt without a matched reference attempt"),
)
RANGE_RULES: Final = {
    "best_reference": f"per QoI the reference with the largest level median d over at least {MIN_SUPPORT} bases; ties count each",
    "bereich_enger": "a reproduced failure attributed parameter-related at an interval level (low, high)",
    "bereich_groesser_pruefen": f"interval level and ID material: the best median d >= {SUBSTANTIAL}, the level among the tied headline "
    "winners at at least half of the bases supporting that median, and a range claim beyond the ID bound on the level's side",
    "frage_fuer_neuen_test": f"any level and material: the best median d >= {SUBSTANTIAL} (support minimum inherited) without a range claim",
}
_TORCH_THREADS: Final = 1
DEFINITIONS: Final = {
    "plan": "PLAN_EFFECTS v4 (items 223 and 230)",
    "qois": QOIS,
    "phases": {"points": PHASES, "rule": "tau = t / t_stop of each case, censored too; linear between the bracketing stored hours only"},
    "train_population": TRAIN_POPULATION,
    "scale": {
        "spread": f"{MAD_SCALE} x MAD over the material's ID Train cases; censored Train cases are counted and excluded",
        "pooled_id": "both ID materials pooled, only for a base of a declared material-OOD material",
        "not_resolved": f"zero, not finite, or below {SCALE_FLOOR_FACTOR:g} x a complete floor; an incomplete floor leaves that test unknown",
    },
    "floor": "largest |delta| between a gate and its published base; complete only when every planned gate pair has a value",
    "within_resolution": f"|delta| <= a complete floor, else {RESOLUTION_UNKNOWN}",
    "bands": {"negligible": NEGLIGIBLE, "substantial": SUBSTANTIAL, "support": MIN_SUPPORT, "negligible_needs": "a resolved scale"},
    "speed_floor": SPEED_FLOOR,
    "disposition_order": DISPOSITION_RULES,
    "range_flags": RANGE_RULES,
    "families": FAMILIES,
    "shares": "median headline d over the ID bases every member supports, over the family sum; the coordinate takes the larger d per base",
    "stall": {
        "margin": STALL_MARGIN,
        "assumed_check_seconds": f"{ASSUMED_CHECK_SECONDS:g}, only when a frozen Source file is missing; no frozen stall rule means no check time",
        "match": "same known CPU model and the reference's accepted draws",
        "crossing": "counts only when recorded for the frozen rule's minimum_simulated_seconds; otherwise unknown and never a match",
        "attribution": ATTRIBUTION_RULES,
        "near_threshold": "a stalled reference attempt, or the median crossing of its successful ones within the margin; unknown without evidence",
        "sensitivity": "the row verdict without non-parameter failures; a result other than keep reads unresolved",
    },
    "reachability_order": REACHABILITY_ORDER,
    "inventory": {"floats": f"equal, or within relative {INVENTORY_RTOL:g}", "thresholds": (verdicts.KEEP_RATIO, verdicts.REMOVE_MARGIN)},
    "owner_decisions": OWNER_DECISIONS,
    "persistence": {"float_digits": FLOAT_DIGITS, "torch_threads": _TORCH_THREADS},
}
# Every module this one calls can change an admission or a result, so each is a member; the endpoint projection adds the
# endpoint and Source-output readers.
EFFECTS_PROJECTION: Final = common.code_identity.CodeProjection(
    name="field_test_effects",
    files=(
        "analysis/field_tests/field_tests_effects.py",
        "analysis/field_tests/field_tests_tolerances.py",
        "analysis/field_tests/field_tests_verdicts.py",
        "common/common_code_identity.py",
        "common/common_paths.py",
        "common/common_serialization.py",
        "datasets/contracts/dataset_contracts_identity.py",
        "datasets/contracts/dataset_contracts_partition.py",
        "datasets/packages/dataset_packages_manifest.py",
        "datasets/preprocessing/dataset_preprocessing_splits.py",
        "domain/domain_moisture.py",
        "experiments/experiments_run.py",
        "generation/field_tests/generation_field_tests_planning.py",
        "generation/field_tests/generation_field_tests_record.py",
        "generation/field_tests/generation_field_tests_roster.py",
        "generation/publication/generation_publication_campaign_evidence.py",
        "generation/publication/generation_publication_catalog.py",
        "generation/runtime/generation_runtime_timing.py",
    ),
    includes=(endpoints.ENDPOINT_PROJECTION,),
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MARKERS: Final = "osD^v<>ph*"
# The verdict module's own assembly reconstructs the persisted inventory; it is called read-only (plan «Binding inputs»).
_ADMITTED_TOLERANCES: Final = verdicts._admitted_tolerances  # noqa: SLF001 -- read-only reuse of the frozen assembly
_REPLICATES: Final = verdicts._replicates  # noqa: SLF001 -- read-only reuse of the frozen assembly
_COMPARISONS: Final = verdicts._comparisons  # noqa: SLF001 -- read-only reuse of the frozen assembly
_ROW_VERDICT: Final = verdicts._row_verdict  # noqa: SLF001 -- read-only reuse of the frozen assembly
_VERIFIED_CASE: Final = verdicts._verified_case  # noqa: SLF001 -- the frozen digest-checked case reader
_WALL_CORE_CONTRAST: Final = endpoints._wall_core_contrast  # noqa: SLF001 -- S4 is the existing contrast definition
_SOURCE_SELECTOR_KEYS: Final = tolerances._SOURCE_SELECTOR_KEYS  # noqa: SLF001 -- the yardstick's Source selector
_RUN_IDENTITY: Final = tolerances._run_identity  # noqa: SLF001 -- the identity the tolerance table records per run
_STALL_PROGRESS: Final = generation.runtime.timing._stall_progress  # noqa: SLF001 -- the one stall-progress validator


@dataclass(frozen=True, slots=True)
class RangeClaim:
    """
    One applicable value beyond the ID support of one row and material, as ``PARAMETER_EVIDENCE.md`` states it.

    Attributes
    ----------
    value : float
        The applicable value, in the row's registry unit.
    unit : str
        Its unit.
    citation : str
        Where the evidence documents state it (its provenance); never empty.
    id_bound : float
        The ID bound it lies beyond: the ID maximum for ``high``, the ID minimum for ``low``.

    """

    value: float
    unit: str
    citation: str
    id_bound: float


@dataclass(frozen=True, slots=True)
class RangeEvidence:
    """
    The range evidence the «Bereich grösser prüfen» flag reads: both bound documents and the claims read from them.

    Attributes
    ----------
    files : Mapping[str, Path | str]
        Exactly ``PARAMETER_EVIDENCE.md`` and ``DECISION_TABLE.md`` (the decision-table revision) by published name; each
        is hashed and copied into the result directory.
    claims : Mapping[tuple[str, str, str], RangeClaim]
        ``(row, ID material, interval level)`` to its claim of an applicable value beyond the ID bound on that level's side.

    """

    files: Mapping[str, Path | str]
    claims: Mapping[tuple[str, str, str], RangeClaim] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaseQoIs:
    """One case's quantities of interest, missing reasons, completion, run-level values and optional locality fields."""

    values: Mapping[str, float]
    missing: Mapping[str, str]
    reached: bool | None = None
    target: tuple[float, float, float] | None = None
    supplementary: Mapping[str, float] = field(default_factory=dict)
    fields: Mapping[str, np.ndarray] = field(default_factory=dict)
    inputs: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One recorded attempt: typed outcome and reason, hardware, accepted draws, both stall-progress fields and their evidence."""

    member_id: str
    number: int
    classification: str | None
    reason: str | None
    cpu_model: str | None
    host: str | None
    draws: tuple[Any, Any]
    minimum_simulated: float | None
    crossing: float | None
    evidence: str

    @property
    def clock(self) -> float | None:
        """Return the crossing clock only when it is evidence for the frozen stall rule's minimum."""
        return self.crossing if self.evidence == CROSSING_CLOCK else None


@dataclass(frozen=True, slots=True)
class _Bound:
    """The admitted frozen stage-A results of one field test with the reconstructed inventory."""

    task: str
    directory: Path
    roster: Mapping[str, Any]
    version: int
    frozen: tolerances.FrozenTolerances
    verdicts_sha256: str
    persisted: Mapping[str, Any]
    members: Mapping[str, Mapping[str, Any]]
    seals: Mapping[str, Mapping[str, Any] | None]
    comparisons: Sequence[verdicts._Comparison]
    source_root: Path


# ----------------------------------------------------------------------------- pure definitions


def bracket(times: np.ndarray, t: float) -> tuple[int, int, float] | str:
    """Return the bracketing stored-hour indices and weight of time ``t``, or why it has none; only an exact hour is read directly."""
    if t < float(times[0]):
        return "before_first_stored_hour"
    if t > float(times[-1]):
        return "beyond_last_stored_hour"
    upper = int(np.searchsorted(times, t, side="left"))
    if float(times[upper]) == t:
        return upper, upper, 0.0
    lower = upper - 1
    return lower, upper, (t - float(times[lower])) / (float(times[upper]) - float(times[lower]))


def _case_qois(values: Mapping[str, float], missing: Mapping[str, str], **context: Any) -> CaseQoIs:
    """Return one case's quantities with every non-finite derived value moved to the missing ones."""
    finite = {name: value for name, value in values.items() if math.isfinite(value)}
    return CaseQoIs(finite, {**missing, **dict.fromkeys(sorted(values.keys() - finite.keys()), "non-finite value")}, **context)


def transient_qois(
    times: np.ndarray,
    states: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    f_surf: float,
    rho_bu_dry: np.ndarray,
    *,
    t_stop: float,
    reached: bool,
    target: tuple[float, float, float],
    keep_fields: bool = False,
) -> CaseQoIs:
    """
    Return the transient quantities of one case from its stored hours only.

    Parameters
    ----------
    times, states : numpy.ndarray
        Stored hours ``[N]`` (strictly increasing) and stored states ``[N, 4, Y, X]`` (T, phi, w_surf, w_int).
    diagnostics : Mapping[str, numpy.ndarray]
        ``f_wet_dm``, ``X_wb_bulk``, ``T_out_mean`` and ``phi_out_mean`` at the stored hours ``[N]``.
    f_surf, rho_bu_dry : float, numpy.ndarray
        Surface water fraction and local dry bulk density ``[Y, X]``.
    t_stop, reached, target : float, bool, tuple
        Exact stop time, target flag, and (target wet basis, fraction limit, horizon).
    keep_fields : bool
        Keep the interpolated state fields for the locality index.

    Returns
    -------
    CaseQoIs
        T1 duration, T2-T5 at each phase point of the case's own stop (also when censored), the run-level supplementary
        values and the requested fields; a phase point outside the stored window is missing with its reason.

    """
    values: dict[str, float] = {DURATION: float(t_stop)}
    missing: dict[str, str] = {}
    fields: dict[str, np.ndarray] = {}
    for tau in PHASES:
        names = [f"{code}/{name}@tau={tau:g}" for code, name in _PHASE_QOIS]
        found = bracket(times, tau * t_stop)
        if isinstance(found, str):
            missing.update(dict.fromkeys(names, found))
            continue
        lower, upper, weight = found
        series = {name: (1.0 - weight) * float(diagnostics[name][lower]) + weight * float(diagnostics[name][upper]) for name in _DIAGNOSTICS}
        state = (1.0 - weight) * states[lower].astype(np.float64) + weight * states[upper].astype(np.float64)
        moisture = domain.moisture.dry_basis_moisture(domain.moisture.granular_water_content(state[2], state[3], f_surf), np.asarray(rho_bu_dry))
        q05, q95 = np.percentile(moisture, [5.0, 95.0])
        measured = (series["f_wet_dm"], series["X_wb_bulk"], float(q95 - q05), series["T_out_mean"], series["phi_out_mean"], float(state[0].max()))
        values.update(zip(names, measured, strict=True))
        if keep_fields:
            fields.update({f"{name}@tau={tau:g}": state[channel] for channel, name in enumerate(endpoints.TRANSIENT_STATES)})
    supplementary = {"bed_T_max_stored": float(states[:, 0].max())}
    if times.size > 1:
        span = float(times[-1] - times[0])
        supplementary.update({f"{name}_time_mean": float(np.trapezoid(diagnostics[name], times)) / span for name in ("T_out_mean", "phi_out_mean")})
    return _case_qois(values, missing, reached=reached, target=target, supplementary=supplementary, fields=fields)


def steady_qois(fields: Mapping[str, np.ndarray], x: np.ndarray, *, keep_fields: bool = False) -> CaseQoIs:
    """
    Return the steady quantities S1-S4 of one case; row means are trapezoid means over the boundary-inclusive ``x``.

    ``fields`` holds p, u, v and speed (and the inputs for ``keep_fields``); the first grid row is the inlet (y-min).
    """
    width = float(x[-1] - x[0])

    def row_mean(row: np.ndarray) -> float:
        return float(np.trapezoid(row, x)) / width

    speed = fields["speed"]
    values = {
        "S1/pressure_drop": row_mean(fields["p"][0]) - row_mean(fields["p"][-1]),
        "S2/outlet_v": row_mean(fields["v"][-1]),
        "S4/wall_core_speed": _WALL_CORE_CONTRAST(speed),
    }
    missing: dict[str, str] = {}
    mean_speed = float(speed.mean())
    if mean_speed < SPEED_FLOOR:
        missing["S3/flow_nonuniformity"] = "mean_speed_below_1e-12"
    else:
        values["S3/flow_nonuniformity"] = float(speed.std()) / mean_speed
    kept = {name: fields[name] for name in endpoints.STEADY_OUTPUTS} if keep_fields else {}
    inputs = {name: fields[name] for name in endpoints.STEADY_INPUTS if name in fields} if keep_fields else {}
    return _case_qois(values, missing, fields=kept, inputs=inputs)


def paired_delta(qoi: str, reference: CaseQoIs, variant: CaseQoIs) -> tuple[float | None, tuple[str, float] | None, str | None]:
    """
    Return the signed paired difference variant - reference of one QoI, a censoring bound, and a missing reason.

    The duration of a transient pair needs equal target definitions and horizons. Variant-only censoring gives a lower
    bound, reference-only censoring an upper bound, and both censored give nothing. A missing value names each side.
    """
    if qoi == DURATION:
        if reference.target != variant.target:
            return None, None, "target_or_horizon_differs"
        raw = variant.values[qoi] - reference.values[qoi]
        if reference.reached and variant.reached:
            return raw, None, None
        if reference.reached:
            return None, ("lower", raw), "censored_variant"
        if variant.reached:
            return None, ("upper", raw), "censored_reference"
        return None, None, "both_censored"
    absent = [f"{side}: {case.missing[qoi]}" for side, case in (("reference", reference), ("variant", variant)) if qoi in case.missing]
    if absent:
        return None, None, "; ".join(absent)
    delta = variant.values[qoi] - reference.values[qoi]
    return (delta, None, None) if math.isfinite(delta) else (None, None, "non-finite difference")


def robust_scale(values: Sequence[float]) -> tuple[float | None, int]:
    """Return 1.4826 x the median absolute deviation and the support, ``None`` without a value."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return None, 0
    return float(MAD_SCALE * np.median(np.abs(finite - np.median(finite)))), int(finite.size)


def resolution_floor(deltas: Sequence[float | None], planned: int) -> dict[str, Any]:
    """Return the largest |delta| over the gate replicate pairs; complete only when every planned pair has a value."""
    finite = [abs(value) for value in deltas if value is not None and math.isfinite(value)]
    return {"value": max(finite) if finite else None, "complete": planned > 0 and len(finite) == planned, "pairs": len(finite), "planned": planned}


def scale_status(scale: float | None, floor: Mapping[str, Any]) -> tuple[bool, str | None]:
    """
    Return whether a scale is resolved and the note on it.

    A scale is not resolved only when it is zero, not finite, or below 20 x a complete floor; an incomplete floor leaves
    the 20 x test at «resolution unknown», noted, while the scale stays resolved.
    """
    if scale is None or not math.isfinite(scale) or scale == 0.0:
        return False, "scale zero or not finite"
    if not floor["complete"]:
        return True, f"{SCALE_FLOOR_FACTOR:g} x floor test: {RESOLUTION_UNKNOWN} (floor incomplete)"
    if scale < SCALE_FLOOR_FACTOR * float(floor["value"]):
        return False, f"scale below {SCALE_FLOOR_FACTOR:g} x floor"
    return True, None


def band(d: float | None, *, resolved: bool) -> str | None:
    """Return the descriptive band of an effect size; «negligible» needs a resolved scale."""
    if d is None:
        return None
    if d < NEGLIGIBLE:
        return "negligible" if resolved else "below 0.05, scale not resolved"
    return "small" if d < SUBSTANTIAL else "substantial"


def effect_entry(
    delta: float | None, scale: Mapping[str, Any], floor: Mapping[str, Any], *, censoring: tuple[str, float] | None = None, reason: str | None = None
) -> dict[str, Any]:
    """
    Return one comparison's measured effect on one QoI: signed delta, d, the resolution flags and a censoring bound.

    A bound establishes a minimum |delta| only on its informative side; it is reported apart from exact values.
    """
    value = float("nan") if scale["value"] is None else float(scale["value"])
    usable = math.isfinite(value) and value > 0.0
    entry: dict[str, Any] = {"delta": delta, "d": None, "reason": reason, "within_resolution": None, "scale_resolved": scale["resolved"]}
    if censoring is not None:
        kind, raw = censoring
        established = max(raw, 0.0) if kind == "lower" else max(-raw, 0.0)
        at_least = established / value if usable and established > 0.0 else None
        entry["bound"] = {"kind": kind, "delta": raw, "d_at_least": at_least, "band_at_least": band(at_least, resolved=scale["resolved"])}
    if delta is None:
        return entry
    entry["within_resolution"] = abs(delta) <= float(floor["value"]) if floor["complete"] else RESOLUTION_UNKNOWN
    if usable:
        entry["d"] = abs(delta) / value
    else:
        entry["reason"] = "scale zero or not finite"
    return entry


def scale_group(material: str, material_roles: Mapping[str, str]) -> str:
    """Return the scale a base's effects are measured against: its own ID material, or the pooled ID benchmark for material OOD."""
    role = material_roles.get(material)
    if role is None:
        msg = f"Base material {material!r} has neither an ID Train population nor a declared material-OOD role."
        raise ValueError(msg)
    return material if role == "id" else POOLED


def tiles(shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    """Return every field-test patch of a grid once, over all regions."""
    seen: dict[tuple[int, int], tuple[slice, slice]] = {}
    for region in ("wall", "inlet", "outlet", "interior"):
        for rows, columns in endpoints.patches(shape, region):
            seen.setdefault((int(rows.start), int(columns.start)), (rows, columns))
    return [seen[key] for key in sorted(seen)]


def locality_index(reference: np.ndarray, other: np.ndarray) -> float | None:
    """Return the largest patch-mean absolute difference over the whole-field RMS difference, ``None`` at zero RMS."""
    difference = np.asarray(other, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rms = math.sqrt(float(np.mean(difference**2)))
    if rms == 0.0:
        return None
    shape = (int(difference.shape[0]), int(difference.shape[1]))
    return max(float(np.mean(np.abs(difference[rows, columns]))) for rows, columns in tiles(shape)) / rms


def patch_coverage(shape: tuple[int, int]) -> float:
    """Return the fraction of grid cells the field-test patches cover."""
    covered = np.zeros(shape, dtype=bool)
    for rows, columns in tiles(shape):
        covered[rows, columns] = True
    return float(covered.mean())


def family_of(row: str) -> str:
    """Return the family of a parameter row; an exact name or a dotted prefix matches."""
    for name, prefixes in FAMILIES:
        if any(row == prefix or (prefix.endswith(".") and row.startswith(prefix)) for prefix in prefixes):
            return name
    return "unassigned"


def trigger_evidence(comparisons: Sequence[Mapping[str, Any]], limits: Mapping[str, float], *, failure: bool) -> dict[str, Any]:
    """
    Return the keep triggers of one row: every persisted endpoint above the keep ratio, by comparison, patch or non-patch.

    Only valid comparisons count, and the duration is non-patch. Evidence is usable when a valid comparison has at least
    one endpoint ratio; a valid pair without one is reported apart. ``failure`` records a reproduced solver failure the
    frozen rule counts as a keep trigger; evidence is missing when there is neither usable evidence nor such a failure.
    """
    patch: list[list[str]] = []
    other: list[list[str]] = []
    usable = False
    valid = [item for item in comparisons if item["problem"] is None]
    for item in valid:
        for name, (value, _contributors) in sorted(item["measured"].items()):
            tolerance = limits.get(name)
            if isinstance(value, float) and tolerance is not None and tolerance > 0.0:
                usable = True
                if value / tolerance > verdicts.KEEP_RATIO:
                    (patch if "/patch_" in name else other).append([str(item["member_id"]), name])
    evidence = "usable" if usable else ("valid pair without measurements" if valid else "no valid comparison")
    return {"patch": sorted(patch), "non_patch": sorted(other), "failure": failure, "evidence": evidence, "missing": not usable and not failure}


def _quantities(triggers: Iterable[Sequence[str]]) -> str:
    """Return the distinct endpoint quantities (without hour) of trigger pairs."""
    return ", ".join(sorted({endpoints.split_hour(name)[0] for _member, name in triggers}))


def disposition(verdict: str, medians: Mapping[str, tuple[float | None, bool]], triggers: Mapping[str, Any]) -> tuple[str, str, float | None]:
    """
    Return the exploratory disposition of one row in one task, its reason and Gmax; the first matching rule applies.

    ``medians`` maps each QoI to its pooled ID headline median ``d`` and whether its scales are resolved; Gmax is the largest
    defined median.
    """
    gmax = max((median for median, _resolved in medians.values() if median is not None), default=None)
    patch, other, failure = triggers["patch"], triggers["non_patch"], triggers["failure"]
    if verdict in {"remove", "unresolved"}:
        return NOT_KEPT, f"frozen verdict {verdict}", gmax
    undefined = sorted(qoi for qoi, (median, _resolved) in medians.items() if median is None)
    if undefined:
        return INDETERMINATE, "undefined headline median: " + ", ".join(undefined), gmax
    if triggers["missing"]:
        return INDETERMINATE, "trigger evidence missing", gmax
    if not (patch or other or failure):
        return INDETERMINATE, f"trigger evidence: {triggers['evidence']}, no keep trigger", gmax
    if failure and not patch and not other:
        return FAILURES_ONLY, "reproduced solver failure is the only keep trigger", gmax
    if gmax is not None and gmax >= NEGLIGIBLE:
        return ACTS_GLOBALLY, f"Gmax {gmax:.3g}", gmax
    resolved = all(flag for _median, flag in medians.values())
    note = "; failure trigger too" if failure else ""
    if resolved and not other and patch:
        return ACTS_LOCALLY, f"patch triggers {_quantities(patch)}{note}", gmax
    if resolved and other:
        return NON_LOCAL, f"non-patch triggers {_quantities(other)}{note}", gmax
    return INSUFFICIENT, "Gmax below 0.05 but a scale not resolved" if not resolved else "no rule applies", gmax


def coordinate_cells(cells: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Mapping[str, Mapping[str, Any]]]:
    """Return a family's headline cells with the anisotropy pair replaced by one coordinate: per base the cell of the larger ``d``."""
    if not set(ANISOTROPY_PAIR) <= set(cells):
        return dict(cells)
    pair = [cells[name] for name in ANISOTROPY_PAIR]
    merged: dict[str, Mapping[str, Any]] = {}
    for base in sorted(set(pair[0]) | set(pair[1])):
        found = [by_base[base] for by_base in pair if base in by_base and by_base[base]["d"] is not None]
        merged[base] = max(found, key=lambda cell: cell["d"]) if found else {"d": None, "within_resolution": None}
    return {**{name: value for name, value in cells.items() if name not in ANISOTROPY_PAIR}, ANISOTROPY_COORDINATE: merged}


def family_shares(cells: Mapping[str, Mapping[str, Mapping[str, Any]]], bases: Sequence[str], challenges: Mapping[str, int]) -> dict[str, Any]:
    """
    Return the share of each member's median headline ``d`` over the common support in the family sum.

    ``cells[member][base]`` holds a member's headline ``d`` and its within-resolution flag; the common support is the set of
    ``bases`` at which every member has a finite ``d``. Without a common base or with a zero sum every share is undefined.
    """
    finite = {member: {base: cell["d"] for base, cell in by_base.items() if cell["d"] is not None} for member, by_base in cells.items()}
    common_support = [base for base in bases if all(base in values for values in finite.values())]
    medians = {member: float(np.median([values[base] for base in common_support])) if common_support else None for member, values in finite.items()}
    total = sum(median for median in medians.values() if median is not None)
    return {
        "common_support": len(common_support),
        "coverage": len(common_support) / len(bases) if bases else None,
        "members": {
            member: {
                "median_d": median,
                "share": median / total if median is not None and total > 0.0 else None,
                "within_resolution": sum(cells[member][base]["within_resolution"] is True for base in common_support),
                "challenges": challenges[member],
            }
            for member, median in medians.items()
        },
    }


def attribute_failure(failing: Sequence[Attempt], references: Sequence[Attempt], check_seconds: float | None) -> str:
    """
    Attribute one reproduced failure from its failing attempts and the reference attempts on the same draws and CPU model.

    A match needs the same known CPU model and the same accepted draws. Base or hardware: every failing attempt has a
    matched reference attempt that failed or crossed within the margin. Parameter-related: every failing attempt has a
    matched reference attempt that succeeded and crossed outside it. Otherwise, including no match: mixed or unknown.
    Only a crossing clock recorded for the frozen rule's minimum counts (``Attempt.clock``).
    """
    margin = None if check_seconds is None else STALL_MARGIN * check_seconds
    hardware, parameter = [], []
    for attempt in failing:
        key = (attempt.cpu_model, attempt.draws)
        matched = [item for item in references if attempt.cpu_model is not None and (item.cpu_model, item.draws) == key]
        # late: crossed at or after the margin; None when the check time or the crossing clock is unknown.
        crossed = [(item.classification, None if margin is None or (clock := item.clock) is None else clock >= margin) for item in matched]
        hardware.append(any(classification == "solver_failure" or late is True for classification, late in crossed))
        parameter.append(any(classification == "succeeded" and late is False for classification, late in crossed))
    if failing and all(hardware):
        return BASE_OR_HARDWARE
    if failing and all(parameter):
        return PARAMETER_RELATED
    return MIXED


def near_threshold(references: Sequence[Attempt], check_seconds: float | None) -> bool | None:
    """
    Return whether a base is near the stall threshold, ``None`` (unknown) when the evidence to decide is missing.

    Near: a reference attempt stalled, or the median crossing clock of its successful reference attempts is within the
    margin. Unknown: no check time, an attempt without its outcome, or a successful attempt without a crossing clock
    recorded for the frozen rule's minimum.
    """
    if any(item.reason == "stalled" for item in references):
        return True
    succeeded = [item for item in references if item.classification == "succeeded"]
    crossings = [clock for item in succeeded if (clock := item.clock) is not None]
    if check_seconds is None or not succeeded or len(crossings) != len(succeeded) or any(item.classification is None for item in references):
        return None
    return float(np.median(crossings)) >= STALL_MARGIN * check_seconds


def reachability_class(comparisons: Sequence[Mapping[str, Any]]) -> str:
    """
    Classify the steady reachability of a row screened remove; the first matching rule of ``REACHABILITY_ORDER`` applies.

    Each comparison holds its ``problem``, ``inputs`` (input name to ``[max |delta|, max |reference|]``) and ``endpoints``
    (name to ``[difference, ratio, replicate difference]``, ``None`` where missing). An infeasible contrast stays listed
    but holds nothing to measure; any other invalid comparison is missing evidence.
    """
    valid = [item for item in comparisons if item["problem"] is None]
    complete = bool(valid) and all(item["problem"] in {None, "infeasible"} for item in comparisons)
    inputs = [item["inputs"] for item in valid]
    if (
        complete
        and all(set(item) == set(endpoints.STEADY_INPUTS) for item in inputs)
        and all(delta <= INPUT_INVARIANT_RTOL * scale for item in inputs for delta, scale in item.values())
    ):
        return INVARIANT
    rows = [entry for item in valid for entry in item["endpoints"].values()]
    if any(replicate is None for _difference, _ratio, replicate in rows):
        return RESOLUTION_UNKNOWN
    if not complete or not rows or any(difference is None or ratio is None for difference, ratio, _replicate in rows):
        return UNKNOWN
    if all(difference <= replicate for difference, _ratio, replicate in rows):
        return WITHIN
    largest = max(ratio for _difference, ratio, _replicate in rows)
    if largest < verdicts.REMOVE_MARGIN:
        return RESOLVED_LOW
    return RESOLVED_MID if largest < verdicts.KEEP_RATIO else RESOLVED_HIGH


def _median(values: Iterable[float]) -> float | None:
    """Return the median of the values, ``None`` without one."""
    found = list(values)
    return float(np.median(found)) if found else None


# ----------------------------------------------------------------------------- binding and reading


def _same(left: Any, right: Any) -> bool:
    """Return whether two JSON values agree: floats equal or within relative 1e-9 (zero absolute), everything else exactly."""
    if isinstance(left, float) and isinstance(right, float):
        return left == right or math.isclose(left, right, rel_tol=INVENTORY_RTOL, abs_tol=0.0)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _sides(record: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, tuple[int, ...] | None]:
    """Return on which side of the keep ratio and the removal margin each ratio of one inventory record lies."""
    ratios: dict[str, Any] = {"worst_ratio": record.get("worst_ratio")}
    for name, (value, _contributors) in record.get("measured", {}).items():
        tolerance = limits.get(name)
        ratios[name] = value / tolerance if isinstance(value, float) and tolerance is not None and tolerance > 0.0 else None
    ratios.update({f"replicate {name}": ratio for name, (_value, ratio) in record.get("endpoints", {}).items()})
    thresholds = (verdicts.KEEP_RATIO, verdicts.REMOVE_MARGIN)
    return {key: tuple(int(r > t) - int(r < t) for t in thresholds) if isinstance(r, float) else None for key, r in ratios.items()}


def _require_same(
    label: str, reconstructed: Sequence[Mapping[str, Any]], persisted: Sequence[Mapping[str, Any]], limits: Mapping[str, float]
) -> None:
    """Refuse unless every reconstructed record equals its persisted record and lies on the same side of every threshold."""
    rebuilt = json.loads(json.dumps(list(reconstructed)))
    if len(rebuilt) != len(persisted):
        msg = f"Reconstructed {label} count {len(rebuilt)} differs from the persisted {len(persisted)}; the frozen inventory is not reproduced."
        raise ValueError(msg)
    for index, (left, right) in enumerate(zip(rebuilt, persisted, strict=True)):
        if not _same(left, right) or _sides(left, limits) != _sides(right, limits):
            msg = f"Reconstructed {label} record {index} differs from the persisted verdict inventory; the analysis refuses."
            raise ValueError(msg)


def _bind_inventory(task: str, test_id: str, *, storage_root: Path | str | None) -> _Bound:
    """
    Admit the frozen stage-A results and reproduce their inventory with the verdict module's own assembly.

    Raises
    ------
    ValueError
        If the field test is not stage A, the verdict file binds another test, stage, task, roster or table, or any
        reconstructed replicate, comparison or row verdict differs from the persisted one.

    """
    record = generation.field_tests.record
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    roster, version = generation.field_tests.roster.load_roster_version(directory)
    if roster["stage"] != "A":
        msg = f"Field test {test_id!r} is stage {roster['stage']}; the effect analysis reads stage-A screening results only."
        raise ValueError(msg)
    frozen = _ADMITTED_TOLERANCES(directory, roster, storage_root=storage_root)
    persisted = verdicts.load_verdicts(task, test_id, storage_root=storage_root)
    verdict_path = record.record_path(directory, "analysis", verdicts.VERDICTS_FILENAME)
    header = (persisted["test_id"], persisted.get("stage"), persisted.get("task"), roster["test_id"], roster["task"], frozen.payload["task"])
    binding = (persisted["roster_sha256"], persisted["roster_version"], persisted["tolerances_sha256"])
    if header != (test_id, "A", task, test_id, task, task) or binding != (roster["roster_sha256"], version, frozen.sha256):
        msg = f"{verdict_path} does not bind field test {test_id!r} (stage A, {task}), roster version {version}, its digest and the admitted table."
        raise ValueError(msg)
    members = record.effective_members(directory, roster)
    seals = {member["member_id"]: record.read_seal(directory, member["member_id"]) for member in members}
    source_root = common.paths.resolve_generation_campaign_directory(task, roster["base_source"]["campaign_name"], storage_root=storage_root)
    replicates, envelopes, base_ok = _REPLICATES(directory, members, seals, frozen.tolerances, source_root=source_root)
    comparisons = _COMPARISONS(directory, members, seals, frozen.tolerances, base_ok=base_ok, envelopes=envelopes)
    _require_same("replicates", replicates, persisted["replicates"], frozen.tolerances)
    _require_same("comparisons", [item.record() for item in comparisons], persisted["comparisons"], frozen.tolerances)
    by_row: dict[str, list[verdicts._Comparison]] = defaultdict(list)
    for item in comparisons:
        if item.role != "joint":
            by_row[str(item.row)].append(item)
    joints = [item for item in comparisons if item.role == "joint"]
    materials = {str(member["base"]["material_family"]) for member in members}
    rows = [
        _ROW_VERDICT(row, "A", items, merge_row=False, materials=materials, joint_cleared=bool(joints) and all(item.clears() for item in joints))
        for row, items in by_row.items()
    ]
    _require_same("row verdicts", rows, persisted["rows"], frozen.tolerances)
    return _Bound(
        task=task,
        directory=directory,
        roster=roster,
        version=version,
        frozen=frozen,
        verdicts_sha256=common.serialization.file_sha256(verdict_path),
        persisted=persisted,
        members={member["member_id"]: member for member in members},
        seals=seals,
        comparisons=comparisons,
        source_root=source_root,
    )


def _increasing(values: np.ndarray) -> bool:
    """Return whether a 1-D array is finite and strictly increasing."""
    return bool(values.ndim == 1 and np.isfinite(values).all() and (np.diff(values) > 0.0).all())


def _require_contract(checks: Mapping[str, object]) -> None:
    """Refuse a case that breaks any named contract."""
    broken = [name for name, holds in checks.items() if not holds]
    if broken:
        msg = "it breaks: " + "; ".join(broken)
        raise ValueError(msg)


def _case_quantities(path: Path, case: endpoints.CaseFields, *, keep_fields: bool) -> CaseQoIs:
    """Check one read case against the contract the analysis needs before any interpolation, and return its quantities."""
    outputs = generation.field_tests.outputs
    with h5py.File(path, "r") as handle:
        if case.task == STEADY:
            x = np.asarray(outputs.dataset(handle, "coords/x")[()], dtype=np.float64)
        else:
            diagnostics = outputs.named_values(handle, "global/values", axis=-1)
            f_surf = float(outputs.named_values(handle, "scalar/values")["f_surf"])
            rho_bu_dry = outputs.named_values(handle, "static/fields")["rho_bu_dry"]
    if case.task == STEADY:
        grids = list(case.fields.values())
        _require_contract(
            {
                "equal [Y, X] fields": all(grid.ndim == 2 and grid.shape == grids[0].shape for grid in grids),  # noqa: PLR2004 -- a 2-D grid
                "finite fields": all(np.isfinite(grid).all() for grid in grids),
                "finite, strictly increasing coords/x across the grid": _increasing(x) and x.shape == grids[0].shape[1:],
            }
        )
        return steady_qois(case.fields, x, keep_fields=keep_fields)
    times, states, completion = case.times, case.states, case.completion
    if times is None or states is None or completion is None:
        msg = "it lacks its stored transient states"
        raise ValueError(msg)
    stored = {name: diagnostics[name][: times.size] for name in ("t", *_DIAGNOSTICS)}
    finite = all(np.isfinite(array).all() for array in (states, rho_bu_dry, *stored.values())) and math.isfinite(f_surf + completion.t_stop)
    _require_contract(
        {
            "finite, strictly increasing stored hours": times.size > 0 and _increasing(times),
            "states [N, 4, Y, X] on the stored hours and rho_bu_dry on their grid": states.shape[:2] == (times.size, len(endpoints.TRANSIENT_STATES))
            and rho_bu_dry.shape == states.shape[2:],
            "finite states, diagnostics, f_surf, rho_bu_dry and stop time": finite,
            "non-negative water states": bool((states[:, 2:] >= 0.0).all()),
            "global diagnostics starting with the stored hours": np.allclose(stored["t"], times, rtol=0.0, atol=endpoints.HOUR_ATOL),
        }
    )
    return transient_qois(
        times,
        states,
        {name: stored[name] for name in _DIAGNOSTICS},
        f_surf,
        rho_bu_dry,
        t_stop=completion.t_stop,
        reached=completion.reached,
        target=(completion.target_wet_basis, completion.fraction_limit, completion.horizon),
        keep_fields=keep_fields,
    )


def _read_case(path: Path, expected_sha256: object, task: str, *, keep_fields: bool = False) -> CaseQoIs:
    """
    Read one Source-format ``case.h5`` through the frozen digest-checked reader and return its quantities.

    Raises
    ------
    ValueError
        Naming the case, if its digest or task differs, or it breaks the contract: shapes, finite values, strictly
        increasing stored hours, diagnostics on those hours, or non-negative water states.

    """
    try:
        case = _VERIFIED_CASE(path, expected_sha256)
        quantities = None if case is None or case.task != task else _case_quantities(path, case, keep_fields=keep_fields)
    except (IndexError, KeyError, ValueError) as error:
        msg = f"{path} breaks the case contract the effect analysis reads: {error}"
        raise ValueError(msg) from error
    if quantities is None:
        msg = f"{path} is missing, no longer has its bound digest, or is not a {task} case."
        raise ValueError(msg)
    return quantities


def _member_case(bound: _Bound, member_id: str, *, keep_fields: bool = False) -> CaseQoIs:
    """Read one sealed member's ``case.h5`` with its sealed digest."""
    seal = bound.seals[member_id] or {}
    path = generation.field_tests.record.member_directory(bound.directory, member_id) / "case.h5"
    return _read_case(path, seal.get("case_h5_sha256"), bound.task, keep_fields=keep_fields)


def _train_population(bound: _Bound, *, storage_root: Path | str | None) -> tuple[list[tuple[str, str, CaseQoIs]], dict[str, Any]]:
    """
    Return the material and quantities of every ID Train case of the yardstick runs, and the binding of that membership.

    Raises
    ------
    ValueError
        If a run no longer has the identity the tolerance table records, the runs disagree on their Train membership, a
        saved split contradicts the Dataset manifest's identity or fixed partition, or the per-material Train population
        is not the expected one.

    """
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    runs, parts = [], []
    for recorded in bound.frozen.payload["model_runs"]:
        with experiments.run.saved_artifact_run_lease(storage / str(recorded["path"])) as admitted:
            identity = _RUN_IDENTITY(admitted, storage_root=storage_root)
            split = datasets.preprocessing.splits.admit_split(admitted["split_indices"])
        if identity != dict(recorded):
            msg = f"Run {recorded['path']!r} no longer has the identity the tolerance table records."
            raise ValueError(msg)
        runs.append({**identity, "membership_digest": split.role("train").membership_digest})
        parts.append((split, split.role("train")))
    first = parts[0][1]
    manifest = datasets.packages.load_package_manifest_evidence(parts[0][0].task, first.dataset_id, storage_root=storage_root)
    binding = datasets.contracts.identity.manifest_binding(manifest)
    partition: Mapping[str, Any] = binding.case_partition or {}
    expected = binding.split_identity()
    subset = datasets.contracts.partition.train_subset_cases
    for split, part in parts:
        # The payload identity (stride, representation) may differ between runs; the Source binding and membership may not.
        if (
            part.case_ids != first.case_ids
            or {key: part.dataset.get(key) for key in expected} != expected
            or dict(split.case_partition) != dict(partition)
            or part.case_ids != subset(partition, material_by_case=binding.materials, train_subset=split.train_subset)
        ):
            msg = f"The yardstick runs' saved Train splits disagree, or contradict the Dataset identity or fixed partition of {first.dataset_id!r}."
            raise ValueError(msg)
    counts = Counter(binding.materials[case_id] for case_id in first.case_ids)
    population = TRAIN_POPULATION[bound.task]
    roles = dict(partition["material_roles"])
    if counts != Counter(population) or sorted(material for material, role in roles.items() if role == "id") != sorted(population):
        msg = f"The Train population {dict(counts)} of {first.dataset_id!r} is not the expected ID population {population}."
        raise ValueError(msg)
    by_case = {entry["package_case_id"]: entry["source"] for entry in manifest["source_case_identities"]}
    sources = [by_case[case_id] for case_id in first.case_ids]
    selectors = [{key: source[key] for key in _SOURCE_SELECTOR_KEYS} for source in sources]
    admitted_cases = generation.publication.catalog.resolve_raw_case_references(manifest["source_task"], selectors, storage_root=storage_root)
    cases = [
        (case_id, binding.materials[case_id], _read_case(Path(case.hdf5_path), source["case_hdf5_sha256"], bound.task))
        for case_id, source, case in zip(first.case_ids, sources, admitted_cases, strict=True)
    ]
    membership = {
        "runs": runs,
        "dataset": {key: expected[key] for key in ("dataset_id", "dataset_digest", "channel_contract_digest")},
        "case_partition_sha256": datasets.contracts.partition.partition_digest(partition),
        "material_roles": roles,
        "id_materials": sorted(population),
        "population": dict(sorted(counts.items())),
    }
    return cases, membership


def _attempts(bound: _Bound, member_id: str, rule_minimum: float | None) -> tuple[list[Attempt], dict[str, Any]]:
    """
    Return one member's attempts with both stall-progress fields and typed evidence, and their evidence digests.

    A recorded crossing is evidence only for the frozen stall rule's minimum ``rule_minimum``; for any other minimum, or
    without a frozen minimum, its evidence is unknown with the reason, and it never matches.
    """
    record = generation.field_tests.record
    attempts: list[Attempt] = []
    digests: dict[str, Any] = {}
    for number in record.attempt_numbers(bound.directory, member_id):
        outcome_path = record.attempt_directory(bound.directory, member_id, number) / "outcome.json"
        timing_path = record.record_path(bound.directory, "members", member_id, "attempts", str(number), "runtime", "timing.json")
        digests[str(number)] = [common.serialization.file_sha256(path) if path.is_file() else None for path in (outcome_path, timing_path)]
        outcome = record.read_outcome(bound.directory, member_id, number) or {}
        minimum = crossing = None
        evidence = "missing outcome.json" if not outcome else "missing timing.json"
        if outcome and timing_path.is_file():
            try:
                minimum, crossing, availability = _STALL_PROGRESS(json.loads(timing_path.read_text(encoding="utf-8")))
            except (TypeError, ValueError) as error:
                msg = f"{timing_path} holds no valid stall progress: {error}"
                raise ValueError(msg) from error
            evidence = f"no crossing evidence ({availability})" if crossing is None else CROSSING_CLOCK
            if crossing is not None and minimum != rule_minimum:
                evidence = f"unknown: crossing for a {minimum} s minimum, frozen rule minimum {rule_minimum}"
        draws = (outcome.get("spatial_support_attempt"), outcome.get("schedule_acceptance_attempt"))
        reason = (outcome.get("failure") or {}).get("reason_code")
        classification, cpu, host = (str(outcome["classification"]) if outcome else None), outcome.get("cpu_model"), outcome.get("host")
        attempts.append(Attempt(member_id, number, classification, reason, cpu, host, draws, minimum, crossing, evidence))
    return attempts, digests


def _frozen_campaign(bound: _Bound, *, storage_root: Path | str | None) -> Any:
    """
    Return the base Source's frozen campaign after its plan-digest check, ``None`` only when a frozen file is missing.

    A changed plan (``BaseBindingError``) or any other contract failure propagates: the analysis fails closed.
    """
    try:
        return generation.field_tests.roster.load_base_campaign(bound.task, bound.roster["base_source"], storage_root=storage_root)
    except FileNotFoundError:
        return None


def _stall_check(bound: _Bound, campaign: Any, *, storage_root: Path | str | None) -> dict[str, Any]:
    """
    Return the stall rule of the Source's frozen runtime (check time and minimum), its status and the execution digest.

    Only a Source whose frozen files are missing assumes the 900 s check time, without a minimum; a frozen runtime without
    a stall rule has none, and a reconstruction that does not reproduce its run record's execution digest refuses.
    """
    unset = {"seconds": None, "minimum_simulated_seconds": None, "execution_config_digest": None}
    if bound.task != TRANSIENT:
        return {**unset, "status": "no stall rule for a stationary profile"}
    if campaign is None:
        return {
            **unset,
            "seconds": ASSUMED_CHECK_SECONDS,
            "status": "assumed 900 s: frozen Source campaign not recoverable; without its minimum no crossing counts",
        }
    run_id = f"source~{bound.task}~{bound.roster['base_source']['campaign_name']}"
    run = generation.publication.campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    digest = common.serialization.canonical_json_sha256(campaign.execution_values)
    if digest != run.get("execution_config_digest"):
        msg = f"The frozen Source campaign of {run_id!r} does not reproduce the execution digest its run record binds."
        raise ValueError(msg)
    rule = campaign.execution_values["runtime"].get("stall_stop")
    if rule is None:
        return {**unset, "status": "no stall rule in the frozen Source runtime", "execution_config_digest": digest}
    seconds, minimum = float(rule["check_after_seconds"]), float(rule["minimum_simulated_seconds"])
    return {"seconds": seconds, "minimum_simulated_seconds": minimum, "status": "verified", "execution_config_digest": digest}


def _id_support(campaign: Any, row: str, material: str) -> tuple[float, float, str] | None:
    """Return the frozen registry's ID support ``(lower, upper, unit)`` of one row and material, ``None`` when not unique."""
    if campaign is None:
        return None
    entries = [batch.scientific_values["material"]["parameter_registry"].get(row) for batch in campaign.batches if batch.material_family == material]
    found = {
        (float(entry["lower"]), float(entry["upper"]), str(entry.get("unit")))
        for entry in entries
        if isinstance(entry, dict) and all(isinstance(entry.get(key), (int, float)) for key in ("lower", "upper"))
    }
    return found.pop() if len(found) == 1 else None


def _admit_range_evidence(evidence: RangeEvidence, task: str, rows: Iterable[str], campaign: Any) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    """
    Return the copies of both range-evidence documents and the admitted claims.

    Raises
    ------
    ValueError
        If a document is missing, a claim names no tested row, ID material and interval level, its unit or citation is
        empty, its value is not beyond its ID bound on the level's side, or that bound or unit contradicts the frozen
        Source registry (when the registry holds the row's ID support for that material).

    """
    files = {name: Path(path) for name, path in evidence.files.items()}
    if sorted(files) != sorted(EVIDENCE_DOCUMENTS) or not all(path.is_file() for path in files.values()):
        msg = f"Range evidence needs both documents {list(EVIDENCE_DOCUMENTS)} as files, got {sorted(files)}."
        raise ValueError(msg)
    tested, claims = set(rows), []
    for (row, material, level), claim in sorted(evidence.claims.items()):
        beyond = claim.value > claim.id_bound if level == "high" else claim.value < claim.id_bound
        support = _id_support(campaign, row, material)
        registry = support is None or (
            claim.unit == support[2] and math.isclose(claim.id_bound, support[1] if level == "high" else support[0], rel_tol=INVENTORY_RTOL)
        )
        if (
            row not in tested
            or material not in TRAIN_POPULATION[task]
            or level not in INTERVAL_LEVELS
            or not (claim.unit.strip() and claim.citation.strip() and math.isfinite(claim.value) and math.isfinite(claim.id_bound) and beyond)
            or not registry
        ):
            msg = (
                f"Range evidence claim {(row, material, level)} is not a cited value beyond the ID bound "
                "of a tested row, ID material and interval level, or contradicts the frozen Source registry."
            )
            raise ValueError(msg)
        claims.append({"row": row, "material": material, "level": level, **dataclasses.asdict(claim), "id_bound_checked": support is not None})
    return {f"evidence-{name}": files[name].read_bytes() for name in sorted(files)}, claims


# ----------------------------------------------------------------------------- measurement and reduction


def _floors(bound: _Bound, qois: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return the resolution floor per QoI: the largest gate-versus-published-base difference over the base gates."""
    deltas: dict[str, list[float | None]] = {qoi: [] for qoi in qois}
    for replicate in bound.persisted["replicates"]:
        if replicate["problem"] is not None:
            for qoi in qois:
                deltas[qoi].append(None)
            continue
        base = bound.members[replicate["gate_member_id"]]["base"]
        published = _read_case(bound.source_root / str(base["case_path"]), base["case_sha256"], bound.task)
        replica = _member_case(bound, replicate["gate_member_id"])
        for qoi in qois:
            deltas[qoi].append(paired_delta(qoi, published, replica)[0])
    planned = len(bound.persisted["replicates"])
    return {qoi: resolution_floor(values, planned) for qoi, values in deltas.items()}


def _scales(
    population: Sequence[tuple[str, str, CaseQoIs]], qois: Sequence[str], floors: Mapping[str, Mapping[str, Any]], id_materials: Sequence[str]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Return the within-material and pooled ID scales per QoI, and the planned and observed Train support with reasons."""
    admitted = [(material, case) for _case_id, material, case in population if case.reached is not False]
    groups = {material: [case for name, case in admitted if name == material] for material in id_materials}
    groups[POOLED] = [case for _material, case in admitted]
    scales: dict[str, dict[str, dict[str, Any]]] = {}
    for group, cases in groups.items():
        scales[group] = {}
        for qoi in qois:
            value, count = robust_scale([case.values[qoi] for case in cases if qoi in case.values])
            resolved, reason = scale_status(value, floors[qoi])
            scales[group][qoi] = {"value": value, "n": count, "resolved": resolved, "reason": reason}
    support = {
        "planned": len(population),
        "censored_excluded": len(population) - len(admitted),
        "observed": {qoi: sum(qoi in case.values for _material, case in admitted) for qoi in qois},
        "missing": {qoi: dict(Counter(case.missing[qoi] for _material, case in admitted if qoi in case.missing)) for qoi in qois},
    }
    return scales, support


def _comparison_table(
    bound: _Bound, scales: Mapping[str, Any], floors: Mapping[str, Any], qois: Sequence[str], material_roles: Mapping[str, str]
) -> tuple[list[dict[str, Any]], float | None]:
    """Measure every persisted challenge comparison, and return the patch coverage of the grid; one reference is held at a time."""
    table: list[dict[str, Any]] = []
    cached: tuple[str, CaseQoIs] | None = None
    coverage: float | None = None
    challenges = [item for item in bound.comparisons if item.role == "challenge"]
    for item in sorted(challenges, key=lambda entry: (str(bound.members[entry.member_id]["reference_member_id"]), entry.member_id)):
        member = bound.members[item.member_id]
        reference = bound.members[str(member["reference_member_id"])]
        before = reference["intervention"]
        moved = {name: [before.get(name), value] for name, value in member["intervention"].items() if before.get(name) != value}
        group = scale_group(item.material, material_roles)
        entry: dict[str, Any] = {
            "member_id": item.member_id,
            "row": item.row,
            "level": member["level"],
            "reference": reference["reference_kind"],
            "material": item.material,
            "base": item.base,
            "displacement": moved,
            "problem": item.problem,
            "scale": "within material" if group != POOLED else "against the ID benchmark",
        }
        table.append(entry)
        if item.problem is not None:
            continue
        if cached is None or cached[0] != reference["member_id"]:
            cached = (str(reference["member_id"]), _member_case(bound, str(reference["member_id"]), keep_fields=True))
        ref, var = cached[1], _member_case(bound, item.member_id, keep_fields=True)
        if coverage is None and ref.fields:
            shape = next(iter(ref.fields.values())).shape
            coverage = patch_coverage((int(shape[0]), int(shape[1])))
        entry["effects"] = {}
        for qoi in qois:
            delta, censoring, reason = paired_delta(qoi, ref, var)
            entry["effects"][qoi] = effect_entry(delta, scales[group][qoi], floors[qoi], censoring=censoring, reason=reason)
        entry["locality"] = {
            name: locality_index(ref.fields[name], var.fields[name]) if name in ref.fields and name in var.fields else None
            for name in LOCALITY_FIELDS[bound.task]
        }
        if bound.task == TRANSIENT:
            entry["supplementary"] = {"label": RUN_LEVEL, "reference": dict(ref.supplementary), "variant": dict(var.supplementary)}
        else:
            entry["inputs"] = {
                name: [float(np.max(np.abs(var.inputs[name] - ref.inputs[name]))), float(np.max(np.abs(ref.inputs[name])))] for name in ref.inputs
            }
    return table, coverage


def _phase_support(table: Sequence[Mapping[str, Any]], qois: Sequence[str]) -> dict[str, Any]:
    """Return per QoI the planned and observed paired support of the comparisons, with the reasons for missing pairs by side."""
    result: dict[str, Any] = {}
    for qoi in qois:
        reasons = Counter(
            entry["effects"][qoi]["reason"] if "effects" in entry else entry["problem"]
            for entry in table
            if "effects" not in entry or entry["effects"][qoi]["delta"] is None
        )
        result[qoi] = {"planned": len(table), "observed": len(table) - sum(reasons.values()), "missing": dict(sorted(reasons.items()))}
    return result


def _level_summaries(table: Sequence[Mapping[str, Any]], floors: Mapping[str, Any], qois: Sequence[str]) -> list[dict[str, Any]]:
    """Return per row, level, reference, material and QoI the median d, its supporting bases, challenges and raw median delta."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in table:
        groups[entry["row"], entry["level"], entry["reference"], entry["material"]].append(entry)
    result = []
    for (row, level, reference, material), entries in groups.items():
        per_qoi = {}
        for qoi in qois:
            measured = [(entry["base"], entry["effects"][qoi]) for entry in entries if "effects" in entry]
            deltas = [effect["delta"] for _base, effect in measured if effect["delta"] is not None]
            supported = sorted((base, effect["d"]) for base, effect in measured if effect["d"] is not None)
            median_abs, floor = _median(abs(value) for value in deltas), floors[qoi]
            per_qoi[qoi] = {
                "median_d": _median(d for _base, d in supported),
                "support": len(supported),
                "bases": [base for base, _d in supported],
                "challenges": len(entries),
                "median_delta": _median(deltas),
                "within_resolution": None if median_abs is None else median_abs <= floor["value"] if floor["complete"] else RESOLUTION_UNKNOWN,
            }
        result.append({"row": row, "level": level, "reference": reference, "material": material, "qois": per_qoi})
    return result


def _headlines(table: Sequence[Mapping[str, Any]], qois: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return per row and base the largest d among the row's valid challenges per QoI, with every tied level and reference."""
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in table:
        groups[entry["row"]][entry["base"]].append(entry)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row, bases in groups.items():
        result[row] = {}
        for base, entries in bases.items():
            valid = [entry for entry in entries if "effects" in entry]
            per_qoi = {}
            for qoi in qois:
                found = [(entry["effects"][qoi]["d"], entry) for entry in valid if entry["effects"][qoi]["d"] is not None]
                d = max((value for value, _entry in found), default=None)
                winners = [entry for value, entry in found if value == d]
                effect = winners[0]["effects"][qoi] if winners else None
                per_qoi[qoi] = {
                    "d": d,
                    "winners": sorted([entry["level"], entry["reference"]] for entry in winners),
                    "resolved": bool(effect and effect["scale_resolved"]),
                    "within_resolution": None if effect is None else effect["within_resolution"],
                }
            result[row][base] = {"material": entries[0]["material"], "incomplete": len(valid) != len(entries), "qois": per_qoi}
    return result


def _row_summary(bases: Mapping[str, Mapping[str, Any]], id_materials: Sequence[str], qois: Sequence[str]) -> dict[str, Any]:
    """
    Return per QoI the median and maximum headline d per material and pooled over ID materials, with support and band.

    A median needs one base and is published with its support; a band needs two bases; an empty group is never resolved.
    """
    groups: dict[str, list[str]] = {POOLED: [base for base, entry in bases.items() if entry["material"] in id_materials]}
    for base, entry in bases.items():
        groups.setdefault(entry["material"], []).append(base)
    summary: dict[str, Any] = {}
    for qoi in qois:
        summary[qoi] = {}
        for group, members in groups.items():
            found = [bases[base]["qois"][qoi] for base in members if bases[base]["qois"][qoi]["d"] is not None]
            median = _median(item["d"] for item in found)
            resolved = bool(found) and all(item["resolved"] for item in found)
            summary[qoi][group] = {
                "median": median,
                "max": max((item["d"] for item in found), default=None),
                "planned": len(members),
                "observed": len(found),
                "incomplete_bases": sum(bases[base]["incomplete"] for base in members),
                "resolved": resolved,
                "band": band(median, resolved=resolved) if len(found) >= MIN_SUPPORT else None,
            }
    return summary


def _range_flags(
    row: str, levels: Sequence[Mapping[str, Any]], bases: Mapping[str, Mapping[str, Any]], claims: Mapping[tuple[str, ...], Any], qois: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Return the «Bereich grösser prüfen» and «Frage für neuen Test» flags of one row per level and material (``RANGE_RULES``)."""
    larger: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for level, material in sorted({(item["level"], item["material"]) for item in levels if item["row"] == row}):
        candidates = [item["qois"] for item in levels if (item["row"], item["level"], item["material"]) == (row, level, material)]
        hits, supplied = [], []
        for qoi in qois:
            ranked = [item[qoi] for item in candidates if item[qoi]["support"] >= MIN_SUPPORT]
            best = max((item["median_d"] for item in ranked), default=None)
            if best is None or best < SUBSTANTIAL:
                continue
            hits.append(qoi)
            for item in (item for item in ranked if item["median_d"] == best):
                won = sum(any(winner[0] == level for winner in bases[base]["qois"][qoi]["winners"]) for base in item["bases"])
                if won * 2 >= len(item["bases"]):
                    supplied.append(qoi)
                    break
        claim = claims.get((row, material, level))
        if hits and claim is None:
            questions.append({"level": level, "material": material, "qois": hits})
        elif supplied:
            side = "above the ID maximum" if level == "high" else "below the ID minimum"
            larger.append({"level": level, "material": material, "side": side, "qois": supplied, "claim": claim})
    return {"bereich_groesser_pruefen": larger, "frage_fuer_neuen_test": questions}


def _stall_analysis(bound: _Bound, attempts: Mapping[str, Sequence[Attempt]], check: Mapping[str, Any]) -> dict[str, Any]:
    """Attribute every reproduced failure, flag bases near the stall threshold and rerun the verdict without non-parameter failures."""
    seconds = check["seconds"]
    failures = []
    for item in bound.comparisons:
        if not item.failure_effect:
            continue
        member = bound.members[item.member_id]
        reference_id = str(member["reference_member_id"])
        reference_seal = bound.seals[reference_id] or {}
        draws = (reference_seal.get("spatial_support_attempt"), reference_seal.get("schedule_acceptance_attempt"))
        failing = [attempt for attempt in attempts[item.member_id] if attempt.classification == "solver_failure" and attempt.draws == draws]
        references = [attempt for attempt in attempts[reference_id] if attempt.draws == draws]
        entry = {"member_id": item.member_id, "row": item.row, "level": member["level"], "base": item.base, "reasons": [a.reason for a in failing]}
        failures.append({**entry, "attribution": attribute_failure(failing, references, seconds)})
    by_base: dict[str, list[Attempt]] = defaultdict(list)
    for member_id, member in bound.members.items():
        if member["role"] == "reference":
            by_base[verdicts.base_label(member["base"])].extend(attempts[member_id])
    excluded = {item["member_id"] for item in failures if item["attribution"] != PARAMETER_RELATED}
    persisted_rows = {row["row"]: row for row in bound.persisted["rows"]}
    sensitivity = []
    for row in sorted({str(item.row) for item in bound.comparisons if item.member_id in excluded}):
        if persisted_rows[row]["verdict"] != "keep":
            continue
        kept = [item for item in bound.comparisons if item.row == row and item.member_id not in excluded]
        rerun = _ROW_VERDICT(row, "A", kept, merge_row=False, materials=set(), joint_cleared=False)
        # A failing challenge stays unmeasured, so only a surviving keep trigger keeps the row; anything else is unresolved.
        verdict = "keep" if rerun["verdict"] == "keep" else "unresolved"
        sensitivity.append({"row": row, "verdict_without_failures": verdict, "worst_ratio": rerun["worst_ratio"]})
    return {
        "check": dict(check),
        # A margin is published only where crossings can be judged against it: with the frozen rule's minimum.
        "margin_seconds": None if seconds is None or check.get("minimum_simulated_seconds") is None else STALL_MARGIN * seconds,
        "attempts": [dataclasses.asdict(attempt) for member_id in sorted(attempts) for attempt in attempts[member_id]],
        "reproduced_failures": failures,
        "near_threshold": {base: near_threshold(items, seconds) for base, items in sorted(by_base.items())},
        "sensitivity": sensitivity,
    }


def _reachability(bound: _Bound, table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the steady reachability class of each steady row screened remove, with the evidence of every comparison."""
    limits = bound.frozen.tolerances
    replicates = {item["base"]: item["endpoints"] for item in bound.persisted["replicates"]}
    persisted = {item["member_id"]: item for item in bound.persisted["comparisons"]}
    result: dict[str, Any] = {}
    for row in (item["row"] for item in bound.persisted["rows"] if item["verdict"] == "remove"):
        evidence = []
        for entry in (item for item in table if item["row"] == row):
            rows = {}
            for name, (value, _contributors) in persisted[entry["member_id"]]["measured"].items():
                replicate = replicates.get(entry["base"], {}).get(name, [None])[0]
                tolerance = limits.get(name)
                ratio = value / tolerance if isinstance(value, float) and tolerance is not None and tolerance > 0.0 else None
                rows[name] = [value if isinstance(value, float) else None, ratio, replicate if isinstance(replicate, float) else None]
            evidence.append({"member_id": entry["member_id"], "problem": entry["problem"], "inputs": entry.get("inputs", {}), "endpoints": rows})
        result[row] = {"class": reachability_class(evidence), "comparisons": evidence}
    return {"input_paths": STEADY_INPUT_PATHS, "rows": result}


def _shares(
    challenges: Mapping[str, int], headlines: Mapping[str, Mapping[str, Mapping[str, Any]]], id_materials: Sequence[str], qois: Sequence[str]
) -> dict[str, Any]:
    """Return the raw-row and coordinate share sets of every family per QoI, with each member's challenge count."""
    result: dict[str, Any] = {}
    for family in sorted({family_of(row) for row in challenges}):
        members = sorted(row for row in challenges if family_of(row) == family)
        bases = sorted({base for row in members for base, entry in headlines.get(row, {}).items() if entry["material"] in id_materials})
        counts = {row: challenges[row] for row in members}
        if set(ANISOTROPY_PAIR) <= set(members):
            merged = {row: count for row, count in counts.items() if row not in ANISOTROPY_PAIR}
            merged[ANISOTROPY_COORDINATE] = sum(counts[row] for row in ANISOTROPY_PAIR)
        else:
            merged = counts
        family_result: dict[str, Any] = {"raw": {}, "coordinates": {}}
        for qoi in qois:
            cells = {row: {base: entry["qois"][qoi] for base, entry in headlines.get(row, {}).items() if base in bases} for row in members}
            family_result["raw"][qoi] = family_shares(cells, bases, counts)
            family_result["coordinates"][qoi] = family_shares(coordinate_cells(cells), bases, merged)
        if any(row in FINE_TEXTURE for row in members):
            family_result["note"] = "fine texture: fine_len_rel, fine_ani_x and fine_ani_y give two effective lengths (algebra, decision table 12.1)"
        result[family] = family_result
    return result


def _row_reports(
    bound: _Bound,
    table: Sequence[Mapping[str, Any]],
    headlines: Mapping[str, Mapping[str, Mapping[str, Any]]],
    levels: Sequence[Mapping[str, Any]],
    stall: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    id_materials: Sequence[str],
) -> dict[str, Any]:
    """Return per tested row the frozen status, summaries, triggers, disposition and the merge and range flags."""
    qois = QOIS[bound.task]
    failures = {item.member_id for item in bound.comparisons if item.failure_effect}
    by_key: dict[tuple[str, ...], Mapping[str, Any]] = {(claim["row"], claim["material"], claim["level"]): claim for claim in claims}
    reports: dict[str, Any] = {}
    for verdict_row in bound.persisted["rows"]:
        row = verdict_row["row"]
        persisted = [item for item in bound.persisted["comparisons"] if item["row"] == row]
        triggers = trigger_evidence(persisted, bound.frozen.tolerances, failure=any(item["member_id"] in failures for item in persisted))
        summary = _row_summary(headlines.get(row, {}), id_materials, qois)
        medians = {qoi: (summary[qoi][POOLED]["median"], summary[qoi][POOLED]["resolved"]) for qoi in qois}
        label, reason, gmax = disposition(verdict_row["verdict"], medians, triggers)
        enger = {item["level"] for item in stall["reproduced_failures"] if item["row"] == row and item["attribution"] == PARAMETER_RELATED}
        merge = []
        if row in FINE_TEXTURE:
            merge.append("fine texture: algebraic, three rows give two effective lengths; no stage B needed for the algebra")
        if row in ANISOTROPY_PAIR:
            merge.append("anisotropy pair strength, max_ratio: one effective coordinate; stage B needed before any member is fixed")
        reports[row] = {
            "family": family_of(row),
            "frozen_verdict": verdict_row["verdict"],
            "owner_decision": OWNER_DECISIONS.get(row),
            "challenges": sum(1 for entry in table if entry["row"] == row),
            "triggers": triggers,
            "disposition": label,
            "disposition_reason": reason,
            "gmax": gmax,
            "merge": merge,
            "bereich_enger": sorted(enger & set(INTERVAL_LEVELS)),
            **_range_flags(row, levels, headlines.get(row, {}), by_key, qois),
            "summary": summary,
        }
    return reports


# ----------------------------------------------------------------------------- publication


def _rounded(value: Any) -> Any:
    """Return a JSON value with floats rounded to 12 significant digits and non-finite floats as ``None``."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.{FLOAT_DIGITS}g}") if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, np.bool_)):
        return bool(value) if isinstance(value, np.bool_) else int(value)
    if isinstance(value, dict):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    msg = f"Cannot persist {type(value).__name__} in an effect record."
    raise TypeError(msg)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical file bytes of one JSON record."""
    return (json.dumps(_rounded(dict(payload)), ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _regular_bytes(path: Path) -> bytes | None:
    """Return the bytes of a regular file without following a link; ``None`` when nothing is there."""
    if not path.is_symlink() and not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        msg = f"Effect output is not a regular file: {path}"
        raise ValueError(msg)
    return path.read_bytes()


def _publish_file(path: Path, payload: bytes) -> None:
    """Create one file exclusively; a file already there (an interrupted earlier run) must be regular and hold exactly these bytes."""
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staged")
    common.serialization.atomic_write_bytes(staged, payload)
    try:
        os.link(staged, path)
    except FileExistsError as error:
        if _regular_bytes(path) != payload:
            msg = f"{path} exists with other content; effect outputs are written once and a rerun must reproduce them byte for byte."
            raise FileExistsError(msg) from error
    finally:
        staged.unlink(missing_ok=True)


def _publish(directory: Path, files: Mapping[str, bytes]) -> None:
    """
    Publish one result directory once, ``manifest.json`` last.

    A committed directory (one holding ``manifest.json``) is verified byte for byte, the manifest first, and never
    written; an uncommitted one is filled by exclusive creation in the given order.
    """
    if _regular_bytes(directory / "manifest.json") is not None:
        for name in sorted(files, key=lambda item: item != "manifest.json"):
            if _regular_bytes(directory / name) != files[name]:
                msg = f"{directory / name} differs in a committed result directory: the 12-character prefix collides or the committed result changed."
                raise FileExistsError(msg)
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        _publish_file(directory / common.paths.validate_logical_name(name, label="effect output"), payload)


def _admit_result(task: str, test_id: str, digest: str, *, storage_root: Path | str | None) -> tuple[dict[str, Any], str]:
    """
    Admit one published task result from its canonical directory and return its verified payload and manifest digest.

    Raises
    ------
    ValueError
        If the digest is malformed, the manifest names another result, binds inputs whose canonical hash is not the
        digest, or lacks a required file, or any listed file is not a regular file holding the bytes its manifest records.

    """
    if not _SHA256.fullmatch(digest):
        msg = f"Task result digest {digest!r} is not a full SHA-256."
        raise ValueError(msg)
    test_dir = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    directory = generation.field_tests.record.record_path(test_dir, "analysis", f"effects-{digest[:12]}")
    raw = _regular_bytes(directory / "manifest.json")
    manifest = json.loads(raw) if raw is not None else {}
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        not isinstance(files, dict)
        or (manifest.get("input_sha256"), manifest.get("task"), manifest.get("test_id")) != (digest, task, test_id)
        or not set(TASK_FILES) <= set(files)
    ):
        msg = f"{directory} holds no committed result {digest} of {task} {test_id!r} with {list(TASK_FILES)}."
        raise ValueError(msg)
    if common.serialization.canonical_json_sha256(manifest.get("binding")) != digest:
        msg = f"{directory / 'manifest.json'} records a binding whose canonical hash is not the claimed result {digest}."
        raise ValueError(msg)
    contents = {name: _regular_bytes(directory / common.paths.validate_logical_name(name, label="effect output")) for name in files}
    if any(payload is None or hashlib.sha256(payload).hexdigest() != files[name] for name, payload in contents.items()):
        msg = f"{directory} does not hold the bytes its manifest records."
        raise ValueError(msg)
    # Parse exactly the effects.json bytes whose hash was just verified.
    payload = json.loads(contents["effects.json"] or b"{}")
    expected = {"input_sha256": digest, "task": task, "test_id": test_id, "binding": manifest["binding"]}
    if not isinstance(payload, dict) or {key: payload.get(key) for key in expected} != expected:
        msg = f"{directory / 'effects.json'} does not belong to its manifest."
        raise ValueError(msg)
    return payload, hashlib.sha256(raw or b"").hexdigest()


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """Return a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def _cell(value: Any) -> str:
    """Return one Markdown cell."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(item) for item in value) or "-"
    return str(value).replace("|", "/")


def _markdown(payload: Mapping[str, Any]) -> bytes:
    """Return the human summary of one task's effect analysis."""
    task, rows, binding = payload["task"], payload["rows"], payload["binding"]
    check = binding["stall_check"]
    lines = [
        LABEL,
        "",
        f"# Field-test effects: {payload['test_id']} ({task})",
        "",
        *(f"- {item}" for item in QUALIFICATIONS),
        f"- result `{payload['input_sha256']}`; verdicts `{binding['verdicts_sha256'][:12]}`, tolerances `{binding['tolerances_sha256'][:12]}`, "
        f"roster `{binding['roster_sha256'][:12]}` v{binding['roster_version']}",
        f"- stall check {_cell(check['seconds'])} s: {check['status']} (execution digest {_cell(check['execution_config_digest'])})",
        "",
        "## Scales and resolution floors",
        *_table(
            ["QoI", *(f"{group} scale (n)" for group in payload["scales"]), "floor", "floor complete"],
            (
                [
                    qoi,
                    *(f"{_cell(scale[qoi]['value'])} ({scale[qoi]['n']})" for scale in payload["scales"].values()),
                    *(payload["floors"][qoi][k] for k in ("value", "complete")),
                ]
                for qoi in payload["qois"]
            ),
        ),
        "",
        "## Rows",
        *_table(
            ["Row", "Family", "Frozen", "Disposition", "Gmax", "Patch", "Non-patch", "Failure", "Merge", "Bereich enger", "Bereich grösser", "Frage"],
            (
                [
                    row,
                    item["family"],
                    item["frozen_verdict"] + (f" ({item['owner_decision']})" if item["owner_decision"] else ""),
                    f"{item['disposition']} ({item['disposition_reason']})",
                    item["gmax"],
                    _quantities(item["triggers"]["patch"]) or "-",
                    _quantities(item["triggers"]["non_patch"]) or "-",
                    item["triggers"]["failure"],
                    "; ".join(item["merge"]) or "-",
                    item["bereich_enger"],
                    [f"{flag['level']}/{flag['material']}" for flag in item["bereich_groesser_pruefen"]],
                    [f"{flag['level']}/{flag['material']}" for flag in item["frage_fuer_neuen_test"]],
                ]
                for row, item in rows.items()
            ),
        ),
    ]
    for kind in ("raw", "coordinates"):
        lines += ["", f"## Family shares, {kind} (pooled ID headline median d over the common support)"]
        for family, shares in payload["families"].items():
            for qoi, entry in shares[kind].items():
                cells = ", ".join(f"{member} {_cell(item['share'])} ({item['challenges']} challenges)" for member, item in entry["members"].items())
                lines.append(f"- {family} / {qoi} (common support {entry['common_support']}): {cells}")
    lines += ["", "## Range claims (PARAMETER_EVIDENCE.md)"]
    lines += [
        f"- {claim['row']} {claim['level']} {claim['material']}: {claim['value']:g} {claim['unit']} beyond {claim['id_bound']:g} "
        f"({claim['citation']}); ID bound checked against the frozen Source registry: {claim['id_bound_checked']}"
        for claim in binding["range_claims"]
    ] or ["- none"]
    stall = payload["stall"]
    lines += ["", "## Stall analysis"]
    lines += [
        f"- {item['member_id']} ({item['row']} {item['level']}): {item['attribution']}; {_cell(item['reasons'])}"
        for item in stall["reproduced_failures"]
    ]
    lines += [
        f"- sensitivity {item['row']}: {item['verdict_without_failures']}, worst ratio {_cell(item['worst_ratio'])}" for item in stall["sensitivity"]
    ]
    lines += [f"- near the stall threshold at {base}: {'unknown' if near is None else near}" for base, near in stall["near_threshold"].items()]
    if payload.get("reachability"):
        lines += ["", "## Steady reachability of the rows screened remove"]
        lines += [f"- {row}: {item['class']}" for row, item in payload["reachability"]["rows"].items()]
        lines += [f"- input {name}: {path}" for name, path in payload["reachability"]["input_paths"].items()]
    lines += ["", "## Untested rows (kept on prior evidence, no measured paired effect)", *(f"- {row}" for row in payload["untested_rows"])]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _style(index: int, *, ood: bool) -> dict[str, Any]:
    """Return the marker style of the QoI at ``index``: one of ten colours and one marker per ten QoIs; material OOD only hollows it."""
    return {"color": f"C{index % 10}", "marker": _MARKERS[index // 10 % len(_MARKERS)], "markerfacecolor": "none" if ood else f"C{index % 10}"}


def _figure(payload: Mapping[str, Any]) -> bytes:
    """Draw one panel per family: pooled ID headline median d per QoI with whiskers to the maximum, material-OOD medians hollow."""
    families, qois = list(payload["families"]), payload["qois"]
    figure = Figure(figsize=(10.0, 1.0 + 2.4 * max(len(families), 1)), layout="constrained")
    axes = figure.subplots(max(len(families), 1), 1, squeeze=False)[:, 0]
    for axis, family in zip(axes, families, strict=False):
        members = [row for row, item in payload["rows"].items() if item["family"] == family]
        for position, row in enumerate(members):
            for index, qoi in enumerate(qois):
                groups = payload["rows"][row]["summary"][qoi]
                y = position + (index - len(qois) / 2) * 0.6 / len(qois)
                pooled = groups[POOLED]
                if pooled["median"] is not None:
                    axis.plot([pooled["median"], pooled["max"]], [y, y], color=f"C{index % 10}", linewidth=0.6)
                    axis.plot([pooled["median"]], [y], markersize=2.5, **_style(index, ood=False))
                for group, values in groups.items():
                    if group not in payload["id_materials"] and group != POOLED and values["median"] is not None:
                        axis.plot([values["median"]], [y], markersize=3, **_style(index, ood=True))
        axis.set_yticks(range(len(members)), members, fontsize=6)
        axis.set_xscale("symlog", linthresh=0.01)
        for threshold in (NEGLIGIBLE, SUBSTANTIAL):
            axis.axvline(threshold, color="black", linewidth=0.6, linestyle="--")
        axis.set_title(f"{family} ({LABEL}; hollow: material-OOD bases against the ID benchmark)", fontsize=7)
    axes[-1].set_xlabel("headline d = |delta| / Train robust spread (median, whisker to max)")
    # Every QoI keeps its own colour and marker, whichever rows have values.
    handles = [Line2D([], [], linestyle="none", markersize=3, label=qoi, **_style(index, ood=False)) for index, qoi in enumerate(qois)]
    axes[0].legend(handles=handles, fontsize=5, ncol=4, loc="lower right")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, metadata={"Software": None, "Description": LABEL})
    return buffer.getvalue()


def _recommendations(payloads: Mapping[str, Mapping[str, Any]], manifests: Mapping[str, str]) -> bytes:
    """Return the combined recommendation table: frozen verdicts, combined status, owner decisions, dispositions and flags."""
    order = sorted(payloads)
    rows = sorted({row for payload in payloads.values() for row in payload["rows"]})
    lines = [LABEL, "", "# Field-test effects: combined recommendations", ""]
    lines += [
        f"- {task}: test `{payloads[task]['test_id']}`, result `{payloads[task]['input_sha256']}`, manifest `{manifests[task]}`" for task in order
    ]
    lines += [*(f"- {item}" for item in QUALIFICATIONS), "- «weg» stays a possibility only; stage B per PLAN_223 still has to confirm it", ""]
    header = ["Row", *(f"{task} frozen" for task in order), "Combined status", "Owner decision", *(f"{task} disposition" for task in order)]
    untested = sorted({row for payload in payloads.values() for row in payload["untested_rows"]})
    body = []
    for row in rows:
        held = {task: payloads[task]["rows"][row] for task in order if row in payloads[task]["rows"]}
        # A row kept untested in either task stays; otherwise it may go only when every task screened it remove.
        removed = row not in untested and all(item["frozen_verdict"] == "remove" for item in held.values())
        status = "weg possible (stage B to confirm)" if removed else "bleibt"
        flags = [
            f"{task}: {kind} {flag['level']}/{flag['material']}"
            for task, item in held.items()
            for kind, key in (("Bereich grösser prüfen", "bereich_groesser_pruefen"), ("Frage für neuen Test", "frage_fuer_neuen_test"))
            for flag in item[key]
        ] + [f"{task}: Bereich enger {level}" for task, item in held.items() for level in item["bereich_enger"]]
        merge = sorted({note for item in held.values() for note in item["merge"]})
        cells = [row, *(held[task]["frozen_verdict"] if task in held else "-" for task in order), status, OWNER_DECISIONS.get(row, "-")]
        cells += [*(held[task]["disposition"] if task in held else "-" for task in order), "; ".join(merge) or "-", "; ".join(flags) or "-"]
        body.append(cells)
    lines += _table([*header, "Merge", "Range flags"], body)
    lines += ["", "## Untested rows (kept on prior evidence, no measured paired effect)", *(f"- {row}" for row in untested)]
    return ("\n".join(lines) + "\n").encode("utf-8")


# ----------------------------------------------------------------------------- public entry points


def measure_field_test_effects(task: str, test_id: str, *, range_evidence: RangeEvidence, storage_root: Path | str | None = None) -> dict[str, Any]:
    """
    Measure and publish the exploratory effect analysis of one stage-A field test.

    Parameters
    ----------
    task, test_id : str
        The field test, resolved through the storage path authority.
    range_evidence : RangeEvidence
        Both range-evidence documents (hashed and copied) and the claims of applicable values beyond an ID bound read from
        them; the claims are curated prose evidence and are checked, not parsed, here.
    storage_root : Path | str | None
        Storage root override.

    Returns
    -------
    dict[str, Any]
        ``payload`` (the content of ``effects.json``) and ``result_directory``
        (``04_field_tests/<task>/<test_id>/analysis/effects-<sha12>/``).

    Raises
    ------
    ValueError
        If any binding fails: the inventory, header or row verdicts are not reproduced, a run identity, the Train
        membership or population differs, a case breaks its contract, the range evidence is incomplete or a claim is
        invalid, or a bound file changed.
    FileExistsError
        If the result directory is committed with other bytes, its prefix belongs to another digest, or an uncommitted
        file holds other bytes.

    """
    torch.set_num_threads(_TORCH_THREADS)
    bound = _bind_inventory(task, test_id, storage_root=storage_root)
    campaign = _frozen_campaign(bound, storage_root=storage_root)
    evidence_files, claims = _admit_range_evidence(range_evidence, task, (row["row"] for row in bound.persisted["rows"]), campaign)
    check = _stall_check(bound, campaign, storage_root=storage_root)
    qois = QOIS[task]
    population, membership = _train_population(bound, storage_root=storage_root)
    floors = _floors(bound, qois)
    scales, train_support = _scales(population, qois, floors, membership["id_materials"])
    table, coverage = _comparison_table(bound, scales, floors, qois, membership["material_roles"])
    headlines = _headlines(table, qois)
    levels = _level_summaries(table, floors, qois)
    read = {member_id: _attempts(bound, member_id, check["minimum_simulated_seconds"]) for member_id in bound.members}
    stall = _stall_analysis(bound, {member_id: found for member_id, (found, _digests) in read.items()}, check)
    rows = _row_reports(bound, table, headlines, levels, stall, claims, membership["id_materials"])
    record = generation.field_tests.record
    binding = {
        "task": task,
        "test_id": test_id,
        "roster_sha256": bound.roster["roster_sha256"],
        "roster_version": bound.version,
        "tolerances_sha256": bound.frozen.sha256,
        "verdicts_sha256": bound.verdicts_sha256,
        "train": membership,
        # An unsealed member (no seal.json) binds explicit nulls; its comparison stays in the table with its reason.
        "seals": {
            member_id: [None, None]
            if seal is None
            else [common.serialization.file_sha256(record.member_directory(bound.directory, member_id) / "seal.json"), seal.get("case_h5_sha256")]
            for member_id, seal in sorted(bound.seals.items())
        },
        "attempts": {member_id: digests for member_id, (_found, digests) in read.items()},
        "stall_check": check,
        "range_evidence": {name: hashlib.sha256(payload).hexdigest() for name, payload in evidence_files.items()},
        "range_claims": claims,
        "effect_code_sha256": EFFECTS_PROJECTION.sha256(),
        "definitions_sha256": common.serialization.canonical_json_sha256(json.loads(json.dumps(DEFINITIONS))),
    }
    # The digest is taken over the binding exactly as persisted, so admission recomputes it from the manifest.
    binding = _rounded(binding)
    digest = common.serialization.canonical_json_sha256(binding)
    payload: dict[str, Any] = {
        "label": LABEL,
        "schema_kind": SCHEMA_KIND,
        "task": task,
        "test_id": test_id,
        "binding": binding,
        "input_sha256": digest,
        "qualifications": QUALIFICATIONS,
        "definitions": DEFINITIONS,
        "qois": qois,
        "id_materials": membership["id_materials"],
        "scales": scales,
        "train_support": train_support,
        "floors": floors,
        "patch_coverage": coverage,
        "phase_support": _phase_support(table, qois),
        "comparisons": table,
        "levels": levels,
        "headlines": [{"row": row, "base": base, **entry} for row, bases in headlines.items() for base, entry in bases.items()],
        "rows": rows,
        "families": _shares({row: item["challenges"] for row, item in rows.items()}, headlines, membership["id_materials"], qois),
        "stall": stall,
        "reachability": _reachability(bound, table) if task == STEADY else None,
        "untested_rows": list(generation.field_tests.planning.KEEP_ROWS[task]),
    }
    payload = _rounded(payload)
    directory = record.record_path(bound.directory, "analysis", f"effects-{digest[:12]}")
    files = {"effects.json": _json_bytes(payload), "effects.md": _markdown(payload), "effects_families.png": _figure(payload), **evidence_files}
    manifest = {"label": LABEL, "qualifications": QUALIFICATIONS, "task": task, "test_id": test_id, "input_sha256": digest, "binding": binding}
    manifest["files"] = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    _publish(directory, {**files, "manifest.json": _json_bytes(manifest)})
    return {"payload": payload, "result_directory": directory}


def combine_field_test_effects(results: Sequence[Mapping[str, Any]], *, storage_root: Path | str | None = None) -> Path:
    """
    Combine both published task results into ``04_field_tests/effects/<sha12>/recommendations.md`` and ``manifest.json``.

    Each result names its task, test and full digest (the ``payload`` a measurement returns); the result itself is admitted
    from its canonical directory below the storage root, never from a caller-supplied path. The sha12 is that of the
    SHA-256 over both task ``manifest.json`` file hashes, sorted by task name.

    Returns
    -------
    Path
        The published ``recommendations.md``.

    Raises
    ------
    ValueError
        If the results are not exactly one steady_flow and one transient_drying result, a task result is not committed
        with the bytes its manifest records, or the combined directory is a link or escapes the field-test area.
    FileExistsError
        If the combined directory is committed with other bytes.

    """
    claimed = {str(result["payload"]["task"]): result["payload"] for result in results}
    if len(claimed) != len(results) or sorted(claimed) != sorted((STEADY, TRANSIENT)):
        msg = f"The combined recommendations need exactly one {STEADY} and one {TRANSIENT} result, got {len(results)} for {sorted(claimed)}."
        raise ValueError(msg)
    payloads: dict[str, dict[str, Any]] = {}
    manifests: dict[str, str] = {}
    for task, item in sorted(claimed.items()):
        payloads[task], manifests[task] = _admit_result(task, str(item["test_id"]), str(item["input_sha256"]), storage_root=storage_root)
    digest = common.serialization.canonical_json_sha256([manifests[task] for task in sorted(manifests)])
    area = common.paths.get_storage_root(storage_root=storage_root).expanduser() / common.paths.FIELD_TESTS_AREA_NAME
    target = area / "effects" / digest[:12]
    if any(path.is_symlink() for path in (area, area / "effects", target)) or not target.resolve(strict=False).is_relative_to(area.resolve()):
        msg = f"Combined effect directory {target} is a symbolic link or escapes {common.paths.FIELD_TESTS_AREA_NAME}/."
        raise ValueError(msg)
    recommendations = _recommendations(payloads, manifests)
    tasks = {task: {**{key: payloads[task][key] for key in ("test_id", "input_sha256")}, "manifest_sha256": sha} for task, sha in manifests.items()}
    manifest = {"label": LABEL, "qualifications": QUALIFICATIONS, "input_sha256": digest, "tasks": tasks}
    manifest["files"] = {"recommendations.md": hashlib.sha256(recommendations).hexdigest()}
    _publish(target, {"recommendations.md": recommendations, "manifest.json": _json_bytes(manifest)})
    return target / "recommendations.md"
